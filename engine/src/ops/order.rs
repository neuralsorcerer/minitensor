// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Order-preserving integer keys, and when a slice is long enough to sort in
//! parallel.
//!
//! Sorting spends itself on comparisons -- around forty million of them for
//! two million elements -- so the ordering rule belongs in the *data* rather
//! than in the comparison. Every dtype here maps to an unsigned integer whose
//! ascending order is that dtype's ascending order, which turns every
//! comparison into a plain integer one: no branches, no NaN test, no tie-break
//! to fall through to. On two million float32 that was 62ms of comparisons
//! against 18.
//!
//! The tensor sort and `mode` both order data this way, so the two cannot
//! disagree about where a NaN or a negative zero belongs.

/// Below this many elements a single slice is not worth handing to rayon: the
/// split-and-merge overhead outweighs sorting it on one core.
///
/// Deliberately conservative. Measured on four cores the crossover is somewhere
/// between 4k and 8k elements and the two paths are within noise of each other
/// across that range, where the whole sort costs well under a millisecond
/// either way. Setting it here gives up a little between 8k and 16k in exchange
/// for never regressing the small-slice path, which is the one that runs inside
/// a training loop.
pub(crate) const PAR_SORT_MIN_LEN: usize = 1 << 14;

/// The order-preserving unsigned key of a float, as its own width.
///
/// The usual bit trick: flipping the sign bit of a non-negative and every bit
/// of a negative turns IEEE-754's sign-magnitude layout into an unsigned
/// integer that compares the same way. Two families need folding first, or the
/// integer order would say things the float order does not:
///
/// * NaN compares with nothing, and this library sorts it after every number.
///   Its bit patterns straddle the range -- a negative NaN would land below
///   negative infinity -- so all of them are folded to the maximum key.
/// * `-0.0` and `0.0` are equal as floats and have different bit patterns, so
///   `-0.0` is folded to `0.0` and the two keep their input order as any other
///   pair of equals does.
macro_rules! float_key {
    ($name:ident, $float:ty, $unsigned:ty, $signed:ty, $shift:expr) => {
        #[inline(always)]
        pub(crate) fn $name(value: $float) -> $unsigned {
            if value.is_nan() {
                return <$unsigned>::MAX;
            }
            let folded = if value == 0.0 { 0.0 } else { value };
            let bits = folded.to_bits();
            bits ^ (((bits as $signed) >> $shift) as $unsigned | (1 << $shift))
        }
    };
}

float_key!(float_key32, f32, u32, i32, 31);
float_key!(float_key64, f64, u64, i64, 63);

/// The order-preserving unsigned key of a signed integer: shift the range so
/// the most negative value becomes zero.
#[inline(always)]
pub(crate) fn int_key32(value: i32) -> u32 {
    (value as u32) ^ (1 << 31)
}

#[inline(always)]
pub(crate) fn int_key64(value: i64) -> u64 {
    (value as u64) ^ (1 << 63)
}

#[inline(always)]
pub(crate) fn bool_key(value: bool) -> u32 {
    value as u32
}

/// One element to be sorted: its order key above its position in the slice.
///
/// The comparison is what a sort spends itself on -- forty million of them for
/// two million elements -- so the ordering rule is moved *out* of it. Every
/// dtype maps to an unsigned integer whose ascending order is that dtype's
/// ascending order (see [`float_key32`] and its neighbours), the position goes
/// in the low bits, and the sort is then a plain integer comparison with no
/// branches, no NaN test and no tie-break to fall through to.
///
/// It is also what makes the answer deterministic: the position makes every
/// entry distinct, so the total order has no ties for an unstable sort to
/// resolve differently on a different day. `sort(stable=true)` and
/// `sort(stable=false)` therefore give the same answer, and both get the
/// faster sort.
///
/// The branchy three-way comparator this replaced cost 62ms where the integer
/// one costs 18ms on the same two million float32.
pub(crate) trait Entry: Ord + Copy + Send + Sync {
    /// Where this element sat in the slice before ordering.
    fn position(self) -> usize;
}

impl Entry for u64 {
    #[inline(always)]
    fn position(self) -> usize {
        (self & u32::MAX as u64) as usize
    }
}

impl Entry for u128 {
    #[inline(always)]
    fn position(self) -> usize {
        (self & u64::MAX as u128) as usize
    }
}

/// Pack a four-byte key above a position, for a slice no longer than
/// `u32::MAX`. Eight bytes an element where a `(position, value)` pair takes
/// sixteen.
#[inline(always)]
pub(crate) fn pack32(key: u32, position: usize) -> u64 {
    ((key as u64) << 32) | position as u64
}

/// Pack an eight-byte key above a position. The general form: every dtype fits
/// and so does every axis length.
#[inline(always)]
pub(crate) fn pack64(key: u64, position: usize) -> u128 {
    ((key as u128) << 64) | position as u128
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The keys have to reproduce the float order exactly, including the two
    /// places the bit pattern and the numeric order disagree.
    #[test]
    fn float_keys_reproduce_the_float_order() {
        let ladder = [
            f32::NEG_INFINITY,
            -3.5,
            -1.0,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            1.0,
            3.5,
            f32::INFINITY,
        ];
        for pair in ladder.windows(2) {
            let (low, high) = (float_key32(pair[0]), float_key32(pair[1]));
            if pair[0] == pair[1] {
                // The two zeros: equal as floats, so equal as keys, so their
                // input order decides and nothing else can.
                assert_eq!(low, high, "{} and {} keyed apart", pair[0], pair[1]);
            } else {
                assert!(low < high, "{} keyed at or above {}", pair[0], pair[1]);
            }
        }

        // Every NaN, of either sign and any payload, is the one key above
        // every number -- which is where this library sorts them.
        for nan in [
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0xffff_ffff),
        ] {
            assert_eq!(float_key32(nan), u32::MAX);
        }
        assert!(float_key32(f32::INFINITY) < u32::MAX);

        // The same at double width.
        for pair in [
            (f64::NEG_INFINITY, -1.0f64),
            (-1.0, -0.0),
            (0.0, 1.0),
            (1.0, f64::INFINITY),
        ] {
            assert!(float_key64(pair.0) < float_key64(pair.1));
        }
        assert_eq!(float_key64(-0.0), float_key64(0.0));
        assert_eq!(float_key64(f64::NAN), u64::MAX);
        assert_eq!(float_key64(-f64::NAN), u64::MAX);
    }

    #[test]
    fn integer_keys_reproduce_the_integer_order() {
        let ladder = [i32::MIN, -7, -1, 0, 1, 7, i32::MAX];
        for pair in ladder.windows(2) {
            assert!(int_key32(pair[0]) < int_key32(pair[1]));
        }
        assert_eq!(int_key32(i32::MIN), 0);
        assert_eq!(int_key32(i32::MAX), u32::MAX);

        let wide = [i64::MIN, -7, 0, 7, i64::MAX];
        for pair in wide.windows(2) {
            assert!(int_key64(pair[0]) < int_key64(pair[1]));
        }
        assert!(bool_key(false) < bool_key(true));
    }

    /// Both entry widths carry the same position back out.
    #[test]
    fn entries_give_their_position_back() {
        for position in [0usize, 1, 12345, u32::MAX as usize - 1] {
            let narrow = ((float_key32(1.5) as u64) << 32) | position as u64;
            assert_eq!(Entry::position(narrow), position);
        }
        for position in [0usize, 1, 12345, u32::MAX as usize + 1] {
            let wide = ((float_key64(1.5) as u128) << 64) | position as u128;
            assert_eq!(Entry::position(wide), position);
        }
    }
}
