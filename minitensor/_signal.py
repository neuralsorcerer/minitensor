# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Window functions, and the two one-dimensional products that use them.

A window is a shape, not an algorithm: five lines of arithmetic over the sample
positions. They are here rather than left to the caller because the *ends* are
where they differ from each other and from a naive transcription -- a window
meant for a spectrum repeats seamlessly (`periodic=True`, the default and
what an FFT wants) while one meant for filter design is symmetric about its
middle (`periodic=False`).

`correlate` and `convolve` are the same sliding product read two ways: the
first slides one signal along the other as it is, the second reverses it first.
Both are `conv1d` underneath -- the convolution in machine learning is a
correlation, which is exactly the confusion these two names exist to keep
apart.
"""

from __future__ import annotations

import math as _math

import numpy as _np

from . import _core as _C
from ._shape import (
    _flat,
    _from_numpy,
    _numpy_view,
    _operand_pair,
    _promote_pair,
)

Tensor = _C.Tensor
_F = _C.functional

# An untracked float64 dot of at least this many samples is NumPy's. Its BLAS
# splits one that long over threads that wake in microseconds, where ours stays
# on one thread because a round trip through the pool costs tens: 13us against
# its 9 at 65,536 samples, even at 32,768, and 3.5 against 5.3 at 16,384 in
# ours' favour. A float32 BLAS dot stays on one thread: a few microseconds
# ahead up to 131,072 samples (10-18us against 17-24 there), level at 262,144
# (about 45 each), and half as fast from 524,288 on (130-185 against 70-97), so
# a float32 dot stays ours at every length.
_NUMPY_DOT_FROM = 1 << 15


def _window_positions(length: int, periodic: bool, name: str) -> tuple[Tensor, float]:
    """The sample positions `0..length-1` and the divisor the window uses.

    A periodic window divides by `length` so that the last sample is one step
    short of repeating the first; a symmetric one divides by `length - 1` so
    that the two ends coincide. That single choice is the whole difference
    between the two families.
    """

    if length < 0:
        raise ValueError(f"{name} requires a non-negative length, got {length}")
    positions = Tensor.arange(0, length, 1, dtype="float64")
    divisor = float(length if periodic else max(length - 1, 1))
    return positions, divisor


def _reflected(positions: Tensor, divisor: float) -> Tensor:
    """Each position's distance from the nearer end of the period.

    A window is symmetric -- sample `k` and sample `divisor - k` are the same
    number -- and a cosine of `k * (2 * pi / divisor)` is not quite: the angle
    to the far sample rounds differently from the near one, and seven pairs
    in ten came out an ulp or more apart. Both samples are taken from the
    same whole number here, which is exact.
    """

    return _F.minimum(positions, divisor - positions)


def _degenerate(length: int) -> Tensor | None:
    """A window of one sample or none, which has no shape to speak of.

    One sample is the whole window, so it is 1 rather than whatever the cosine
    happens to give at position zero -- which keeps a windowed single sample
    equal to itself.
    """

    if length == 0:
        return Tensor.zeros([0], dtype="float64")
    if length == 1:
        return Tensor.full([1], 1.0, dtype="float64")
    return None


def _sine_squared(positions: Tensor, divisor: float) -> Tensor:
    """`sin(pi * k / N) ** 2`, which is `(1 - cos(2 * pi * k / N)) / 2`.

    The angle is measured from whichever of the end, the quarter and the
    middle is nearest, in whole numbers, before pi is applied: `N - 4k` and
    `N - 2k` are exact, so the rounding of pi and of the product is relative
    to a small angle and not to `pi / 2`. Up to an eighth of the period it
    is `sin(pi k / N) ** 2`, to three eighths `(1 - sin(pi (N - 4k) / 2N)) / 2`
    and past that `cos(pi (N - 2k) / 2N) ** 2`. A cosine near one less one
    kept a handful of correct digits -- a sample a millionth from zero was
    off in its sixth -- and the quarter, whose angle rounded short of
    `pi / 2`, came out `0.49999999999999994` rather than `0.5`.
    """

    reflected = _reflected(positions, divisor)
    unit = _math.pi / divisor
    half_unit = _math.pi / (2.0 * divisor)
    sine = _F.sin(reflected * unit)
    end = sine * sine
    quarter = 0.5 - 0.5 * _F.sin((divisor - 4.0 * reflected) * half_unit)
    cosine = _F.cos((divisor - 2.0 * reflected) * half_unit)
    middle = cosine * cosine
    eighths = reflected * 8.0
    return _F.where(
        eighths <= divisor,
        end,
        _F.where(eighths <= 3.0 * divisor, quarter, middle),
    )


def hann_window(length: int, periodic: bool = True) -> Tensor:
    """The raised cosine window: `0.5 - 0.5 * cos(2 * pi * k / N)`."""

    degenerate = _degenerate(length)
    if degenerate is not None:
        return degenerate
    positions, divisor = _window_positions(length, periodic, "hann_window")
    return _sine_squared(positions, divisor)


def hamming_window(
    length: int, periodic: bool = True, alpha: float = 0.54, beta: float = 0.46
) -> Tensor:
    """The Hamming window, `alpha - beta * cos(2 * pi * k / N)`.

    The default coefficients are the classical ones, which do not quite reach
    zero at the ends -- that is the point of them, and why this is not a Hann
    window with different numbers.
    """

    degenerate = _degenerate(length)
    if degenerate is not None:
        return degenerate
    positions, divisor = _window_positions(length, periodic, "hamming_window")
    # `cos(2 pi k / N)` is `1 - 2 sin(pi k / N) ** 2`.
    return alpha - beta * (1.0 - 2.0 * _sine_squared(positions, divisor))


def blackman_window(length: int, periodic: bool = True) -> Tensor:
    """The Blackman window: two cosine terms, for lower side lobes.

    `0.42 - 0.5 * cos(t) + 0.08 * cos(2t)` is `s * (0.36 + 0.64 * s)` for
    `s = sin(t / 2) ** 2`, with the coefficients the decimals they are. The
    three terms cancel at the ends, where the window is zero, and summed as
    floats they left -1.4e-17 there and few correct digits beside it.
    """

    degenerate = _degenerate(length)
    if degenerate is not None:
        return degenerate
    positions, divisor = _window_positions(length, periodic, "blackman_window")
    square = _sine_squared(positions, divisor)
    return square * (0.36 + 0.64 * square)


def bartlett_window(length: int, periodic: bool = True) -> Tensor:
    """The triangular window: up to one at the middle and back down.

    Sample `k` is `(N - |2k - N|) / N`, whose numerator is a whole number, so
    it is rounded once. `1 - |k * (2 / N) - 1|` rounded up to four times, and
    half the samples were off.
    """

    degenerate = _degenerate(length)
    if degenerate is not None:
        return degenerate
    positions, divisor = _window_positions(length, periodic, "bartlett_window")
    return (divisor - _F.abs(2.0 * positions - divisor)) / divisor


def kaiser_window(length: int, periodic: bool = True, beta: float = 12.0) -> Tensor:
    """The Kaiser window, whose `beta` trades main-lobe width for side lobes.

    Built on `i0`, the modified Bessel function this library already has, so
    the shape is the definition rather than an approximation of it.
    """

    degenerate = _degenerate(length)
    if degenerate is not None:
        return degenerate
    positions, divisor = _window_positions(length, periodic, "kaiser_window")
    # `sqrt(1 - x**2)` for the offset `x = (2k - N) / N` from the middle is
    # `2 * sqrt(k * (N - k)) / N`, from a product of whole numbers. Near the
    # ends `1 - x**2` cancelled to a few correct digits, and the rounded
    # offsets of mirrored samples were not each other's negation.
    inside = 2.0 * _F.sqrt(positions * (divisor - positions)) / divisor
    # The normaliser is built at double precision: `i0` of a float32 `beta`
    # differs from `i0` of the double in the seventh digit, which is visible in
    # the window's peak.
    peak = _F.i0(Tensor.full([1], float(beta), dtype="float64"))
    return _F.i0(inside * beta) / float(peak.item())


def _sliding(first: object, second: object, mode: str, flip: bool, name: str) -> Tensor:
    """The sliding product of two 1-D signals, correlated or convolved.

    Every overlap is computed once -- the `'full'` answer -- and the narrower
    modes are a window onto it. Where that window starts is the only place the
    two operations differ once the reversal is done, and it is not symmetric:
    with a kernel longer than the signal, `'same'` starts one sample later for
    a correlation than for a convolution. The untracked path's NumPy product
    answers that way and the tracked one has to agree, and it is easier to
    state as an offset than to arrive at by padding.
    """

    signal, kernel = (_flat(side) for side in _operand_pair(first, second))
    if signal.shape[0] == 0 or kernel.shape[0] == 0:
        raise ValueError(f"{name} needs two non-empty sequences")
    if mode not in ("full", "same", "valid"):
        raise ValueError(f"{name} takes mode 'full', 'same' or 'valid', got {mode!r}")
    # `conv1d` is a float kernel, so an integer side becomes one -- `float64`,
    # which holds every `int32` and every `int64` below 2**53 exactly. What
    # the two sides must not do is meet at *one* of their dtypes: a `float64`
    # kernel against a `float32` signal used to be rounded to `float32`
    # before the multiply, losing precision the caller had asked for.
    signal, kernel = _promote_pair(signal, kernel)
    if "float" not in str(signal.dtype):
        signal = signal.astype("float64")
        kernel = kernel.astype("float64")

    length, taps = signal.shape[0], kernel.shape[0]
    tracked = signal.requires_grad or kernel.requires_grad
    if mode == "valid" and length == taps:
        # One overlap, which is one dot product. NumPy's sliding product
        # copies an input it cannot write to, and the views below are
        # read-only by design: two 8MB copies made a 1M-sample correlation
        # 4.9ms where the dot it amounts to takes 0.2. Its `dot` copies a
        # reversed view as well, so a convolution reverses into a buffer of
        # its own.
        if flip:
            kernel = _F.flip(kernel, [0])
        if tracked or length < _NUMPY_DOT_FROM or str(signal.dtype) != "float64":
            return _F.dot(signal, kernel).reshape(1)
        return _from_numpy(
            _np.array([_np.dot(_numpy_view(signal), _numpy_view(kernel))])
        )

    if not tracked:
        # Nothing to differentiate, so NumPy's own sliding product answers,
        # over views and in the dtype promoted above. `conv1d` goes through
        # its general convolution machinery for a single channel, 5-70x
        # NumPy's time on these shapes: 1.9ms against 0.3 for 16,384
        # samples and 64 taps.
        product = _np.convolve if flip else _np.correlate
        return _from_numpy(product(_numpy_view(signal), _numpy_view(kernel), mode))

    if flip:
        kernel = _F.flip(kernel, [0])

    # `conv1d` is a correlation, which is why the reversal above is what
    # separates the two names. It pads both ends by the same amount, and
    # that amount decides where its answer starts.
    def overlaps(padding: int) -> Tensor:
        return _C.functional.conv1d(
            signal.reshape(1, 1, -1), kernel.reshape(1, 1, -1), None, 1, padding, 1, 1
        ).reshape(-1)

    shortest, longest = min(length, taps), max(length, taps)
    if mode == "valid":
        # The full overlaps run from `1 - taps` to `length - 1`; the valid
        # ones are the `longest - shortest + 1` starting at `shortest - taps`,
        # which padding by `taps - shortest` on each side yields exactly.
        # Computing them all and keeping these did `length + taps - 1` dot
        # products for as few as one: `correlate` of two 1,024-sample
        # signals took 1.9ms.
        return overlaps(taps - shortest)
    full = overlaps(taps - 1)
    if mode == "full":
        return full
    start = (shortest - 1) // 2 if flip or length >= taps else length // 2
    return _C.functional.narrow(full, 0, start, longest)


def correlate(input: object, other: object, mode: str = "valid") -> Tensor:
    """The sliding dot product of two 1-D sequences, without reversing either.

    `mode` says how much overlap counts: `'valid'` only where they fully
    overlap, `'same'` an answer as long as the longer input, `'full'` every
    overlap including the ones hanging off the ends.
    """

    return _sliding(input, other, mode, False, "correlate")


def convolve(input: object, other: object, mode: str = "full") -> Tensor:
    """The convolution of two 1-D sequences: `correlate` with one reversed.

    The reversal is the whole difference, and it is why the machine-learning
    "convolution" is really a correlation -- this is the arithmetic that name
    originally meant.
    """

    return _sliding(input, other, mode, True, "convolve")
