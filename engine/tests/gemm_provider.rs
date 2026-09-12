// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The seam a faster dense GEMM is plugged into, exercised without one.
//!
//! The engine cannot test the provider the Python bindings install -- there is
//! no interpreter in this binary, which is the whole reason the hook exists.
//! What it can test is its half of the contract: that a provider is offered the
//! product with the dimensions and the layouts the engine says it will offer,
//! that declining is honoured all the way back to the right answer, and that
//! handling is honoured too.
//!
//! One provider for the whole file, because installing one is deliberately a
//! once-per-process decision. It recognises two contraction lengths and
//! declines everything else, so the tests below can ask for either behaviour
//! and nothing else in this binary is affected. They run one at a time: what
//! each checks is the list of offers the provider saw, and cargo runs tests on
//! several threads into one process.

use engine::ops::linalg::{self, Gemm, GemmProvider, Storage, set_gemm_provider};
use engine::tensor::{DataType, Shape, Tensor, TensorData};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// The contraction length the provider answers to.
const HANDLED_K: usize = 37;
/// The contraction length it records and then declines.
const DECLINED_K: usize = 41;
/// What a handled product writes instead of the product, so a test can tell
/// which side computed the answer.
const SENTINEL: f32 = -12.5;

#[derive(Debug, PartialEq, Eq)]
struct Offer {
    /// How many independent products the request carried. One for an ordinary
    /// matrix product; more when a stack of them is offered whole.
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    lhs_storage: Storage,
    rhs_storage: Storage,
}

fn offers() -> &'static Mutex<Vec<Offer>> {
    static OFFERS: OnceLock<Mutex<Vec<Offer>>> = OnceLock::new();
    OFFERS.get_or_init(|| Mutex::new(Vec::new()))
}

fn f64_calls() -> &'static AtomicUsize {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    &CALLS
}

struct Recorder;

impl Recorder {
    fn record<T>(request: &Gemm<'_, T>) {
        if request.k != HANDLED_K && request.k != DECLINED_K {
            return;
        }
        offers().lock().unwrap().push(Offer {
            batch: request.batch,
            m: request.m,
            k: request.k,
            n: request.n,
            lhs_storage: request.lhs_storage,
            rhs_storage: request.rhs_storage,
        });
    }
}

impl GemmProvider for Recorder {
    fn gemm_f32(&self, request: Gemm<'_, f32>) -> bool {
        Self::record(&request);
        if request.k != HANDLED_K {
            return false;
        }
        request.out.fill(SENTINEL);
        true
    }

    fn gemm_f64(&self, request: Gemm<'_, f64>) -> bool {
        Self::record(&request);
        f64_calls().fetch_add(1, Ordering::Relaxed);
        false
    }
}

/// Install the provider, take the file's lock, and start from no offers.
///
/// The guard is what serialises the tests; a failing one poisons the lock and
/// the next would then fail for the wrong reason, so the poison is stepped
/// over -- one real failure is the report worth reading.
fn begin() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    static INSTALLED: OnceLock<()> = OnceLock::new();

    let guard = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    INSTALLED.get_or_init(|| {
        assert!(
            set_gemm_provider(Box::new(Recorder)),
            "nothing else in this binary may install a provider"
        );
    });
    let _ = taken();
    guard
}

fn taken() -> Vec<Offer> {
    let mut recorded = offers()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    std::mem::take(&mut *recorded)
}

fn tensor_f32(values: Vec<f32>, dims: Vec<usize>) -> Tensor {
    let shape = Shape::new(dims);
    let mut data = TensorData::zeros(shape.numel(), DataType::Float32);
    data.as_f32_slice_mut().unwrap().copy_from_slice(&values);
    Tensor::new(
        Arc::new(data),
        shape,
        DataType::Float32,
        engine::device::Device::cpu(),
        false,
    )
}

fn tensor_f64(values: Vec<f64>, dims: Vec<usize>) -> Tensor {
    let shape = Shape::new(dims);
    let mut data = TensorData::zeros(shape.numel(), DataType::Float64);
    data.as_f64_slice_mut().unwrap().copy_from_slice(&values);
    Tensor::new(
        Arc::new(data),
        shape,
        DataType::Float64,
        engine::device::Device::cpu(),
        false,
    )
}

fn ramp_f32(len: usize) -> Vec<f32> {
    (0..len).map(|i| (i % 7) as f32 - 3.0).collect()
}

fn ramp_f64(len: usize) -> Vec<f64> {
    (0..len).map(|i| (i % 7) as f64 - 3.0).collect()
}

/// The product the engine should have computed, written the slow obvious way.
fn reference(lhs: &[f64], rhs: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut total = 0.0;
            for l in 0..k {
                total += lhs[i * k + l] * rhs[l * n + j];
            }
            out[i * n + j] = total;
        }
    }
    out
}

#[test]
fn a_two_dimensional_product_is_offered_row_major_on_both_sides() {
    let _serial = begin();

    let (m, k, n) = (5, DECLINED_K, 6);
    let lhs = tensor_f32(ramp_f32(m * k), vec![m, k]);
    let rhs = tensor_f32(ramp_f32(k * n), vec![k, n]);
    let out = linalg::matmul(&lhs, &rhs).unwrap();

    assert_eq!(
        taken(),
        vec![Offer {
            batch: 1,
            m,
            k,
            n,
            lhs_storage: Storage::RowMajor,
            rhs_storage: Storage::RowMajor,
        }]
    );

    // Declining is honoured: the engine's own kernel produced the answer.
    let expected = reference(&ramp_f64(m * k), &ramp_f64(k * n), m, k, n);
    for (got, want) in out.data().as_f32_slice().unwrap().iter().zip(&expected) {
        assert!((*got as f64 - want).abs() < 1e-4, "{got} vs {want}");
    }
}

#[test]
fn a_dense_layer_offers_its_weight_transposed_where_it_lies() {
    let _serial = begin();

    // `linear` holds the weight `[out, in]` for a product that wants
    // `[in, out]`, and hands it over that way rather than copying it.
    let (rows, in_features, out_features) = (4, DECLINED_K, 3);
    let input = tensor_f32(ramp_f32(rows * in_features), vec![rows, in_features]);
    let weight = tensor_f32(
        ramp_f32(out_features * in_features),
        vec![out_features, in_features],
    );
    let out = linalg::linear(&input, &weight, None).unwrap();

    assert_eq!(
        taken(),
        vec![Offer {
            batch: 1,
            m: rows,
            k: in_features,
            n: out_features,
            lhs_storage: Storage::RowMajor,
            rhs_storage: Storage::Transposed,
        }]
    );

    let weight_values = ramp_f64(out_features * in_features);
    let mut transposed = vec![0.0; in_features * out_features];
    for r in 0..out_features {
        for c in 0..in_features {
            transposed[c * out_features + r] = weight_values[r * in_features + c];
        }
    }
    let expected = reference(
        &ramp_f64(rows * in_features),
        &transposed,
        rows,
        in_features,
        out_features,
    );
    for (got, want) in out.data().as_f32_slice().unwrap().iter().zip(&expected) {
        assert!((*got as f64 - want).abs() < 1e-4, "{got} vs {want}");
    }
}

#[test]
fn a_dense_layer_backward_offers_both_of_its_products() {
    let _serial = begin();

    // `grad_input = grad_output @ weight`, and `grad_weight = grad_output^T @
    // input` -- the second with `grad_output` handed over transposed where it
    // lies, since a copy of it is the size of the activations.
    // Both the batch and the output width are lengths the recorder answers
    // to, since they are the contractions of the two backward products; the
    // forward's own contraction is `in_features`, which is not, so its offer
    // stays out of the list below.
    let (rows, in_features, out_features) = (DECLINED_K, 4, DECLINED_K);
    let input =
        tensor_f32(ramp_f32(rows * in_features), vec![rows, in_features]).requires_grad_(true);
    let weight = tensor_f32(
        ramp_f32(out_features * in_features),
        vec![out_features, in_features],
    )
    .requires_grad_(true);

    let out = linalg::linear(&input, &weight, None).unwrap();
    engine::ops::reduction::sum(&out, None, false)
        .unwrap()
        .backward(None)
        .unwrap();

    let seen = taken();
    assert_eq!(
        seen,
        vec![
            Offer {
                batch: 1,
                m: rows,
                k: out_features,
                n: in_features,
                lhs_storage: Storage::RowMajor,
                rhs_storage: Storage::RowMajor,
            },
            Offer {
                batch: 1,
                m: out_features,
                k: rows,
                n: in_features,
                lhs_storage: Storage::Transposed,
                rhs_storage: Storage::RowMajor,
            },
        ],
        "grad_input then grad_weight"
    );

    // Declining is honoured on both: `d(sum(x W^T))/dW` is the column sums of
    // `x` repeated down every output row, and `d/dx` the column sums of `W`.
    let input_values = ramp_f64(rows * in_features);
    let weight_values = ramp_f64(out_features * in_features);

    let grad_input = engine::autograd::get_gradient(&input).expect("input gradient");
    for i in 0..rows {
        for c in 0..in_features {
            let want: f64 = (0..out_features)
                .map(|r| weight_values[r * in_features + c])
                .sum();
            let got = grad_input.data().as_f32_slice().unwrap()[i * in_features + c] as f64;
            assert!(
                (got - want).abs() < 1e-4,
                "grad_input[{i}][{c}]: {got} vs {want}"
            );
        }
    }

    let grad_weight = engine::autograd::get_gradient(&weight).expect("weight gradient");
    for r in 0..out_features {
        for c in 0..in_features {
            let want: f64 = (0..rows).map(|i| input_values[i * in_features + c]).sum();
            let got = grad_weight.data().as_f32_slice().unwrap()[r * in_features + c] as f64;
            assert!(
                (got - want).abs() < 1e-4,
                "grad_weight[{r}][{c}]: {got} vs {want}"
            );
        }
    }

    engine::autograd::clear_graph().unwrap();
}

#[test]
fn handling_the_product_is_honoured() {
    let _serial = begin();

    let (m, k, n) = (3, HANDLED_K, 4);
    let lhs = tensor_f32(ramp_f32(m * k), vec![m, k]);
    let rhs = tensor_f32(ramp_f32(k * n), vec![k, n]);
    let out = linalg::matmul(&lhs, &rhs).unwrap();

    assert_eq!(taken().len(), 1);
    // Nonsense, but the provider's nonsense: it said it had computed the
    // product, so nothing recomputed it.
    assert_eq!(out.data().as_f32_slice().unwrap(), vec![SENTINEL; m * n]);
}

#[test]
fn each_dtype_reaches_its_own_entry_point() {
    let _serial = begin();

    let before = f64_calls().load(Ordering::Relaxed);
    let (m, k, n) = (4, DECLINED_K, 4);
    let lhs = tensor_f64(ramp_f64(m * k), vec![m, k]);
    let rhs = tensor_f64(ramp_f64(k * n), vec![k, n]);
    let out = linalg::matmul(&lhs, &rhs).unwrap();

    assert_eq!(f64_calls().load(Ordering::Relaxed), before + 1);
    assert_eq!(taken().len(), 1);

    let expected = reference(&ramp_f64(m * k), &ramp_f64(k * n), m, k, n);
    for (got, want) in out.data().as_f64_slice().unwrap().iter().zip(&expected) {
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
    }
}

#[test]
fn an_integer_product_is_never_offered() {
    let _serial = begin();

    let (m, k, n) = (3, DECLINED_K, 3);
    let shape = Shape::new(vec![m, k]);
    let mut data = TensorData::zeros(shape.numel(), DataType::Int64);
    for (i, slot) in data.as_i64_slice_mut().unwrap().iter_mut().enumerate() {
        *slot = (i % 5) as i64 - 2;
    }
    let lhs = Tensor::new(
        Arc::new(data),
        shape,
        DataType::Int64,
        engine::device::Device::cpu(),
        false,
    );

    let rhs_shape = Shape::new(vec![k, n]);
    let mut rhs_data = TensorData::zeros(rhs_shape.numel(), DataType::Int64);
    for (i, slot) in rhs_data.as_i64_slice_mut().unwrap().iter_mut().enumerate() {
        *slot = (i % 3) as i64 - 1;
    }
    let rhs = Tensor::new(
        Arc::new(rhs_data),
        rhs_shape,
        DataType::Int64,
        engine::device::Device::cpu(),
        false,
    );

    let out = linalg::matmul(&lhs, &rhs).unwrap();
    assert!(
        taken().is_empty(),
        "an integer product has no BLAS to go to"
    );
    assert_eq!(out.data().as_i64_slice().unwrap().len(), m * n);
}

#[test]
fn a_batched_product_is_offered_as_one_request() {
    let _serial = begin();

    // This used to assert the opposite -- that a stack of matrices stayed with
    // the engine, on the reasoning that the batch axis already fills the thread
    // pool with one whole matrix per worker. Measured, that reasoning did not
    // hold: batched products ran at 0.27-0.66x of NumPy, and offering the stack
    // is 1.2-3.1x faster. So they are offered, and offered *whole*: handing
    // them over one matrix at a time would spend the crossing per matrix, which
    // at a batch of 256 costs more than the product does.
    let (batch, m, k, n) = (3, 4, DECLINED_K, 5);
    let lhs = tensor_f32(ramp_f32(batch * m * k), vec![batch, m, k]);
    let rhs = tensor_f32(ramp_f32(batch * k * n), vec![batch, k, n]);
    let out = linalg::matmul(&lhs, &rhs).unwrap();

    assert_eq!(
        taken(),
        vec![Offer {
            batch,
            m,
            k,
            n,
            lhs_storage: Storage::RowMajor,
            rhs_storage: Storage::RowMajor,
        }],
        "one offer carrying the whole stack, not one per matrix"
    );
    assert_eq!(out.shape().dims(), &[batch, m, n]);
}

#[test]
fn a_declined_batch_is_computed_correctly_by_the_engine() {
    let _serial = begin();

    // `DECLINED_K` is recorded and refused, so this exercises the path where
    // the offer is made, turned down, and every matrix in the stack still has
    // to come back right -- including the ones after the first, which is what
    // a wrong batch offset would break.
    let (batch, m, k, n) = (3, 2, DECLINED_K, 2);
    let lhs_values = ramp_f32(batch * m * k);
    let rhs_values = ramp_f32(batch * k * n);
    let lhs = tensor_f32(lhs_values.clone(), vec![batch, m, k]);
    let rhs = tensor_f32(rhs_values.clone(), vec![batch, k, n]);
    let out = linalg::matmul(&lhs, &rhs).unwrap();
    let got = out.data().as_f32_slice().unwrap();

    for b in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let expected: f32 = (0..k)
                    .map(|l| lhs_values[b * m * k + i * k + l] * rhs_values[b * k * n + l * n + j])
                    .sum();
                let actual = got[b * m * n + i * n + j];
                assert!(
                    (actual - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                    "batch {b} element ({i}, {j}): {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn a_handled_batch_writes_the_whole_stack() {
    let _serial = begin();

    // `HANDLED_K` fills the output with the sentinel, so this says the request
    // handed over covers every element of the batch and not just the first
    // matrix -- a `batch` the provider believed but an `out` sized for one
    // product would leave the rest untouched.
    let (batch, m, k, n) = (4, 3, HANDLED_K, 2);
    let lhs = tensor_f32(ramp_f32(batch * m * k), vec![batch, m, k]);
    let rhs = tensor_f32(ramp_f32(batch * k * n), vec![batch, k, n]);
    let out = linalg::matmul(&lhs, &rhs).unwrap();

    let got = out.data().as_f32_slice().unwrap();
    assert_eq!(got.len(), batch * m * n);
    assert!(
        got.iter().all(|v| *v == SENTINEL),
        "the provider was given the whole stack to write"
    );
}

#[test]
fn a_provider_is_installed_once_and_the_second_attempt_says_so() {
    let _serial = begin();
    assert!(
        !set_gemm_provider(Box::new(Recorder)),
        "a second provider must not displace the first"
    );
}
