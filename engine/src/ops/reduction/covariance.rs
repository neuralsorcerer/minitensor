// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Covariance and correlation matrices, each entry rounded once; see
//! `arithmetic::exact_sum::covariances`.

use std::sync::Arc;

use crate::{
    error::{MinitensorError, Result},
    ops::arithmetic::exact_sum::{Output, Refusal, Sample, covariances},
    tensor::{DataType, Shape, Tensor, TensorData},
};

/// The covariance matrix of the rows of the 2-D `input` -- each row a
/// variable, each column an observation -- or with `correlation` their
/// correlation matrix, every entry the float nearest the exact one.
///
/// `frequencies` counts each observation and `reliabilities` weighs it,
/// float64 and one each, and the divisor is the weights' total less
/// `correction`, or with reliabilities less `correction` times their
/// weighted mean. A float32 or int32 input gives a float32 matrix and any
/// other float64. An integer row is taken as its differences from its
/// first value, exactly; past `2^53` apart, as two floats a value. No
/// gradient: `cov` and `corrcoef` give it.
pub fn covariance_matrix(
    input: &Tensor,
    correction: f64,
    frequencies: Option<&Tensor>,
    reliabilities: Option<&Tensor>,
    correlation: bool,
) -> Result<Tensor> {
    let name = if correlation { "corrcoef" } else { "cov" };
    if input.ndim() != 2 {
        return Err(MinitensorError::invalid_operation(format!(
            "{name} takes a 2-D tensor of rows, got {} dimensions",
            input.ndim()
        )));
    }
    let input = input.contiguous()?;
    let (variables, observations) = (input.shape().dims()[0], input.shape().dims()[1]);
    let missing = || MinitensorError::internal_error("covariance_matrix: a typed slice");
    let (values, lows): (Vec<f64>, Option<Vec<f64>>) = match input.dtype() {
        DataType::Float32 => {
            let values = input.data().as_f32_slice().ok_or_else(missing)?;
            (values.iter().map(|&value| f64::from(value)).collect(), None)
        }
        DataType::Float64 => (
            input.data().as_f64_slice().ok_or_else(missing)?.to_vec(),
            None,
        ),
        DataType::Int32 => {
            let values = input.data().as_i32_slice().ok_or_else(missing)?;
            (values.iter().map(|&value| f64::from(value)).collect(), None)
        }
        DataType::Int64 => {
            let values = input.data().as_i64_slice().ok_or_else(missing)?;
            integer_rows(values, observations)
        }
        other => {
            return Err(MinitensorError::invalid_operation(format!(
                "{name} takes a float or integer tensor, got {other:?}"
            )));
        }
    };
    let weights = |tensor: Option<&Tensor>, what: &str| -> Result<Option<Vec<f64>>> {
        let Some(tensor) = tensor else {
            return Ok(None);
        };
        if tensor.ndim() != 1 || tensor.numel() != observations {
            return Err(MinitensorError::invalid_argument(format!(
                "{name} requires {what} to be 1-D with one entry per observation \
                 ({observations})"
            )));
        }
        let tensor = tensor.astype(DataType::Float64)?.contiguous()?;
        Ok(Some(
            tensor.data().as_f64_slice().ok_or_else(missing)?.to_vec(),
        ))
    };
    let frequencies = weights(frequencies, "fweights")?;
    let reliabilities = weights(reliabilities, "aweights")?;
    let sample = Sample {
        values: &values,
        lows: lows.as_deref(),
        variables,
        observations,
        frequencies: frequencies.as_deref(),
        reliabilities: reliabilities.as_deref(),
        correction,
    };
    let dtype = match input.dtype() {
        DataType::Float32 | DataType::Int32 => DataType::Float32,
        _ => DataType::Float64,
    };
    let mut data =
        TensorData::uninitialized_on_device(variables * variables, dtype, input.device());
    let refused = if dtype == DataType::Float32 {
        fill(
            &sample,
            correlation,
            data.as_f32_slice_mut().ok_or_else(missing)?,
        )
    } else {
        fill(
            &sample,
            correlation,
            data.as_f64_slice_mut().ok_or_else(missing)?,
        )
    };
    match refused {
        Ok(()) => {}
        Err(Refusal::NoWeight) => {
            return Err(MinitensorError::invalid_argument(format!(
                "{name} requires the weights to sum to a positive value"
            )));
        }
        Err(Refusal::NoDivisor) => {
            return Err(MinitensorError::invalid_argument(format!(
                "{name}: the correction {correction} leaves a non-positive divisor for \
                 {observations} observations"
            )));
        }
    }
    Ok(Tensor::new(
        Arc::new(data),
        Shape::new(vec![variables, variables]),
        dtype,
        input.device(),
        false,
    ))
}

fn fill<T: Output>(
    sample: &Sample,
    correlation: bool,
    out: &mut [T],
) -> std::result::Result<(), Refusal> {
    covariances(sample, correlation, out)
}

/// Rows of int64s as their differences from each row's first value, which
/// move no covariance, as exact float64s: one a value where every difference
/// in the matrix is under `2^53`, and otherwise the multiple of `2^32` at or
/// below it and what is left, both exact.
fn integer_rows(values: &[i64], observations: usize) -> (Vec<f64>, Option<Vec<f64>>) {
    let differences = |row: &[i64]| -> Vec<i128> {
        row.iter()
            .map(|&value| i128::from(value) - i128::from(row[0]))
            .collect::<Vec<_>>()
    };
    let rows: Vec<i128> = values
        .chunks(observations.max(1))
        .flat_map(differences)
        .collect();
    let limit = 1i128 << 53;
    if rows.iter().all(|&difference| difference.abs() < limit) {
        return (
            rows.iter().map(|&difference| difference as f64).collect(),
            None,
        );
    }
    let low = |difference: i128| difference.rem_euclid(1 << 32);
    let highs = rows
        .iter()
        .map(|&difference| (difference - low(difference)) as f64)
        .collect();
    let lows = rows
        .iter()
        .map(|&difference| low(difference) as f64)
        .collect();
    (highs, Some(lows))
}
