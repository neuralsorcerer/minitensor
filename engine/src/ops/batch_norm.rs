// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Batch normalization over a `[N, C, ...]` input in two passes, and its
//! gradient written out.
//!
//! It was a chain of tensor operations -- a mean, a subtraction, a product, a
//! second mean, a sum with `eps`, a square root, a division, the weight and the
//! bias -- each allocating and walking a tensor the size of the input, with the
//! backward composed from all of theirs. Over `[64, 64, 32, 32]` float32 that
//! was 6.6ms forward and 25.1ms with the backward; this reads the input once
//! for each statistic and once to write the answer.
//!
//! The data is a run of `N * C` segments, each the `S` values of one channel of
//! one image, so every pass walks memory in order. The statistics are summed in
//! float64 and in bands of images fixed by the shape alone, so they do not
//! depend on the thread count.

use crate::{
    autograd::{GradientFunction, TensorId, with_grad_fn},
    device::Device,
    error::{MinitensorError, Result},
    ops::map::{par_map_indexed, par_out_chunks},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;

/// The float types the fused kernels take.
pub(crate) trait NormFloat: Copy + Send + Sync + 'static {
    const DTYPE: DataType;
    fn slice(data: &TensorData) -> Option<&[Self]>;
    fn into_data(values: Vec<Self>, device: Device) -> TensorData;
    fn to_f64(self) -> f64;
    fn from_f64(value: f64) -> Self;
}

impl NormFloat for f32 {
    const DTYPE: DataType = DataType::Float32;
    fn slice(data: &TensorData) -> Option<&[Self]> {
        data.as_f32_slice()
    }
    fn into_data(values: Vec<Self>, device: Device) -> TensorData {
        TensorData::from_vec::<f32>(values, DataType::Float32, device)
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl NormFloat for f64 {
    const DTYPE: DataType = DataType::Float64;
    fn slice(data: &TensorData) -> Option<&[Self]> {
        data.as_f64_slice()
    }
    fn into_data(values: Vec<Self>, device: Device) -> TensorData {
        TensorData::from_vec::<f64>(values, DataType::Float64, device)
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        value
    }
}

/// `[N, C, S]`: images, channels and the values of one channel of one image.
#[derive(Clone, Copy)]
struct Layout {
    images: usize,
    channels: usize,
    spatial: usize,
}

impl Layout {
    /// Values a channel's statistics are taken over.
    fn per_channel(self) -> usize {
        self.images * self.spatial
    }
}

/// Elements a task of a pass covers, at the least.
const TASK_ELEMENTS: usize = 1 << 15;

/// `Σ term(a[i], b[i])` across eight float64 lanes, so it vectorizes.
#[inline(always)]
fn lane_sum2<A: Copy, B: Copy>(a: &[A], b: &[B], term: impl Fn(A, B) -> f64) -> f64 {
    let b = &b[..a.len()];
    let mut acc = [0.0f64; 8];
    let (a_blocks, a_rest) = a.as_chunks::<8>();
    let (b_blocks, b_rest) = b.as_chunks::<8>();
    for (a_block, b_block) in a_blocks.iter().zip(b_blocks) {
        for ((lane, &x), &y) in acc.iter_mut().zip(a_block).zip(b_block) {
            *lane += term(x, y);
        }
    }
    for ((lane, &x), &y) in acc.iter_mut().zip(a_rest).zip(b_rest) {
        *lane += term(x, y);
    }
    ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]))
}

/// Per channel, `Σ term(c, a, b)` over every `(a, b)` pair of that channel's
/// values -- `a` and `b` laid out alike, `[N, C, S]` with `S > 1`.
///
/// Summed a band of images at a time and the bands folded in order.
fn channel_sums<T: NormFloat, F: Fn(usize, T, T) -> f64 + Sync>(
    layout: Layout,
    a: &[T],
    b: &[T],
    term: &F,
) -> Vec<f64> {
    let Layout {
        channels, spatial, ..
    } = layout;
    banded(layout, &|sums, at| {
        for (c, sum) in sums.iter_mut().enumerate() {
            let at = at + c * spatial;
            *sum += lane_sum2(&a[at..at + spatial], &b[at..at + spatial], |x, y| {
                term(c, x, y)
            });
        }
    })
    .into_iter()
    .take(channels)
    .collect()
}

/// Per channel, the sum `add(sums, at)` builds up when called with each
/// image's offset `at`, a band of images to a task and the bands folded in
/// order -- so the answer does not depend on the thread count.
///
/// The row form of [`channel_sums`], for `S == 1`: a `[N, C]` input has one
/// value per channel per image, so there is nothing to vectorize along a
/// channel, and `add` adds a whole row of `C` into the accumulators instead,
/// zipped against whatever per-channel values its term needs.
fn banded<F: Fn(&mut [f64], usize) + Sync>(layout: Layout, add: &F) -> Vec<f64> {
    let image = layout.channels * layout.spatial;
    let band = TASK_ELEMENTS.div_ceil(image.max(1)).max(1);
    let partials = par_map_indexed(layout.images.div_ceil(band), &|index| {
        let mut sums = vec![0.0f64; layout.channels];
        for n in index * band..((index + 1) * band).min(layout.images) {
            add(&mut sums, n * image);
        }
        sums
    });
    let mut total = vec![0.0f64; layout.channels];
    for partial in partials {
        for (t, p) in total.iter_mut().zip(partial) {
            *t += p;
        }
    }
    total
}

/// Write `row(out_row, at)` for every image's row of `C` values of `out`, the
/// row starting at element `at`: the `S == 1` form of [`per_segment`].
fn per_row<T: NormFloat, F: Fn(&mut [T], usize) + Sync>(layout: Layout, out: &mut [T], row: &F) {
    let channels = layout.channels.max(1);
    let per_task = TASK_ELEMENTS.div_ceil(channels).max(1) * channels;
    par_out_chunks(out, per_task, &|first, chunk| {
        for (k, out_row) in chunk.chunks_mut(channels).enumerate() {
            row(out_row, first + k * channels);
        }
    });
}

/// Write `segment(c, out, offset)` for every `[S]` segment of `out`, whose
/// channel is `c` and which starts at element `offset` of the buffer.
fn per_segment<T: NormFloat, F: Fn(usize, &mut [T], usize) + Sync>(
    layout: Layout,
    out: &mut [T],
    segment: &F,
) {
    let spatial = layout.spatial.max(1);
    let per_task = TASK_ELEMENTS.div_ceil(spatial).max(1) * spatial;
    par_out_chunks(out, per_task, &|first, chunk| {
        // The channel is carried along rather than recovered per segment,
        // which with one value a segment was two divisions an element.
        let mut c = (first / spatial) % layout.channels;
        for (k, seg) in chunk.chunks_mut(spatial).enumerate() {
            segment(c, seg, first + k * spatial);
            c += 1;
            if c == layout.channels {
                c = 0;
            }
        }
    });
}

/// The statistics a normalization used, and where they came from.
struct Stats {
    mean: Vec<f64>,
    inv_std: Vec<f64>,
    /// Whether they are this batch's, so the gradient flows through them.
    from_batch: bool,
}

/// The running statistics, handed back when the fused kernel declines.
pub(crate) type Declined<'a> = (Option<&'a mut Tensor>, Option<&'a mut Tensor>);

/// The fused `batch_norm`, for a float input whose parameters and running
/// statistics share its dtype. When the input falls outside that, the running
/// statistics come back untouched and the caller takes the composed path,
/// which handles every other case.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_batch_norm<'a>(
    input: &Tensor,
    running_mean: Option<&'a mut Tensor>,
    running_var: Option<&'a mut Tensor>,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    use_batch_stats: bool,
    training: bool,
    momentum: f64,
    eps: f64,
) -> std::result::Result<Result<Tensor>, Declined<'a>> {
    let dtype = input.dtype();
    let same = |t: Option<&Tensor>| t.is_none_or(|t| t.dtype() == dtype);
    if !matches!(dtype, DataType::Float32 | DataType::Float64)
        || input.numel() == 0
        || !same(weight)
        || !same(bias)
        || !same(running_mean.as_deref())
        || !same(running_var.as_deref())
    {
        return Err((running_mean, running_var));
    }
    Ok(match dtype {
        DataType::Float32 => typed::<f32>(
            input,
            running_mean,
            running_var,
            weight,
            bias,
            use_batch_stats,
            training,
            momentum,
            eps,
        ),
        _ => typed::<f64>(
            input,
            running_mean,
            running_var,
            weight,
            bias,
            use_batch_stats,
            training,
            momentum,
            eps,
        ),
    })
}

fn values<T: NormFloat>(tensor: &Tensor, what: &str) -> Result<Vec<f64>> {
    let slice = T::slice(tensor.data()).ok_or_else(|| {
        MinitensorError::internal_error(format!("batch_norm: {what} has the wrong dtype"))
    })?;
    Ok(slice.iter().map(|&v| v.to_f64()).collect())
}

#[allow(clippy::too_many_arguments)]
fn typed<T: NormFloat>(
    input: &Tensor,
    running_mean: Option<&mut Tensor>,
    running_var: Option<&mut Tensor>,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    use_batch_stats: bool,
    training: bool,
    momentum: f64,
    eps: f64,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    let layout = Layout {
        images: dims[0],
        channels: dims[1],
        spatial: dims[2..].iter().product(),
    };
    let x = T::slice(input.data())
        .ok_or_else(|| MinitensorError::internal_error("batch_norm: input has the wrong dtype"))?;
    let device = input.device();
    let count = layout.per_channel();

    let stats = if use_batch_stats {
        let recip = 1.0 / count as f64;
        // Two passes rather than `E[x^2] - E[x]^2`, which loses every digit
        // when the mean dominates the spread.
        let c = layout.channels;
        let sums = if layout.spatial == 1 {
            banded(layout, &|acc, at| {
                for (a, &v) in acc.iter_mut().zip(&x[at..at + c]) {
                    *a += v.to_f64();
                }
            })
        } else {
            channel_sums(layout, x, x, &|_, v, _| v.to_f64())
        };
        let mean: Vec<f64> = sums.into_iter().map(|s| s * recip).collect();
        let squares = if layout.spatial == 1 {
            banded(layout, &|acc, at| {
                for ((a, &v), &m) in acc.iter_mut().zip(&x[at..at + c]).zip(&mean) {
                    let d = v.to_f64() - m;
                    *a += d * d;
                }
            })
        } else {
            channel_sums(layout, x, x, &|c, v, _| {
                let d = v.to_f64() - mean[c];
                d * d
            })
        };
        let var: Vec<f64> = squares.into_iter().map(|s| s * recip).collect();
        if training && let (Some(rm), Some(rv)) = (running_mean, running_var) {
            // The running variance is the unbiased one, `n / (n - 1)` times the
            // biased estimate the normalization itself uses; with one value
            // per channel there is no unbiased variance and it is left as is.
            let correction = if count > 1 {
                count as f64 / (count as f64 - 1.0)
            } else {
                1.0
            };
            let old_mean = values::<T>(rm, "running_mean")?;
            let old_var = values::<T>(rv, "running_var")?;
            let blend = |old: &[f64], new: &[f64], scale: f64| -> Vec<T> {
                old.iter()
                    .zip(new)
                    .map(|(&o, &n)| T::from_f64(o * (1.0 - momentum) + n * scale * momentum))
                    .collect()
            };
            *rm = Tensor::new(
                Arc::new(T::into_data(blend(&old_mean, &mean, 1.0), rm.device())),
                rm.shape().clone(),
                T::DTYPE,
                rm.device(),
                false,
            );
            *rv = Tensor::new(
                Arc::new(T::into_data(blend(&old_var, &var, correction), rv.device())),
                rv.shape().clone(),
                T::DTYPE,
                rv.device(),
                false,
            );
        }
        Stats {
            inv_std: var.iter().map(|&v| 1.0 / (v + eps).sqrt()).collect(),
            mean,
            from_batch: true,
        }
    } else {
        let (Some(rm), Some(rv)) = (running_mean, running_var) else {
            return Err(MinitensorError::internal_error(
                "batch_norm: evaluation without running statistics",
            ));
        };
        Stats {
            mean: values::<T>(rm, "running_mean")?,
            inv_std: values::<T>(rv, "running_var")?
                .into_iter()
                .map(|v| 1.0 / (v + eps).sqrt())
                .collect(),
            from_batch: false,
        }
    };

    let w = weight.map(|w| values::<T>(w, "weight")).transpose()?;
    let b = bias.map(|b| values::<T>(b, "bias")).transpose()?;
    // A missing weight is a gain of one and a missing bias a shift of zero,
    // both exact, so one formula serves every combination.
    let gain = w.clone().unwrap_or_else(|| vec![1.0; layout.channels]);
    let shift = b.unwrap_or_else(|| vec![0.0; layout.channels]);
    let mut out = vec![T::from_f64(0.0); x.len()];
    if layout.spatial == 1 {
        per_row(layout, &mut out, &|out_row, at| {
            let coefficients = stats.mean.iter().zip(&stats.inv_std).zip(&gain).zip(&shift);
            for ((o, &v), (((&m, &scale), &g), &sh)) in
                out_row.iter_mut().zip(&x[at..]).zip(coefficients)
            {
                *o = T::from_f64((v.to_f64() - m) * scale * g + sh);
            }
        });
    } else {
        per_segment(layout, &mut out, &|c, seg, offset| {
            let (mean, scale, g, sh) = (stats.mean[c], stats.inv_std[c], gain[c], shift[c]);
            let source = &x[offset..offset + seg.len()];
            for (o, &v) in seg.iter_mut().zip(source) {
                *o = T::from_f64((v.to_f64() - mean) * scale * g + sh);
            }
        });
    }

    let requires_grad = crate::autograd::is_grad_enabled()
        && (input.requires_grad()
            || weight.is_some_and(|w| w.requires_grad())
            || bias.is_some_and(|b| b.requires_grad()));
    let output = Tensor::new(
        Arc::new(T::into_data(out, device)),
        input.shape().clone(),
        T::DTYPE,
        device,
        requires_grad,
    );
    if !requires_grad {
        return Ok(output);
    }

    let mut input_ids: SmallVec<[TensorId; 3]> = SmallVec::new();
    input_ids.push(input.id());
    if let Some(w) = weight {
        input_ids.push(w.id());
    }
    if let Some(b) = bias {
        input_ids.push(b.id());
    }
    let grad_fn = Arc::new(BatchNormBackward {
        input_ids,
        input_id: input.id(),
        weight_id: weight.map(|w| w.id()),
        bias_id: bias.map(|b| b.id()),
        input: input.detach(),
        weight: w,
        stats,
        input_requires_grad: input.requires_grad(),
        weight_requires_grad: weight.is_some_and(|w| w.requires_grad()),
        bias_requires_grad: bias.is_some_and(|b| b.requires_grad()),
    });
    with_grad_fn(output, grad_fn)
}

/// The gradient of [`fused_batch_norm`].
///
/// With `x̂ = (x - mean) * inv_std` and `y = x̂ * w + b`, per channel over its
/// `M` values: `db = Σ g`, `dw = Σ g x̂`, and for batch statistics
/// `dx = w * inv_std * (g - Σg / M - x̂ * Σ(g x̂) / M)` -- the two sums are the
/// paths through the mean and the variance. Running statistics are constants,
/// so there `dx = w * inv_std * g`.
struct BatchNormBackward {
    input_ids: SmallVec<[TensorId; 3]>,
    input_id: TensorId,
    weight_id: Option<TensorId>,
    bias_id: Option<TensorId>,
    input: Tensor,
    /// The weight, widened, when there is one.
    weight: Option<Vec<f64>>,
    stats: Stats,
    input_requires_grad: bool,
    weight_requires_grad: bool,
    bias_requires_grad: bool,
}

impl BatchNormBackward {
    fn typed<T: NormFloat>(
        &self,
        grad_output: &Tensor,
        gradients: &mut FxHashMap<TensorId, Tensor>,
    ) -> Result<()> {
        let bad = |what: &str| {
            MinitensorError::internal_error(format!("batch_norm backward: bad {what}"))
        };
        let grad = grad_output.contiguous()?;
        let g = T::slice(grad.data()).ok_or_else(|| bad("gradient"))?;
        let x = T::slice(self.input.data()).ok_or_else(|| bad("input"))?;
        let dims = self.input.shape().dims();
        let layout = Layout {
            images: dims[0],
            channels: dims[1],
            spatial: dims[2..].iter().product(),
        };
        let device = self.input.device();
        let Stats {
            mean,
            inv_std,
            from_batch,
        } = &self.stats;

        let need_sums = self.weight_requires_grad
            || self.bias_requires_grad
            || (self.input_requires_grad && *from_batch);
        let channels = layout.channels;
        let (sum_g, sum_gx) = if !need_sums {
            (Vec::new(), Vec::new())
        } else if layout.spatial == 1 {
            (
                banded(layout, &|acc, at| {
                    for (a, &gv) in acc.iter_mut().zip(&g[at..at + channels]) {
                        *a += gv.to_f64();
                    }
                }),
                banded(layout, &|acc, at| {
                    let coefficients = mean.iter().zip(inv_std.iter());
                    for ((a, (&gv, &xv)), (&m, &inv)) in acc
                        .iter_mut()
                        .zip(g[at..at + channels].iter().zip(&x[at..at + channels]))
                        .zip(coefficients)
                    {
                        *a += gv.to_f64() * ((xv.to_f64() - m) * inv);
                    }
                }),
            )
        } else {
            (
                channel_sums(layout, g, g, &|_, v, _| v.to_f64()),
                channel_sums(layout, g, x, &|c, gv, xv| {
                    gv.to_f64() * ((xv.to_f64() - mean[c]) * inv_std[c])
                }),
            )
        };

        let channel_tensor = |values: &[f64]| {
            Tensor::new(
                Arc::new(T::into_data(
                    values.iter().map(|&v| T::from_f64(v)).collect(),
                    device,
                )),
                Shape::new(vec![layout.channels]),
                T::DTYPE,
                device,
                false,
            )
        };

        if self.input_requires_grad {
            let recip = 1.0 / layout.per_channel() as f64;
            let scale: Vec<f64> = match self.weight.as_deref() {
                Some(w) => inv_std.iter().zip(w).map(|(&i, &w)| i * w).collect(),
                None => inv_std.clone(),
            };
            let mut dx = vec![T::from_f64(0.0); x.len()];
            if !*from_batch {
                // Running statistics are constants, so `dx = scale * g`, and
                // with no `x` in it: a NaN input leaves the gradient finite.
                if layout.spatial == 1 {
                    per_row(layout, &mut dx, &|out_row, at| {
                        for ((o, &gv), &sc) in out_row.iter_mut().zip(&g[at..]).zip(&scale) {
                            *o = T::from_f64(sc * gv.to_f64());
                        }
                    });
                } else {
                    per_segment(layout, &mut dx, &|c, seg, offset| {
                        let sc = scale[c];
                        let g = &g[offset..offset + seg.len()];
                        for (o, &gv) in seg.iter_mut().zip(g) {
                            *o = T::from_f64(sc * gv.to_f64());
                        }
                    });
                }
            } else {
                let mean_g: Vec<f64> = sum_g.iter().map(|&s| s * recip).collect();
                let mean_gx: Vec<f64> = sum_gx.iter().map(|&s| s * recip).collect();
                if layout.spatial == 1 {
                    per_row(layout, &mut dx, &|out_row, at| {
                        let coefficients = mean
                            .iter()
                            .zip(inv_std.iter())
                            .zip(&scale)
                            .zip(mean_g.iter().zip(&mean_gx));
                        for ((o, (&gv, &xv)), (((&m, &inv), &sc), (&mg, &mgx))) in out_row
                            .iter_mut()
                            .zip(g[at..].iter().zip(&x[at..]))
                            .zip(coefficients)
                        {
                            let xhat = (xv.to_f64() - m) * inv;
                            *o = T::from_f64(sc * (gv.to_f64() - mg - xhat * mgx));
                        }
                    });
                } else {
                    per_segment(layout, &mut dx, &|c, seg, offset| {
                        let (m, inv, sc) = (mean[c], inv_std[c], scale[c]);
                        let (mg, mgx) = (mean_g[c], mean_gx[c]);
                        let g = &g[offset..offset + seg.len()];
                        let xs = &x[offset..offset + seg.len()];
                        for ((o, &gv), &xv) in seg.iter_mut().zip(g).zip(xs) {
                            let xhat = (xv.to_f64() - m) * inv;
                            *o = T::from_f64(sc * (gv.to_f64() - mg - xhat * mgx));
                        }
                    });
                }
            }
            let dx = Tensor::new(
                Arc::new(T::into_data(dx, device)),
                self.input.shape().clone(),
                T::DTYPE,
                device,
                false,
            );
            crate::autograd::accumulate_grad(gradients, self.input_id, dx)?;
        }
        if self.weight_requires_grad
            && let Some(id) = self.weight_id
        {
            crate::autograd::accumulate_grad(gradients, id, channel_tensor(&sum_gx))?;
        }
        if self.bias_requires_grad
            && let Some(id) = self.bias_id
        {
            crate::autograd::accumulate_grad(gradients, id, channel_tensor(&sum_g))?;
        }
        Ok(())
    }
}

impl GradientFunction for BatchNormBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        match grad_output.dtype() {
            DataType::Float32 => self.typed::<f32>(grad_output, &mut gradients)?,
            DataType::Float64 => self.typed::<f64>(grad_output, &mut gradients)?,
            _ => {
                return Err(MinitensorError::internal_error(
                    "batch_norm backward expects a floating point gradient",
                ));
            }
        }
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}
