# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Window functions, and the two one-dimensional products that use them.

A window is a shape, not an algorithm: five lines of arithmetic over the sample
positions. They are here rather than left to the caller because the *ends* are
where they differ from each other and from a naive transcription -- a window
meant for a spectrum repeats seamlessly (`periodic=True`, which is what
`torch` defaults to and what an FFT wants) while one meant for filter design is
symmetric about its middle (`periodic=False`, which is what NumPy's `hanning`
and friends give).

`correlate` and `convolve` are the same sliding product read two ways: the
first slides one signal along the other as it is, the second reverses it first.
Both are `conv1d` underneath -- the convolution in machine learning is a
correlation, which is exactly the confusion these two names exist to keep
apart.
"""

from __future__ import annotations

import math as _math

from . import _core as _C
from ._shape import _atleast_tensor

Tensor = _C.Tensor
_F = _C.functional


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


def _degenerate(length: int) -> Tensor | None:
    """A window of one sample or none, which has no shape to speak of.

    One sample is the whole window, so it is 1 rather than whatever the cosine
    happens to give at position zero -- which is what NumPy answers and what
    keeps a windowed single sample equal to itself.
    """

    if length == 0:
        return Tensor.zeros([0], dtype="float64")
    if length == 1:
        return Tensor.full([1], 1.0, dtype="float64")
    return None


def hann_window(length: int, periodic: bool = True) -> Tensor:
    """The raised cosine window: `0.5 - 0.5 * cos(2 * pi * k / N)`."""

    degenerate = _degenerate(length)
    if degenerate is not None:
        return degenerate
    positions, divisor = _window_positions(length, periodic, "hann_window")
    return 0.5 - 0.5 * _F.cos(positions * (2.0 * _math.pi / divisor))


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
    return alpha - beta * _F.cos(positions * (2.0 * _math.pi / divisor))


def blackman_window(length: int, periodic: bool = True) -> Tensor:
    """The Blackman window: two cosine terms, for lower side lobes."""

    degenerate = _degenerate(length)
    if degenerate is not None:
        return degenerate
    positions, divisor = _window_positions(length, periodic, "blackman_window")
    angle = positions * (2.0 * _math.pi / divisor)
    return 0.42 - 0.5 * _F.cos(angle) + 0.08 * _F.cos(angle * 2.0)


def bartlett_window(length: int, periodic: bool = True) -> Tensor:
    """The triangular window: up to one at the middle and back down."""

    degenerate = _degenerate(length)
    if degenerate is not None:
        return degenerate
    positions, divisor = _window_positions(length, periodic, "bartlett_window")
    return 1.0 - _F.abs(positions * (2.0 / divisor) - 1.0)


def kaiser_window(length: int, periodic: bool = True, beta: float = 12.0) -> Tensor:
    """The Kaiser window, whose `beta` trades main-lobe width for side lobes.

    Built on `i0`, the modified Bessel function this library already has, so
    the shape is the definition rather than an approximation of it.
    """

    degenerate = _degenerate(length)
    if degenerate is not None:
        return degenerate
    positions, divisor = _window_positions(length, periodic, "kaiser_window")
    offset = positions * (2.0 / divisor) - 1.0
    inside = _F.sqrt(_F.clamp(1.0 - offset * offset, 0.0, None))
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
    a correlation than for a convolution. That is NumPy's behaviour and it is
    easier to state as an offset than to arrive at by padding.
    """

    signal = _atleast_tensor(first).reshape(-1)
    kernel = _atleast_tensor(second).reshape(-1)
    if signal.shape[0] == 0 or kernel.shape[0] == 0:
        raise ValueError(f"{name} needs two non-empty sequences")
    if mode not in ("full", "same", "valid"):
        raise ValueError(f"{name} takes mode 'full', 'same' or 'valid', got {mode!r}")
    if "float" not in str(signal.dtype):
        signal = signal.astype("float64")
    if str(kernel.dtype) != str(signal.dtype):
        kernel = kernel.astype(str(signal.dtype))

    length, taps = signal.shape[0], kernel.shape[0]
    if flip:
        kernel = _F.flip(kernel, [0])

    # `conv1d` is a correlation, which is why the reversal above is what
    # separates the two names.
    full = _C.functional.conv1d(
        signal.reshape(1, 1, -1), kernel.reshape(1, 1, -1), None, 1, taps - 1, 1, 1
    ).reshape(-1)
    if mode == "full":
        return full

    shortest, longest = min(length, taps), max(length, taps)
    if mode == "valid":
        return _C.functional.narrow(full, 0, shortest - 1, longest - shortest + 1)
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
