# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A count no machine can meet is an exception, not a crash.

Found by calling every public function with extreme arguments. Three of these
aborted the interpreter outright -- a refused allocation in Rust does not
raise, it ends the process -- and two panicked:

* `chunk(empty, 2**40)`: an empty axis divides evenly into any number of
  pieces, and the list of 2^40 piece sizes could not be allocated.
* `repeat_interleave(bools, 2**40)` through the functional form, which skipped
  the size check the method made, and the total of the repeats was unchecked
  besides, so `2**63` repeats of three elements could wrap.
* `local_response_norm(x, size=2**63)` padded the channel axis out by the
  whole window, a shape whose element count overflowed.
* `cosine_embedding_loss` of two 0-d inputs asked an empty shape for its
  feature axis.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

F = mt.functional


def test_a_chunk_count_no_machine_can_hold_raises():
    with pytest.raises(MemoryError, match="pieces"):
        mt.chunk(mt.zeros((0,)), 2**40)
    assert [tuple(p.shape) for p in mt.chunk(mt.zeros((0,)), 3)] == [(0,)] * 3


@pytest.mark.parametrize("form", ["function", "functional", "method"])
def test_a_repeat_count_no_machine_can_hold_raises(form):
    bools = mt.Tensor([True, False])
    call = {
        "function": lambda r: mt.repeat_interleave(bools, r),
        "functional": lambda r: F.repeat_interleave(bools, r),
        "method": lambda r: bools.repeat_interleave(r),
    }[form]
    with pytest.raises(MemoryError):
        call(2**40)
    with pytest.raises(ValueError, match="count"):
        call(2**63)


def test_a_pad_no_machine_can_hold_raises():
    with pytest.raises((ValueError, MemoryError)):
        F.pad(mt.zeros((1, 2)), (2**62, 2**62))


def test_a_response_window_wider_than_the_channels_is_the_formula():
    x = np.random.default_rng(113).normal(0, 1, (2, 3, 4))

    def reference(size, alpha=0.5, beta=0.75, k=1.0):
        out = np.empty_like(x)
        for c in range(x.shape[1]):
            low, high = max(c - size // 2, 0), min(c + (size - 1) // 2, x.shape[1] - 1)
            energy = (x[:, low : high + 1] ** 2).sum(1) / size
            out[:, c] = x[:, c] / (k + alpha * energy) ** beta
        return out

    for size in (3, 5, 6, 100, 2**63):
        got = F.local_response_norm(mt.Tensor(x, dtype="float64"), size, alpha=0.5)
        np.testing.assert_allclose(got.numpy(), reference(size), rtol=1e-14)


def test_a_cosine_of_0d_inputs_is_refused_with_a_reason():
    scalar = mt.Tensor(2.5)
    with pytest.raises(ValueError, match="0-d"):
        F.cosine_embedding_loss(scalar, scalar, mt.Tensor(1.0))
