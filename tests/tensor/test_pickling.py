# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A tensor pickles, and so copies, as its values and its `requires_grad` flag.

`pickle.dumps(tensor)` and `copy.deepcopy(tensor)` both raised `TypeError`,
which ruled out sending a tensor to a worker process, returning one from it, or
keeping a copy of one inside any structure that is itself deep-copied.

The copy is a new leaf with the same values, dtype and shape. Its history and
its `.grad` stay behind: the graph that produced a tensor belongs to the run it
was recorded in, and a gradient is only meaningful beside that graph.
"""

from __future__ import annotations

import copy
import multiprocessing
import pickle

import numpy as np
import pytest

import minitensor as mt

CASES = {
    "float32": lambda: mt.from_numpy(np.array([1.0, -0.0, np.nan, np.inf], np.float32)),
    "float64": lambda: mt.from_numpy(np.array([[1e-300, 5e-324]], np.float64)),
    "int32": lambda: mt.from_numpy(np.array([1, -2, 2**31 - 1], np.int32)),
    "int64": lambda: mt.from_numpy(np.array([-(2**63), 2**63 - 1], np.int64)),
    "bool": lambda: mt.from_numpy(np.array([True, False, True])),
    "0-d": lambda: mt.from_numpy(np.array(3.5, np.float32)),
    "empty": lambda: mt.from_numpy(np.zeros((0, 3), np.float32)),
    "transposed": lambda: mt.from_numpy(
        np.arange(6, dtype=np.int64).reshape(2, 3)
    ).transpose(0, 1),
}

ROUND_TRIPS = {
    "pickle": lambda t: pickle.loads(pickle.dumps(t)),
    "pickle-protocol-5": lambda t: pickle.loads(pickle.dumps(t, protocol=5)),
    "deepcopy": copy.deepcopy,
    "copy": copy.copy,
}


def _same(a, b):
    a, b = a.numpy(), b.numpy()
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.mark.parametrize("how", list(ROUND_TRIPS))
@pytest.mark.parametrize("case", list(CASES))
def test_values_dtype_and_shape_survive_bit_for_bit(case, how):
    tensor = CASES[case]()
    copied = ROUND_TRIPS[how](tensor)
    assert _same(tensor, copied)
    assert copied.requires_grad == tensor.requires_grad


@pytest.mark.parametrize("how", list(ROUND_TRIPS))
def test_requires_grad_is_kept(how):
    tensor = mt.Tensor([1.0, 2.0], requires_grad=True)
    assert ROUND_TRIPS[how](tensor).requires_grad


@pytest.mark.parametrize("how", list(ROUND_TRIPS))
def test_a_copy_made_under_no_grad_keeps_requires_grad(how):
    """`no_grad` is where a snapshot is usually taken, and a tensor built there
    does not require a gradient unless told to after the fact."""
    tensor = mt.Tensor([1.0, 2.0], requires_grad=True)
    with mt.no_grad():
        copied = ROUND_TRIPS[how](tensor)
    assert copied.requires_grad


def test_unpickling_under_no_grad_keeps_requires_grad():
    saved = pickle.dumps(mt.Tensor([1.0], requires_grad=True))
    with mt.no_grad():
        assert pickle.loads(saved).requires_grad


def test_the_copy_is_an_independent_leaf():
    weight = mt.Tensor([1.0, 2.0], requires_grad=True)
    (weight * 3.0).sum().backward()

    copied = copy.deepcopy(weight)
    assert copied.grad is None
    (copied * 5.0).sum().backward()

    np.testing.assert_array_equal(weight.grad.numpy(), [3.0, 3.0])
    np.testing.assert_array_equal(copied.grad.numpy(), [5.0, 5.0])


def test_a_result_copies_as_a_leaf():
    weight = mt.Tensor([1.0, 2.0], requires_grad=True)
    result = weight * 2.0
    copied = copy.deepcopy(result)

    assert copied.requires_grad
    copied.sum().backward()
    assert weight.grad is None
    np.testing.assert_array_equal(copied.grad.numpy(), [1.0, 1.0])


def test_writing_to_the_copy_leaves_the_original_alone():
    original = mt.Tensor([1.0, 2.0])
    copied = copy.copy(original)
    copied[0] = 9.0
    np.testing.assert_array_equal(original.numpy(), [1.0, 2.0])


def test_a_structure_holding_tensors_deep_copies():
    state = {"step": 3, "moments": [mt.Tensor([1.0]), mt.Tensor([2.0])]}
    copied = copy.deepcopy(state)
    assert copied["step"] == 3
    assert [m.numpy().tolist() for m in copied["moments"]] == [[1.0], [2.0]]


def _double(tensor):
    return tensor * 2.0


def test_tensors_cross_a_spawned_process_both_ways():
    context = multiprocessing.get_context("spawn")
    tensors = [mt.Tensor([1.0, 2.0]), mt.from_numpy(np.array([3, 4], np.int64))]
    with context.Pool(1) as pool:
        doubled = pool.map(_double, tensors)
    np.testing.assert_array_equal(doubled[0].numpy(), [2.0, 4.0])
    np.testing.assert_array_equal(doubled[1].numpy(), [6, 8])
