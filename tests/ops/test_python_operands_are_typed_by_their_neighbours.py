# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A Python number or list handed to an operation is typed by what it stands beside.

A Python value carries no dtype. Beside a typed operand -- a tensor, an array
-- it takes that operand's width, by the rule `x + 0.1` already used; on its
own it is typed by what it holds, as `as_tensor` types it. Every operation
that was not an operator built such a value at the default float instead, so:

- `append(float64_x, 0.1)` joined the float32 `0.1`, `0.10000000149011612`,
  and so did `insert`, `cat`, `stack` and the rest of the joins;
- `insert(int32_x, 1, 7)` came back int64, where `int32_x + 7` stays int32;
- `searchsorted(float64_x, [0.1])` and `einsum` refused the float32 they made;
- `isclose([0.1], float64_x)` said a float64 holding `0.1` was not `0.1`;
- `cummax([1, 3, 2])` was a float, a float64 array came out of
  `logcumsumexp` as float32, and a list of labels was no use to `ctc_loss`.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

DIGITS = 0.1  # not a float32: read at float32 it is 0.10000000149011612
nc = mt.numpy_compat


def _f64(values):
    return mt.Tensor(values, dtype="float64")


def _i32(values):
    return mt.Tensor(values, dtype="int32")


JOINS = {
    "append": lambda x, v: mt.append(x, v),
    "insert": lambda x, v: mt.insert(x, 1, v),
    "cat": lambda x, v: mt.cat([x, v]),
    "concatenate-flat": lambda x, v: nc.concatenate([x, v], axis=None),
    "hstack": lambda x, v: mt.hstack([x, v]),
    "ediff1d": lambda x, v: mt.ediff1d(x, to_begin=v),
    "diff": lambda x, v: mt.diff(x, prepend=v),
    "union1d": lambda x, v: mt.union1d(x, v),
    "setxor1d": lambda x, v: mt.setxor1d(x, v),
}


@pytest.mark.parametrize("join", list(JOINS.values()), ids=list(JOINS))
def test_a_float_joined_onto_a_float64_keeps_its_digits(join):
    got = join(_f64([1.0, 2.0]), [DIGITS])
    assert str(got.dtype) == "float64"
    # Every value in these answers is either an input or built from one, and
    # with the float32 `0.1` none of them is what the float64 arithmetic gives.
    want = join(_f64([1.0, 2.0]), _f64([DIGITS]))
    assert got.tolist() == want.tolist()


@pytest.mark.parametrize(
    "join",
    [JOINS["append"], JOINS["insert"], JOINS["cat"], JOINS["hstack"]],
    ids=["append", "insert", "cat", "hstack"],
)
def test_an_int_joined_onto_an_int32_stays_int32(join):
    got = join(_i32([1, 2]), [7])
    assert str(got.dtype) == "int32"
    if join is not JOINS["cat"]:  # `cat` joins tensors of one rank
        assert str(join(_i32([1, 2]), 7).dtype) == "int32"


def test_a_float_beside_an_int64_widens_to_float64_as_in_arithmetic():
    big = mt.Tensor([2**53], dtype="int64")
    assert str((big + 0.5).dtype) == "float64"
    for got in (mt.append(big, 0.5), mt.insert(big, 0, [0.5]), mt.cat([big, [0.5]])):
        assert str(got.dtype) == "float64"


@pytest.mark.parametrize(
    "stack,want",
    [
        (mt.stack, [[1.0, 2.0], [DIGITS, DIGITS]]),
        (mt.vstack, [[1.0, 2.0], [DIGITS, DIGITS]]),
        (mt.column_stack, [[1.0, DIGITS], [2.0, DIGITS]]),
        (mt.dstack, [[[1.0, DIGITS], [2.0, DIGITS]]]),
    ],
    ids=["stack", "vstack", "column_stack", "dstack"],
)
def test_the_stacking_helpers_read_a_list_beside_the_tensors(stack, want):
    got = stack([_f64([1.0, 2.0]), [DIGITS, DIGITS]])
    assert str(got.dtype) == "float64"
    assert got.tolist() == want


def test_a_list_takes_the_width_the_typed_operands_promote_to():
    # Beside an int32 and a float64, a list of ints is a float64.
    got = mt.hstack([_i32([1]), _f64([0.5]), [3]])
    assert str(got.dtype) == "float64"
    assert got.tolist() == [1.0, 0.5, 3.0]
    # With nothing typed, each list is typed by what it holds.
    assert str(mt.cat([[1, 2], [3]]).dtype) == "int64"


def test_convolve_reads_the_kernel_at_the_signal_width():
    got = mt.convolve(_f64([1.0, 2.0]), [DIGITS])
    assert got.tolist() == [DIGITS, 2 * DIGITS]


@pytest.mark.parametrize("left", [True, False], ids=["literal-left", "literal-right"])
def test_membership_reads_the_list_beside_the_tensor(left):
    tensor = _f64([DIGITS, 0.5])
    pair = ([DIGITS], tensor) if left else (tensor, [DIGITS])
    assert mt.isin(*pair).tolist() == ([True] if left else [True, False])
    assert mt.intersect1d(*pair).tolist() == [DIGITS]
    assert mt.setdiff1d(*pair).tolist() == ([] if left else [0.5])


@pytest.mark.parametrize("left", [True, False], ids=["literal-left", "literal-right"])
def test_closeness_and_equality_read_the_list_beside_the_tensor(left):
    tensor = _f64([DIGITS])
    pair = ([DIGITS], tensor) if left else (tensor, [DIGITS])
    assert mt.isclose(*pair, rtol=0, atol=0).tolist() == [True]
    assert mt.allclose(*pair, rtol=0, atol=0)
    assert mt.array_equal(*pair)
    assert nc.allclose(*pair, rtol=0, atol=0)
    assert nc.array_equal(*pair)


def test_a_comparison_list_is_still_read_exactly():
    # The comparison rule, not the arithmetic one: an int beside a float32 is
    # the integer it is, so 16777217 is not the float32 16777216.
    single = mt.Tensor([16_777_216.0], dtype="float32")
    assert not mt.array_equal([16_777_217], single)
    assert mt.isin([16_777_217], single).tolist() == [False]
    # A list of them is read as exactly as one is.
    assert (single == 16_777_217).tolist() == [False]
    assert (single == [16_777_217]).tolist() == [False]
    assert (single < [16_777_217]).tolist() == [True]
    assert ([16_777_217] > single).tolist() == [True]
    assert (single == [16_777_216]).tolist() == [True]
    # Past int64 there is no exact reading, and the float one stands.
    assert (single == [2**70]).tolist() == [False]


@pytest.mark.parametrize("search", [mt.searchsorted, mt.bucketize])
def test_a_search_reads_a_list_on_either_side_at_the_other_width(search):
    tensor = _f64([DIGITS, 0.5])
    if search is mt.searchsorted:
        assert search(tensor, [DIGITS]).tolist() == [0]
        assert search([DIGITS, 0.5], _f64([DIGITS])).tolist() == [0]
    else:
        assert search([DIGITS], tensor).tolist() == [0]
        assert search(tensor, [DIGITS, 0.5]).tolist() == [0, 1]


def test_products_and_selection_read_a_list_beside_the_tensor():
    tensor = _f64([DIGITS, 0.2, 0.3])
    want = float(np.dot([DIGITS, 0.2, 0.3], [DIGITS, 0.2, 0.3]))
    assert mt.einsum("i,i->", tensor, [DIGITS, 0.2, 0.3]).item() == want
    assert nc.dot([DIGITS, 0.2, 0.3], tensor).item() == want
    assert nc.matmul(tensor, [DIGITS, 0.2, 0.3]).item() == want
    assert str(nc.cross([1.0, 0.0, DIGITS], tensor).dtype) == "float64"
    mask = mt.as_tensor([True, False, True])
    for where in (mt.where, nc.where):
        got = where(mask, [1, 2, 3], _i32([4, 5, 6]))
        assert str(got.dtype) == "int32" and got.tolist() == [1, 5, 3]


def test_a_value_written_into_a_tensor_takes_its_width():
    target = mt.zeros(1, dtype="float64")
    assert target.copy_([DIGITS]).tolist() == [DIGITS]
    weight = _f64([1.0, 2.0]).requires_grad_(True)
    weight.grad = [DIGITS, 0.2]
    assert str(weight.grad.dtype) == "float64"
    assert weight.grad.tolist() == [DIGITS, 0.2]
    scattered = mt.functional.scatter_reduce(
        mt.zeros(3, dtype="float64"), 0, [0, 1], [DIGITS, 0.2], "sum"
    )
    assert scattered.tolist() == [DIGITS, 0.2, 0.0]


def test_an_operand_on_its_own_is_typed_by_what_it_holds():
    assert str(mt.cummax([1, 3, 2])[0].dtype) == "int64"
    assert str(mt.mode([1, 1, 2])[0].dtype) == "int64"
    wide = np.array([DIGITS, 0.2])
    assert str(mt.logcumsumexp(wide).dtype) == "float64"
    assert mt.logcumsumexp(wide).tolist()[0] == DIGITS
    assert str(mt.zeros_like(wide).dtype) == "float64"
    assert str(nc.asarray(wide).dtype) == "float64"
    assert nc.asarray(wide).tolist() == wide.tolist()


def test_ctc_loss_takes_its_labels_and_lengths_as_lists():
    log_probs = mt.Tensor(np.log(np.full((4, 1, 3), 1 / 3)), dtype="float32")
    from_lists = mt.functional.ctc_loss(log_probs, [[1, 2]], [4], [2])
    from_tensors = mt.functional.ctc_loss(
        log_probs, mt.as_tensor([[1, 2]]), mt.as_tensor([4]), mt.as_tensor([2])
    )
    assert from_lists.item() == from_tensors.item()


def test_a_list_the_width_cannot_hold_is_refused_as_a_number_is():
    # The result's dtype is fixed by the operands' dtypes, never by the values
    # in them: `int32 + [2**40]` came back int64 where `int32 + 2**40` refused.
    tensor = _i32([1, 2])
    with pytest.raises(OverflowError):
        tensor + 2**40
    with pytest.raises(OverflowError):
        tensor + [2**40, 1]
    assert str((tensor + [7, 8]).dtype) == "int32"


BINARY = [
    "add", "sub", "mul", "div", "pow", "atan2", "hypot", "fmod", "remainder",
    "maximum", "minimum", "fmax", "fmin", "copysign", "logaddexp", "logaddexp2",
    "xlogy", "nextafter", "float_power", "floor_divide", "heaviside", "dist",
    "dot", "inner", "outer", "kron", "matmul", "vdot", "eq", "ne", "lt", "le",
    "gt", "ge",
]  # fmt: skip


@pytest.mark.parametrize("name", BINARY)
@pytest.mark.parametrize("side", ["left", "right"])
def test_a_binary_function_reads_a_literal_on_either_side_as_its_typed_twin(name, side):
    # Built on its own, a literal first operand was the float32 `0.1` -- or,
    # for the functions written in the bindings, refused outright.
    fn = getattr(mt, name)
    tensor, typed = _f64([0.5]), _f64([DIGITS])
    pair = ([DIGITS], tensor) if side == "left" else (tensor, [DIGITS])
    twin = (typed, tensor) if side == "left" else (tensor, typed)
    got, want = fn(*pair), fn(*twin)
    assert str(got.dtype) == str(want.dtype)
    assert got.tolist() == want.tolist()


@pytest.mark.parametrize("name", ["add", "sub", "mul", "maximum", "minimum", "fmod"])
def test_an_int_literal_first_stays_at_the_int32_beside_it(name):
    fn = getattr(mt, name)
    got = fn([7], _i32([3]))
    assert str(got.dtype) == "int32"
    assert got.tolist() == fn(_i32([7]), _i32([3])).tolist()


def test_three_operand_forms_read_every_literal_beside_the_typed_ones():
    start, end = _f64([0.0]), _f64([1.0])
    assert mt.lerp(start, end, DIGITS).tolist() == [DIGITS]
    assert mt.lerp([0.0], end, DIGITS).tolist() == [DIGITS]
    assert mt.addcmul(_f64([0.0]), [DIGITS], [1.0]).tolist() == [DIGITS]
    assert mt.addcdiv([0.0], _f64([DIGITS]), [1.0]).tolist() == [DIGITS]
    assert mt.float_power([DIGITS], 1).tolist() == [DIGITS]


def test_a_scalar_base_and_a_plain_number_are_accepted():
    assert mt.pow(2, mt.Tensor([3.0])).tolist() == [8.0]
    assert mt.atan2(0.0, mt.Tensor([1.0])).tolist() == [0.0]
    assert mt.exp(0.0).item() == 1.0
    assert str(mt.sqrt([4, 9]).dtype) == "float64"  # int64 widens to float64
    assert mt.sqrt([4, 9]).tolist() == [2.0, 3.0]
    # A comparison reads an int literal exactly, on the left as on the right.
    single = mt.Tensor([16_777_216.0], dtype="float32")
    assert mt.lt([16_777_217], single).tolist() == [False]
    assert mt.gt([16_777_217], single).tolist() == [True]


@pytest.mark.parametrize(
    "name",
    ["mm", "mv", "tensordot", "vecdot", "matvec", "vecmat", "cdist"],
)
def test_matrix_forms_read_a_literal_beside_the_tensor(name):
    fn = getattr(mt, name)
    matrix = _f64([[DIGITS, 0.2], [0.3, 0.4]])
    vector = [DIGITS, 0.2]
    if name in ("mm", "cdist"):
        literal, typed = [[DIGITS, 0.2], [0.3, 0.4]], matrix
    else:
        literal, typed = vector, _f64(vector)
    if name in ("mv", "matvec"):
        got, want = fn(matrix, literal), fn(matrix, typed)
    elif name == "vecmat":
        got, want = fn(literal, matrix), fn(typed, matrix)
    elif name == "tensordot":
        got, want = fn(literal, matrix, 1), fn(typed, matrix, 1)
    elif name == "vecdot":
        got, want = fn(literal, _f64(vector)), fn(typed, _f64(vector))
    else:
        got, want = fn(literal, matrix), fn(typed, matrix)
    assert str(got.dtype) == "float64"
    assert got.tolist() == want.tolist()
