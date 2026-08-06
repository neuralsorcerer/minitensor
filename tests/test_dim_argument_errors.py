# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A bad `dim` argument should say so, and say the right thing about it.

Every one of these operations used to report a `dim` mistake as if it were an
element index: `x.sum(5)` on a 2-D tensor raised

    Index out of bounds: index 5 is out of bounds for dimension 0 with size 2
    💡 Suggestion: Index must be in range [0, 2)

where no dimension 0 is involved, `2` is the rank rather than anybody's size,
and the suggestion denies the negative dims that in fact work. Several ops also
reported the dim *after* resolving it, so `x.sum(-5)` complained about `-3` --
a number the caller never wrote.

The important test here is `test_stated_range_is_the_range_that_works`: it
pins the range in the message to the range the op actually accepts, so the two
cannot drift apart later.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import minitensor as mt

# (name, callable taking (tensor, dim)). Every entry takes a dim naming an
# existing axis, so all of them accept exactly [-ndim, ndim).
AXIS_OPS = {
    "sum": lambda t, d: t.sum(d),
    "mean": lambda t, d: t.mean(d),
    "prod": lambda t, d: t.prod(d),
    "max": lambda t, d: t.max(d),
    "min": lambda t, d: t.min(d),
    "argmax": lambda t, d: t.argmax(d),
    "argmin": lambda t, d: t.argmin(d),
    "cumsum": lambda t, d: t.cumsum(d),
    "cumprod": lambda t, d: t.cumprod(d),
    "softmax": lambda t, d: t.softmax(d),
    "log_softmax": lambda t, d: t.log_softmax(d),
    "logsumexp": lambda t, d: t.logsumexp(d),
    "var": lambda t, d: t.var(d),
    "std": lambda t, d: t.std(d),
    "median": lambda t, d: t.median(d),
    "sort": lambda t, d: t.sort(d),
    "squeeze": lambda t, d: t.squeeze(d),
    "norm": lambda t, d: t.norm(2, d),
    "all": lambda t, d: t.all(d),
    "any": lambda t, d: t.any(d),
    "flip": lambda t, d: t.flip(d),
    "chunk": lambda t, d: t.chunk(1, d),
    "split": lambda t, d: t.split(3, d),
    "cat": lambda t, d: mt.cat([t, t], d),
    "transpose": lambda t, d: t.transpose(d, 0),
    "flatten": lambda t, d: t.flatten(d, 2),
}

# These insert an axis rather than naming one, so a dim one past the last is
# legal and the accepted range is one wider: [-(ndim + 1), ndim].
INSERTING_OPS = {
    "unsqueeze": lambda t, d: t.unsqueeze(d),
    "stack": lambda t, d: mt.stack([t, t], d),
}

OUT_OF_RANGE = (-9, -5, -4, 3, 7, 100)

RANGE_PATTERN = re.compile(
    r"expected to be in range of \[(-?\d+), (-?\d+)\], but got (-?\d+)"
)


def _tensor():
    return mt.randn(3, 4, 5)


@pytest.mark.parametrize("name", sorted(AXIS_OPS))
@pytest.mark.parametrize("dim", OUT_OF_RANGE)
def test_axis_ops_reject_out_of_range_dims(name, dim):
    with pytest.raises(IndexError) as excinfo:
        AXIS_OPS[name](_tensor(), dim)
    assert "Dimension out of range" in str(excinfo.value)


@pytest.mark.parametrize("name", sorted(AXIS_OPS))
@pytest.mark.parametrize("dim", OUT_OF_RANGE)
def test_message_names_the_dim_the_caller_wrote(name, dim):
    """Not the value after `dim + ndim`.

    `sum(-5)` on a 3-D tensor used to report `-2`, sending the reader looking
    for a `-2` that appears nowhere in their code.
    """
    with pytest.raises(IndexError) as excinfo:
        AXIS_OPS[name](_tensor(), dim)
    match = RANGE_PATTERN.search(str(excinfo.value))
    assert match is not None, str(excinfo.value)
    assert int(match.group(3)) == dim


@pytest.mark.parametrize("name", sorted(AXIS_OPS) + sorted(INSERTING_OPS))
def test_stated_range_is_the_range_that_works(name):
    """The bracket in the message must be exactly the accepted set.

    This is what keeps the message honest: the old one advertised `[0, 2)`
    while `-1` and `-2` both worked, and nothing caught it.
    """
    op = AXIS_OPS.get(name) or INSERTING_OPS[name]
    tensor = _tensor()

    accepted = []
    for dim in range(-9, 10):
        try:
            op(tensor, dim)
        except IndexError:
            continue
        accepted.append(dim)

    with pytest.raises(IndexError) as excinfo:
        op(tensor, 99)
    match = RANGE_PATTERN.search(str(excinfo.value))
    assert match is not None, str(excinfo.value)
    low, high = int(match.group(1)), int(match.group(2))

    assert accepted == list(
        range(low, high + 1)
    ), f"{name} advertises [{low}, {high}] but accepts {accepted}"


@pytest.mark.parametrize("name", sorted(INSERTING_OPS))
@pytest.mark.parametrize("dim", (-5, 4, 9))
def test_inserting_ops_reject_beyond_their_wider_range(name, dim):
    with pytest.raises(IndexError) as excinfo:
        INSERTING_OPS[name](_tensor(), dim)
    assert "Dimension out of range" in str(excinfo.value)


@pytest.mark.parametrize("name", sorted(INSERTING_OPS))
def test_inserting_ops_accept_one_past_the_last_axis(name):
    result = INSERTING_OPS[name](_tensor(), 3)
    assert result.shape[-1] in (1, 2, 5)


def test_suggestion_does_not_deny_negative_dims():
    """The old suggestion read "Index must be in range [0, 2)"."""
    with pytest.raises(IndexError) as excinfo:
        mt.randn(3, 4).sum(7)
    message = str(excinfo.value)
    assert "-1" in message
    assert "Index must be in range" not in message


def test_transpose_names_the_argument_at_fault():
    """`transpose(-5, 1)` used to complain about `1`.

    It reported `max(dim0, dim1)` after resolving both, and -5 resolved to -3,
    so the maximum was the argument the caller had got right.
    """
    x = mt.randn(3, 4)

    with pytest.raises(IndexError) as excinfo:
        x.transpose(-5, 1)
    assert "dim0" in str(excinfo.value)
    assert "but got -5" in str(excinfo.value)

    with pytest.raises(IndexError) as excinfo:
        x.transpose(0, 9)
    assert "dim1" in str(excinfo.value)
    assert "but got 9" in str(excinfo.value)


@pytest.mark.parametrize("dim", OUT_OF_RANGE)
def test_permute_rejects_out_of_range_dims(dim):
    """Kept apart from AXIS_OPS: permute's other arguments have to be the
    remaining axes, so there is no fixed tail that stays a permutation as the
    first one varies."""
    with pytest.raises(IndexError) as excinfo:
        _tensor().permute(dim, 1, 2)
    assert "Dimension out of range" in str(excinfo.value)
    assert f"but got {dim}" in str(excinfo.value)


def test_permute_accepts_the_negative_dims_it_advertises():
    x = mt.randn(3, 4, 5)
    np.testing.assert_array_equal(
        x.permute(-1, -3, -2).numpy(), x.permute(2, 0, 1).numpy()
    )


def test_flatten_names_which_end_is_wrong():
    x = mt.randn(3, 4, 5)

    with pytest.raises(IndexError) as excinfo:
        x.flatten(7, 2)
    assert "start_dim" in str(excinfo.value)

    with pytest.raises(IndexError) as excinfo:
        x.flatten(0, 7)
    assert "end_dim" in str(excinfo.value)


def test_movedim_names_which_side_is_wrong():
    x = mt.randn(3, 4, 5)

    with pytest.raises(IndexError) as excinfo:
        x.movedim(7, 0)
    assert "source" in str(excinfo.value)

    with pytest.raises(IndexError) as excinfo:
        x.movedim(0, 7)
    assert "destination" in str(excinfo.value)


def test_zero_dimensional_tensor_says_it_has_no_dimensions():
    """`[-0, 0)` would be an empty and rather unhelpful range to print."""
    scalar = mt.Tensor(np.float32(3.0))
    with pytest.raises(IndexError) as excinfo:
        scalar.sum(0)
    message = str(excinfo.value)
    assert "0-dimensional" in message
    assert "without a dim" in message


def test_a_dim_mistake_is_not_reported_as_an_element_index():
    """The two are different mistakes and used to share a message."""
    x = mt.randn(3, 4)

    with pytest.raises(IndexError) as dim_error:
        x.sum(5)
    with pytest.raises(IndexError) as element_error:
        x[5]

    assert "Dimension out of range" in str(dim_error.value)
    assert "Dimension out of range" not in str(element_error.value)
    assert "out of bounds" in str(element_error.value)


@pytest.mark.parametrize("name", sorted(AXIS_OPS))
def test_negative_and_positive_dims_agree(name):
    """The negative dims the message advertises must do what it says.

    A message can be well-formed and still wrong about which axis `-1` names.
    """
    op = AXIS_OPS[name]
    tensor = mt.randn(3, 4, 5)
    for negative, positive in ((-1, 2), (-2, 1), (-3, 0)):
        left, right = op(tensor, negative), op(tensor, positive)
        left = left[0] if isinstance(left, tuple) else left
        right = right[0] if isinstance(right, tuple) else right
        if isinstance(left, list):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                np.testing.assert_array_equal(a.numpy(), b.numpy())
        else:
            np.testing.assert_array_equal(left.numpy(), right.numpy())
