# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every tensor an op produces has to be contiguous.

`engine::Tensor` states this as a layout invariant, and the whole engine is
built on it: kernels fetch `data().as_*_slice()` and index it directly, and
several size their work from the buffer length rather than from the shape. An
op that returned a strided view instead of materialising it would not merely be
slow -- every kernel downstream of it would read the wrong elements, quietly
and with no error anywhere.

Transposes, slices, expansions and permutations are exactly the operations a
strided library would hand back as views, so they are checked by name on top of
the sweep. Nothing failed when this was written; it is here because the
invariant is load-bearing for code that never checks it.
"""

import inspect
import re

import numpy as np
import pytest

import minitensor as mt

_BASES = {
    "1d": np.arange(1.0, 9.0) * 0.4 + 0.2,
    "2x3": np.arange(1.0, 7.0).reshape(2, 3) * 0.7 + 0.3,
    "4x6": np.arange(1.0, 25.0).reshape(4, 6) * 0.37 + 0.5,
    "3d": np.arange(1.0, 25.0).reshape(2, 3, 4) * 0.31 + 0.6,
}

# Not ops on a tensor: these take a seed or a path.
_EXEMPT = re.compile(r"^(manual_seed|save|load|get_gradient)$")


def _tensors(result):
    """Every tensor in an op's answer, however it is packaged."""

    if isinstance(result, tuple):
        for item in result:
            yield from _tensors(item)
    elif hasattr(result, "is_contiguous"):
        yield result


def _check(label, produced, offenders, counter):
    for tensor in _tensors(produced):
        counter[0] += 1
        if not tensor.is_contiguous():
            offenders.append(f"{label} -> {tuple(tensor.shape)}")


@pytest.mark.parametrize("base", sorted(_BASES))
def test_every_free_function_returns_a_contiguous_tensor(base):
    values = _BASES[base]
    offenders, counter = [], [0]
    for name in sorted(mt.__all__):
        op = getattr(mt, name, None)
        if (
            not callable(op)
            or inspect.isclass(op)
            or inspect.ismodule(op)
            or _EXEMPT.match(name)
        ):
            continue
        try:
            params = list(inspect.signature(op).parameters)
        except (ValueError, TypeError):
            continue
        if not params or params[0] not in ("input", "a", "x", "tensor"):
            continue
        try:
            produced = op(mt.Tensor(values, dtype="float64"))
        except Exception:
            continue  # needs more arguments, or does not apply to this shape
        _check(name, produced, offenders, counter)

    assert not offenders, "non-contiguous results: " + ", ".join(offenders)
    assert counter[0] > 50, f"only {counter[0]} results reached on {base}"


@pytest.mark.parametrize("base", sorted(_BASES))
def test_every_no_arg_method_returns_a_contiguous_tensor(base):
    values = _BASES[base]
    offenders, counter = [], [0]
    for name in sorted(n for n in dir(mt.Tensor) if not n.startswith("_")):
        try:
            attr = getattr(mt.Tensor(values, dtype="float64"), name)
            if not callable(attr):
                continue
            produced = attr()
        except Exception:
            continue
        _check(f"Tensor.{name}", produced, offenders, counter)

    assert not offenders, "non-contiguous results: " + ", ".join(offenders)
    assert counter[0] > 50, f"only {counter[0]} results reached on {base}"


def test_the_operations_a_strided_library_would_call_views_are_materialised():
    grid = mt.Tensor(_BASES["4x6"], dtype="float64")
    cube = mt.Tensor(_BASES["3d"], dtype="float64")
    row = mt.Tensor(np.ones((1, 6)), dtype="float64")

    cases = {
        "transpose": grid.transpose(0, 1),
        "t": grid.t(),
        "matrix_transpose": mt.matrix_transpose(grid),
        "permute": cube.permute([2, 0, 1]),
        "movedim": mt.movedim(cube, 0, 2),
        "swapaxes": mt.swapaxes(cube, 0, 2),
        "step-slice": grid[:, ::2],
        "row-slice": grid[::2, :],
        "offset-slice": grid[1:3, 2:5],
        "narrow": mt.narrow(grid, 1, 1, 3),
        "select": mt.select(grid, 0, 2),
        "flip": grid.flip([0, 1]),
        "diagonal": grid.diagonal(),
        "reshape": grid.reshape([6, 4]),
        "expand": row.expand([4, 6]),
        "broadcast_to": mt.broadcast_to(row, (4, 6)),
        "fancy-index": grid[mt.Tensor(np.array([2, 0, 3]), dtype="int64")],
        "bool-mask": grid[grid > 3.0],
    }

    strided = [name for name, made in cases.items() if not made.is_contiguous()]
    assert not strided, "handed back as a strided view: " + ", ".join(strided)

    # And the values are the ones the layout claims, which is what a
    # materialisation gets wrong when it copies from the wrong offsets.
    np.testing.assert_allclose(cases["transpose"].numpy(), _BASES["4x6"].T)
    np.testing.assert_allclose(cases["step-slice"].numpy(), _BASES["4x6"][:, ::2])
    np.testing.assert_allclose(cases["offset-slice"].numpy(), _BASES["4x6"][1:3, 2:5])
    np.testing.assert_allclose(cases["flip"].numpy(), _BASES["4x6"][::-1, ::-1])
    np.testing.assert_allclose(
        cases["permute"].numpy(), _BASES["3d"].transpose(2, 0, 1)
    )


@pytest.mark.parametrize("name", ["floor", "ceil", "trunc", "round"])
@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_an_identity_result_does_not_alias_its_input(name, dtype):
    """Rounding an integer hands back the input, storage and all.

    That is the cheapest correct answer -- the values are already what the op
    would compute -- and it is safe only because a write to either tensor goes
    through the engine's copy-on-write: `data_mut` clones the buffer when the
    `Arc` is shared. This pins that, because the identity returns rely on it
    and nothing else in the suite does.
    """

    values = np.array([-3, -1, 0, 2, 5], dtype=dtype)
    original = mt.Tensor(values, dtype=dtype)
    rounded = getattr(original, name)()
    np.testing.assert_array_equal(rounded.numpy(), values)

    rounded.fill_(99)
    np.testing.assert_array_equal(
        original.numpy(), values
    ), f"{name} let a write to its result reach the input"
    np.testing.assert_array_equal(rounded.numpy(), np.full_like(values, 99))
