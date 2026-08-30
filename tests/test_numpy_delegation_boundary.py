# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""NumPy computes indices and shapes; the engine computes values.

MiniTensor depends on NumPy at runtime, so anything NumPy already does well is
free to use and reimplementing it costs the library twice: once in code, and
again every time the reimplementation disagrees with the array library everyone
compares against. `tril_indices` was six tensor allocations -- two ranges, a
broadcast subtraction, a comparison, a `nonzero` and a transpose -- computing
from three Python integers what `np.tril_indices` returns in one call.

The line is not "use NumPy where it is faster", because that reading loses real
behaviour. A tensor argument carries three things a NumPy array does not: a
device, a place in the autograd graph, and a share of the seeded random stream.
Handing one to NumPy copies it to the host, drops its gradient *silently* -- the
values still come out right, so nothing that checks values would notice -- and,
for the samplers, draws from NumPy's generator rather than the one
`manual_seed` controls. So `isin`, `multinomial` and `bernoulli` stay written in
terms of kernels even though NumPy has all three: their arguments are tensors.

Which leaves a rule that can be checked rather than remembered. A function whose
arguments are Python integers has no device, no gradient and no seed to protect,
and there is nothing left to weigh: NumPy does it. The tests below hold both
halves of that -- the integers-only helpers delegate, and the tensor-taking
samplers still answer to `manual_seed`.

There is a third case, and it is the one that goes wrong quietly: a function
that computes *part* of its answer with NumPy -- an index map, a coordinate
grid, a mask -- and then combines it with tensor data. What NumPy builds arrives
in float64 on the host, so the crossing back has to carry the input's dtype and
device or the result silently changes precision. `_index_tensor` and
`_constant_like` are that crossing, and the sweep at the end of this file holds
every function that uses one to the dtype it was given.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

import minitensor as mt

# The modules that hold operations. `_api` is introspection and `__init__` is
# re-exports; neither computes anything, so neither is subject to the rule.
_OPERATION_MODULES = (
    "_calculus",
    "_derived",
    "_elementwise",
    "_indexing",
    "_matrix",
    "_nn_extras",
    "_sampling",
    "_shape",
)


def _integers_only_functions() -> dict[str, str]:
    """Every public operation whose parameters are all annotated `int`.

    The value is the function's source together with the source of any helper
    it calls in the same module, because a two-line public function delegating
    to a private one is the shape these take.
    """

    package = pathlib.Path(mt.__file__).parent
    found: dict[str, str] = {}
    for name in _OPERATION_MODULES:
        tree = ast.parse((package / f"{name}.py").read_text())
        bodies = {
            node.name: ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                continue
            arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            if not arguments or not all(
                isinstance(argument.annotation, ast.Name)
                and argument.annotation.id == "int"
                for argument in arguments
            ):
                continue
            source = [bodies[node.name]]
            source += [
                bodies[called.func.id]
                for called in ast.walk(node)
                if isinstance(called, ast.Call)
                and isinstance(called.func, ast.Name)
                and called.func.id in bodies
            ]
            found[node.name] = "\n".join(source)
    return found


def test_an_operation_over_plain_integers_is_left_to_numpy():
    integers_only = _integers_only_functions()

    # A guard that found nothing would pass without checking anything.
    assert "tril_indices" in integers_only
    assert "triu_indices" in integers_only

    for name, source in sorted(integers_only.items()):
        assert "_np." in source, (
            f"{name} takes only Python integers, so it has no device, no "
            "gradient and no seed to protect -- see 'Where an operation "
            "belongs' in docs/development.md. Use NumPy rather than kernels."
        )


def test_the_triangle_builders_agree_with_the_library_they_delegate_to():
    for row, col, offset in [(4, 2, -1), (2, 5, 1), (6, 6, 0), (1, 1, 3)]:
        np.testing.assert_array_equal(
            mt.tril_indices(row, col, offset).numpy(),
            np.array(np.tril_indices(row, offset, col)),
        )
        np.testing.assert_array_equal(
            mt.triu_indices(row, col, offset).numpy(),
            np.array(np.triu_indices(row, offset, col)),
        )


def test_broadcast_shapes_agrees_with_the_library_it_delegates_to():
    assert mt.broadcast_shapes((3, 1), (1, 4)) == np.broadcast_shapes((3, 1), (1, 4))
    with pytest.raises(ValueError, match="cannot be broadcast"):
        mt.broadcast_shapes((3,), (4,))


@pytest.mark.parametrize("draw", ["bernoulli", "multinomial"])
def test_a_sampler_answers_to_the_librarys_own_seed(draw):
    """The reason the samplers are not delegated, stated as a test.

    NumPy has `binomial` and `multinomial` and they would be shorter to call,
    but they draw from NumPy's generator. `manual_seed` would then set the seed
    for `rand` and `randn` and quietly not for these, which is worse than not
    having the function.
    """

    def sample():
        if draw == "bernoulli":
            return mt.bernoulli(mt.Tensor([0.5] * 16)).numpy()
        return mt.multinomial(mt.Tensor([1.0, 2.0, 3.0, 4.0]), 16, True).numpy()

    mt.manual_seed(101)
    first = sample()
    mt.manual_seed(101)
    np.testing.assert_array_equal(sample(), first)

    mt.manual_seed(202)
    assert not np.array_equal(sample(), first), "a different seed, the same draw"


# --- the crossing back ------------------------------------------------------


def _f32(shape, fill=None):
    values = (
        np.full(shape, fill, dtype=np.float32)
        if fill is not None
        else np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) / 7.0
    )
    return mt.Tensor(values, dtype="float32")


#: Every operation that builds part of its answer with NumPy and then combines
#: it with tensor data. Named here so a new one is added by being written.
_MIXES_NUMPY_WITH_TENSORS = {
    "unfold": lambda: mt.functional.unfold(_f32((1, 2, 5, 5)), 2, padding=1),
    "fold": lambda: mt.functional.fold(_f32((1, 8, 36)), (5, 5), 2, padding=1),
    "conv3d": lambda: mt.functional.conv3d(
        _f32((1, 2, 4, 4, 4)), _f32((3, 2, 2, 2, 2))
    ),
    "max_pool3d": lambda: mt.functional.max_pool3d(_f32((1, 2, 4, 4, 4)), 2),
    "avg_pool3d": lambda: mt.functional.avg_pool3d(_f32((1, 2, 4, 4, 4)), 2),
    "local_response_norm": lambda: mt.functional.local_response_norm(
        _f32((1, 4, 3, 3)), 3
    ),
    "affine_grid": lambda: mt.functional.affine_grid(
        _f32((1, 2, 3), fill=1.0), (1, 1, 3, 3)
    ),
    "embedding": lambda: mt.functional.embedding(
        mt.Tensor.from_numpy(np.array([0, 1], dtype=np.int64)), _f32((4, 3)), 0
    ),
    "slice_scatter": lambda: mt.slice_scatter(_f32((3, 4)), _f32((1, 4)), 0, 1, 2),
    "select_scatter": lambda: mt.select_scatter(_f32((3, 4)), _f32((4,)), 0, 1),
    "diagonal_scatter": lambda: mt.diagonal_scatter(_f32((3, 4)), _f32((3,))),
    "put": lambda: mt.put(
        _f32((3, 4)), mt.Tensor.from_numpy(np.array([0], dtype=np.int64)), _f32((1,))
    ),
    "pdist": lambda: mt.pdist(_f32((4, 3))),
}


@pytest.mark.parametrize("name", sorted(_MIXES_NUMPY_WITH_TENSORS))
def test_a_numpy_built_part_does_not_change_the_dtype(name):
    """What NumPy builds is float64 on the host; the input's dtype has to win.

    Promoting to float64 here would double the memory of a half-precision model
    and still produce right-looking numbers, which is why it needs a test and
    not a convention.
    """

    result = _MIXES_NUMPY_WITH_TENSORS[name]()
    assert "float32" in str(result.dtype), f"{name} returned {result.dtype}"


@pytest.mark.parametrize("name", sorted(_MIXES_NUMPY_WITH_TENSORS))
def test_a_numpy_built_part_stays_on_the_input_device(name):
    """Vacuous on a CPU-only build, and the assertion that would catch it."""

    result = _MIXES_NUMPY_WITH_TENSORS[name]()
    assert str(result.device) == str(_f32((1,)).device)
