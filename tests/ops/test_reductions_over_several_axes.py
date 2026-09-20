# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which reductions take a list of axes, and what they answer when they do.

The reference said reductions take a list of dims, and ten of them did.
`amax`, `amin`, `all`, `any`, `count_nonzero`, `ptp`, `average`, `quantile`,
`nanmedian`, `nanvar` -- twenty-two in total -- took an integer and answered a
list with "'list' object cannot be interpreted as an integer", including
`amax`, whose entire reason for existing next to `max` is that it has no index
to tie it to one axis. NumPy takes a tuple for every one of them and PyTorch
takes one for `amax`, `amin` and `count_nonzero`.

The line that remains is not about which op it is, it is about whether the
forward reports an index: an index names a position along one axis, so `max`,
`min`, `median`, `mode`, `argmax` and their NaN-aware forms still take exactly
one, and now say so instead of failing an integer conversion.

Two things are worth stating about the answers themselves:

  * Several axes are gathered into one and reduced together, not reduced one
    after another. For a value the two agree; for a *gradient* they do not.
    `amax` over both axes of `[[3, 3], [3, 1]]` has three tied maxima, so each
    gets a third -- reducing axis by axis would split within each axis first
    and hand out a quarter, a quarter and a half.
  * `quantile` with several probabilities reported no gradient at all before
    this: the batched kernel sorts once for every probability and attached no
    backward, so the result carried `requires_grad` with no `grad_fn`,
    `backward()` walked past it as a leaf, and a multi-quantile loss trained
    nothing without ever raising.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

RNG = np.random.default_rng(20260920)
VALUES = RNG.standard_normal((3, 4, 5))
# A NaN in one slice of every axis, and a row that is all NaN, since the
# NaN-aware forms differ from the plain ones exactly there.
NANS = VALUES.copy()
NANS[0, 1, 2] = np.nan
NANS[2, 0, :] = np.nan
FLAGS = RNG.random((3, 4, 5)) > 0.4

AXIS_SETS = [[0, 1], [1, 2], [0, 2], [0, 1, 2], [-3, -2], [-1, -3]]


def _t(values, requires_grad=False):
    return mt.Tensor(
        np.asarray(values, dtype=np.float64),
        dtype="float64",
        requires_grad=requires_grad,
    )


def _flags():
    return mt.Tensor(FLAGS.tolist(), dtype="bool")


# (name, minitensor call, numpy call). Each is checked at every axis set, both
# ways round on keepdim.
CASES = [
    (
        "amax",
        lambda d, k: mt.amax(_t(VALUES), d, k),
        lambda d, k: np.max(VALUES, d, keepdims=k),
    ),
    (
        "amin",
        lambda d, k: mt.amin(_t(VALUES), d, k),
        lambda d, k: np.min(VALUES, d, keepdims=k),
    ),
    (
        "nanamax",
        lambda d, k: mt.nanamax(_t(NANS), d, k),
        lambda d, k: np.nanmax(NANS, d, keepdims=k),
    ),
    (
        "nanamin",
        lambda d, k: mt.nanamin(_t(NANS), d, k),
        lambda d, k: np.nanmin(NANS, d, keepdims=k),
    ),
    (
        "ptp",
        lambda d, k: mt.ptp(_t(VALUES), d, k),
        lambda d, k: np.ptp(VALUES, d, keepdims=k),
    ),
    (
        "all",
        lambda d, k: mt.all(_flags(), d, k),
        lambda d, k: np.all(FLAGS, d, keepdims=k),
    ),
    (
        "any",
        lambda d, k: mt.any(_flags(), d, k),
        lambda d, k: np.any(FLAGS, d, keepdims=k),
    ),
    (
        "count_nonzero",
        lambda d, k: mt.count_nonzero(_flags(), d, k),
        lambda d, k: np.count_nonzero(FLAGS, d, keepdims=k),
    ),
    (
        "average",
        lambda d, k: mt.average(_t(VALUES), d, None, k),
        lambda d, k: np.mean(VALUES, d, keepdims=k),
    ),
    (
        "quantile",
        lambda d, k: mt.quantile(_t(VALUES), 0.4, d, k),
        lambda d, k: np.quantile(VALUES, 0.4, axis=d, keepdims=k),
    ),
    (
        "quantile[3]",
        lambda d, k: mt.quantile(_t(VALUES), [0.1, 0.5, 0.9], d, k),
        lambda d, k: np.quantile(VALUES, [0.1, 0.5, 0.9], axis=d, keepdims=k),
    ),
    (
        "nanquantile",
        lambda d, k: mt.nanquantile(_t(NANS), 0.4, d, k),
        lambda d, k: np.nanquantile(NANS, 0.4, axis=d, keepdims=k),
    ),
    (
        "percentile",
        lambda d, k: mt.percentile(_t(VALUES), 40.0, d, k),
        lambda d, k: np.percentile(VALUES, 40.0, axis=d, keepdims=k),
    ),
    (
        "nanvar",
        lambda d, k: mt.nanvar(_t(NANS), d, False, k),
        lambda d, k: np.nanvar(NANS, d, keepdims=k),
    ),
    (
        "nanstd",
        lambda d, k: mt.nanstd(_t(NANS), d, False, k),
        lambda d, k: np.nanstd(NANS, d, keepdims=k),
    ),
]


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("axes", AXIS_SETS)
@pytest.mark.parametrize("name,call,reference", CASES, ids=[case[0] for case in CASES])
def test_a_reduction_over_several_axes_matches_numpy(
    name, call, reference, axes, keepdim
):
    with np.errstate(invalid="ignore", divide="ignore"):
        want = np.asarray(reference(tuple(axes), keepdim))
    got = call(axes, keepdim).numpy()

    assert got.shape == want.shape, f"{name}{axes} keepdim={keepdim}"
    np.testing.assert_allclose(
        got.astype(np.float64), want.astype(np.float64), rtol=1e-12, equal_nan=True
    )


def test_nanmedian_over_several_axes_takes_the_lower_middle_value():
    """NumPy averages the two middle values and this does not -- the note on
    `median` gives reporting an index as the reason -- so the reference here is
    the sorted group rather than `np.nanmedian`."""
    for axes in AXIS_SETS:
        axes = [axis % NANS.ndim for axis in axes]
        got = mt.nanmedian(_t(NANS), axes).numpy()

        kept = [axis for axis in range(NANS.ndim) if axis not in axes]
        gathered = np.moveaxis(NANS, axes, range(-len(axes), 0))
        gathered = gathered.reshape(*[NANS.shape[axis] for axis in kept], -1)
        want = np.array(
            [
                (
                    np.sort(row[~np.isnan(row)])[(int((~np.isnan(row)).sum()) - 1) // 2]
                    if np.isfinite(row).any()
                    else np.nan
                )
                for row in gathered.reshape(-1, gathered.shape[-1])
            ]
        ).reshape(got.shape)
        np.testing.assert_allclose(got, want, equal_nan=True)


@pytest.mark.parametrize(
    "name,build",
    [
        ("amax", lambda z, d: mt.amax(z, d).sum()),
        ("amin", lambda z, d: mt.amin(z, d).sum()),
        ("nanamax", lambda z, d: mt.nanamax(z, d).sum()),
        ("amax keepdim", lambda z, d: mt.amax(z, d, True).sum()),
        ("ptp", lambda z, d: mt.ptp(z, d).sum()),
        ("average", lambda z, d: mt.average(z, d).sum()),
        ("quantile", lambda z, d: mt.quantile(z, 0.4, d).sum()),
        ("quantile[3]", lambda z, d: mt.quantile(z, [0.1, 0.5, 0.9], d).sum()),
        ("nanquantile[2]", lambda z, d: mt.nanquantile(z, [0.3, 0.7], d).sum()),
        ("nanmedian", lambda z, d: mt.nanmedian(z, d).sum()),
        ("nanvar", lambda z, d: mt.nanvar(z, d).sum()),
        ("nanstd", lambda z, d: mt.nanstd(z, d).sum()),
    ],
)
@pytest.mark.parametrize("axes", [[0, 1], [1, 2], [0, 2]])
def test_the_gradient_over_several_axes_is_the_numerical_one(name, build, axes):
    """The shape ops that gather the axes carry the gradient, so what is being
    checked is that gathering is the right thing to differentiate through --
    not a backward pass written a second time for the multi-axis case."""
    assert mt.gradcheck(lambda z: build(z, axes), [_t(VALUES, requires_grad=True)])


def test_a_tie_over_several_axes_is_split_over_all_of_it():
    """Gathering the axes and folding them one at a time give the same value
    and different gradients, which is why the axes are gathered."""
    tied = mt.Tensor([[3.0, 3.0], [3.0, 1.0]], requires_grad=True)
    mt.amax(tied, [0, 1]).backward()
    np.testing.assert_allclose(tied.grad.numpy(), [[1 / 3, 1 / 3], [1 / 3, 0.0]])

    folded = mt.Tensor([[3.0, 3.0], [3.0, 1.0]], requires_grad=True)
    mt.amax(mt.amax(folded, 1, True), 0).backward()
    np.testing.assert_allclose(folded.grad.numpy(), [[0.25, 0.25], [0.5, 0.0]])


def test_several_quantiles_report_a_gradient():
    """`quantile(x, [0.1, 0.9])` came back with `requires_grad` set and no
    `grad_fn`, so `backward()` treated it as a leaf: no gradient, no error."""
    several = _t(VALUES, requires_grad=True)
    mt.quantile(several, [0.25, 0.75], 1).sum().backward()
    assert several.grad is not None

    # And it is the sum of what each probability asks for on its own.
    apart = np.zeros_like(VALUES)
    for q in (0.25, 0.75):
        one = _t(VALUES, requires_grad=True)
        mt.quantile(one, q, 1).sum().backward()
        apart += one.grad.numpy()
    np.testing.assert_allclose(several.grad.numpy(), apart, rtol=1e-12)


def test_the_batched_quantile_still_answers_the_same_values():
    """The gradient path is a different computation -- one sorted pass per
    probability instead of one for all of them -- so it has to agree with the
    batched one it replaces, value for value."""
    for axes in ([1], [0, 2], None):
        want = mt.quantile(_t(VALUES), [0.1, 0.5, 0.9], axes).numpy()
        got = mt.quantile(_t(VALUES, requires_grad=True), [0.1, 0.5, 0.9], axes).numpy()
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    "name,call,alternative",
    [
        ("max", lambda t, d: mt.max(t, d), "amax"),
        ("min", lambda t, d: mt.min(t, d), "amin"),
        ("nanmax", lambda t, d: mt.nanmax(t, d), "nanamax"),
        ("nanmin", lambda t, d: mt.nanmin(t, d), "nanamin"),
        ("median", lambda t, d: mt.median(t, d), "nanmedian"),
        ("argmax", lambda t, d: mt.argmax(t, d), None),
        ("argmin", lambda t, d: mt.argmin(t, d), None),
        ("nanargmax", lambda t, d: mt.nanargmax(t, d), None),
        ("nanargmin", lambda t, d: mt.nanargmin(t, d), None),
        ("mode", lambda t, d: mt.mode(t, d), None),
    ],
)
def test_a_reduction_that_reports_an_index_says_why_it_takes_one_axis(
    name, call, alternative
):
    with pytest.raises(TypeError) as raised:
        call(_t(VALUES), [0, 1])
    message = str(raised.value)
    assert name in message
    assert "one axis" in message, message
    if alternative is not None:
        assert alternative in message, message

    # One axis still works, under either spelling of it.
    call(_t(VALUES), 1)
    call(_t(VALUES), -2)


def test_naming_every_axis_is_the_same_as_naming_none():
    for axes in ([0, 1, 2], [-1, -2, -3]):
        np.testing.assert_allclose(mt.amax(_t(VALUES), axes).numpy(), VALUES.max())
        np.testing.assert_allclose(mt.ptp(_t(VALUES), axes).numpy(), np.ptp(VALUES))
        assert mt.count_nonzero(_flags(), axes).numpy() == np.count_nonzero(FLAGS)


def test_a_repeated_axis_is_reduced_once():
    """`sum` already deduplicates, and the ops that reach it through the same
    normaliser inherit that rather than reducing an axis twice."""
    np.testing.assert_allclose(
        mt.amax(_t(VALUES), [0, 0, 1]).numpy(), np.max(VALUES, (0, 1))
    )
    np.testing.assert_allclose(
        mt.count_nonzero(_flags(), [1, 1]).numpy(), np.count_nonzero(FLAGS, 1)
    )


def test_an_axis_out_of_range_is_reported_as_one():
    for axes in ([0, 3], [-4, 0], [5]):
        with pytest.raises((IndexError, ValueError)):
            mt.amax(_t(VALUES), axes)


def test_weighted_average_over_several_axes_needs_matching_weights():
    """NumPy's rule: one-dimensional weights line up with a single reduced
    axis, and there is no single axis to line them up with here."""
    weights = _t(np.abs(VALUES) + 0.5)
    got = mt.average(_t(VALUES), [0, 1], weights).numpy()
    want = np.average(VALUES, axis=(0, 1), weights=np.abs(VALUES) + 0.5)
    np.testing.assert_allclose(got, want, rtol=1e-12)

    with pytest.raises(ValueError, match="shaped like the input"):
        mt.average(_t(VALUES), [0, 1], _t(np.arange(3.0)))


def test_the_numpy_compatibility_module_takes_a_tuple_too():
    """`numpy_compat` spells these NumPy's way, so an `axis` tuple is exactly
    what a caller reaching for it will pass."""
    for axes in ([0, 1], (1, 2)):
        np.testing.assert_allclose(
            mt.numpy_compat.max(_t(VALUES), axes).numpy(), np.max(VALUES, tuple(axes))
        )
        np.testing.assert_allclose(
            mt.numpy_compat.nanmin(_t(NANS), axes).numpy(),
            np.nanmin(NANS, tuple(axes)),
        )


def _reductions():
    """Every reduction on the public surface, under one name each."""
    import inspect

    found = {}
    for module in (mt, mt.functional):
        for name in sorted(dir(module)):
            if name.startswith("_"):
                continue
            function = getattr(module, name)
            if not callable(function) or inspect.isclass(function):
                continue
            try:
                parameters = inspect.signature(function).parameters
            except (ValueError, TypeError):
                continue
            if "dim" in parameters and "keepdim" in parameters:
                found.setdefault(name, function)
    return found


def test_the_only_reductions_that_take_one_axis_are_the_ones_reporting_an_index():
    """The rule, swept rather than restated: read the surface, call each
    reduction with two axes, and require every refusal to be an index
    reporter that says so. A reduction added later that quietly takes one
    axis fails here, which is the point -- that is how this list grew to
    twenty-two in the first place."""
    took, refused = [], []
    for name, function in _reductions().items():
        try:
            function(_t(VALUES), dim=0)
        except Exception:
            continue  # needs more than a tensor and a dim; covered above
        try:
            function(_t(VALUES), dim=[0, 1])
            took.append(name)
        except TypeError as error:
            refused.append((name, str(error)))
        except Exception as error:  # noqa: BLE001 - reported below
            refused.append((name, f"{type(error).__name__}: {error}"))

    assert len(took) > 20, f"only {len(took)} reductions took a list of axes"
    for name, message in refused:
        assert "reports an index" in message, f"{name}: {message}"
        assert "one axis" in message, f"{name}: {message}"
