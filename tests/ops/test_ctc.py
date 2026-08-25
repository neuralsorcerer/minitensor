# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Connectionist temporal classification.

The loss for a model that emits one distribution per input step and is trained
against a shorter, unaligned target. It is the total probability of every
alignment of the target to the input, which is exponentially many paths summed
by a dynamic program -- so the tests that matter most check the dynamic program
against the definition by *enumerating* those paths, at sizes where that is
possible.

The rest is arithmetic that is easy to get subtly wrong and hard to notice: a
recursion in the log domain, a backward recursion that has to line up with the
forward one, and a reduction that divides by the target length.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import minitensor as mt


def _t(a):
    return mt.Tensor.from_numpy(np.ascontiguousarray(a))


def _log_softmax(x):
    top = x.max(axis=-1, keepdims=True)
    return x - top - np.log(np.exp(x - top).sum(axis=-1, keepdims=True))


def _random_logs(rng, steps, batch, classes):
    return _log_softmax(rng.normal(0.0, 1.5, (steps, batch, classes)))


def _pad(targets):
    """Rows of unequal length, padded into the block the padded layout wants."""
    if isinstance(targets, np.ndarray):
        return targets.astype(np.int64)
    if targets and isinstance(targets[0], (list, tuple)):
        width = max(len(row) for row in targets)
        return np.array(
            [list(row) + [0] * (width - len(row)) for row in targets], dtype=np.int64
        )
    return np.asarray(targets, dtype=np.int64)


def _call(logs, targets, input_lengths, target_lengths, **kwargs):
    return mt.nn.ctc_loss(
        _t(logs),
        _t(_pad(targets)),
        _t(np.asarray(input_lengths, dtype=np.int64)),
        _t(np.asarray(target_lengths, dtype=np.int64)),
        **kwargs,
    )


def _one(logs, target, steps=None, **kwargs):
    """One batch element, as a `(steps, 1, classes)` input."""
    steps = logs.shape[0] if steps is None else steps
    return _call(logs, [list(target)], [steps], [len(target)], **kwargs)


# --------------------------------------------------------------------------
# Against the definition: every path, enumerated
# --------------------------------------------------------------------------


def _collapse(path, blank):
    """Merge adjacent equal classes, then drop the blank."""
    out, previous = [], None
    for symbol in path:
        if symbol != previous:
            if symbol != blank:
                out.append(symbol)
            previous = symbol
    return tuple(out)


def _by_enumeration(logs, target, blank=0):
    """-log of the total probability of every path collapsing to `target`."""
    steps, classes = logs.shape
    total = -np.inf
    for path in itertools.product(range(classes), repeat=steps):
        if _collapse(path, blank) == tuple(target):
            total = np.logaddexp(total, sum(logs[t, path[t]] for t in range(steps)))
    return -total


def test_the_collapse_is_the_one_from_the_paper():
    """Stated here because everything else is checked against it: adjacent
    equal symbols merge and *then* the blank goes, which is what lets a target
    repeat a symbol at all."""
    assert _collapse([1, 1, 0, 1, 2], blank=0) == (1, 1, 2)
    assert _collapse([1, 1, 1, 2, 2], blank=0) == (1, 2)
    assert _collapse([0, 0, 0], blank=0) == ()


@pytest.mark.parametrize("trial", range(25))
def test_the_dynamic_program_sums_the_same_paths_as_enumerating_them(trial):
    rng = np.random.default_rng(100 + trial)
    steps = int(rng.integers(1, 7))
    classes = int(rng.integers(2, 5))
    length = int(rng.integers(0, min(steps, 3) + 1))
    target = [int(v) for v in rng.integers(1, classes, length)]
    logs = _random_logs(rng, steps, 1, classes)

    want = _by_enumeration(logs[:, 0, :], target)
    got = _one(logs, target, reduction="none").numpy()[0]
    if np.isfinite(want):
        assert abs(got - want) < 1e-9
    else:
        assert not np.isfinite(got)


@pytest.mark.parametrize("trial", range(6))
def test_the_dynamic_program_enumerates_the_same_paths_across_a_batch(trial):
    """The same reference, pointed at more than one sample.

    At batch size one the time-major and sample-major index expressions
    coincide, so every single-sample test here -- including the enumeration
    above, which is otherwise the strongest check in the file -- is blind to a
    transposed input. And most of the batched tests compare two spellings that
    share that transpose, so they agree on the same wrong answer. This is the
    independent anchor the batched path was missing.
    """
    rng = np.random.default_rng(300 + trial)
    batch = int(rng.integers(2, 4))
    steps = int(rng.integers(2, 6))
    classes = int(rng.integers(2, 5))
    lengths = [int(rng.integers(0, min(steps, 3) + 1)) for _ in range(batch)]
    targets = [[int(v) for v in rng.integers(1, classes, n)] for n in lengths]
    logs = _random_logs(rng, steps, batch, classes)

    got = _call(logs, targets, [steps] * batch, lengths, reduction="none").numpy()
    for sample in range(batch):
        want = _by_enumeration(logs[:, sample, :], targets[sample])
        if np.isfinite(want):
            assert abs(got[sample] - want) < 1e-9, f"sample {sample}"
        else:
            assert not np.isfinite(got[sample]), f"sample {sample}"


def test_a_target_that_uses_every_step_has_exactly_one_path():
    """With `len(target) == steps` and no repeats, the only alignment is the
    target itself, so the loss is the plain sum of its log probabilities."""
    rng = np.random.default_rng(1)
    logs = _random_logs(rng, 4, 1, 5)
    target = [1, 2, 3, 4]
    want = -sum(logs[t, 0, target[t]] for t in range(4))
    assert _one(logs, target, reduction="none").numpy()[0] == pytest.approx(want)


def test_an_empty_target_is_the_all_blank_path():
    rng = np.random.default_rng(2)
    logs = _random_logs(rng, 6, 1, 4)
    want = -logs[:, 0, 0].sum()
    assert _one(logs, [], reduction="none").numpy()[0] == pytest.approx(want)


def test_a_repeat_needs_a_blank_between_the_two():
    """The reason the extended sequence exists. `[1, 1]` in two steps is
    unreachable -- `1 1` collapses to `1` -- and in three steps the only path
    is `1 blank 1`."""
    rng = np.random.default_rng(3)
    two = _random_logs(rng, 2, 1, 3)
    assert not np.isfinite(_one(two, [1, 1], reduction="none").numpy()[0])

    three = _random_logs(rng, 3, 1, 3)
    want = -(three[0, 0, 1] + three[1, 0, 0] + three[2, 0, 1])
    assert _one(three, [1, 1], reduction="none").numpy()[0] == pytest.approx(want)


def test_two_different_symbols_may_skip_the_blank_between_them():
    """The other half of the same rule: `[1, 2]` in two steps *is* reachable,
    by `1 2`, and in three steps has three paths."""
    rng = np.random.default_rng(4)
    two = _random_logs(rng, 2, 1, 3)
    want = -(two[0, 0, 1] + two[1, 0, 2])
    assert _one(two, [1, 2], reduction="none").numpy()[0] == pytest.approx(want)


def test_a_target_longer_than_the_input_is_unreachable():
    rng = np.random.default_rng(5)
    logs = _random_logs(rng, 2, 1, 6)
    assert not np.isfinite(_one(logs, [1, 2, 3, 4], reduction="none").numpy()[0])


# --------------------------------------------------------------------------
# The gradient
# --------------------------------------------------------------------------


def _grad_of(logs, targets, input_lengths, target_lengths, **kwargs):
    tensor = mt.Tensor.from_numpy(np.ascontiguousarray(logs), requires_grad=True)
    out = mt.nn.ctc_loss(
        tensor,
        _t(_pad(targets)),
        _t(np.asarray(input_lengths, dtype=np.int64)),
        _t(np.asarray(target_lengths, dtype=np.int64)),
        **kwargs,
    )
    out.backward()
    return out, tensor.grad.numpy()


@pytest.mark.parametrize("trial", range(8))
def test_the_gradient_matches_central_differences(trial):
    rng = np.random.default_rng(200 + trial)
    steps = int(rng.integers(3, 8))
    classes = int(rng.integers(2, 5))
    length = int(rng.integers(1, min(steps, 3) + 1))
    target = [int(v) for v in rng.integers(1, classes, length)]
    logs = _random_logs(rng, steps, 1, classes)
    if not np.isfinite(_one(logs, target, reduction="sum").item()):
        pytest.skip("unreachable target has no gradient")

    _, got = _grad_of(logs, [target], [steps], [length], reduction="sum")

    step = 1e-6
    want = np.zeros_like(logs)
    for i in range(steps):
        for k in range(classes):
            up, down = logs.copy(), logs.copy()
            up[i, 0, k] += step
            down[i, 0, k] -= step
            want[i, 0, k] = (
                _one(up, target, reduction="sum").item()
                - _one(down, target, reduction="sum").item()
            ) / (2 * step)
    assert np.abs(got - want).max() < 2e-5 * max(np.abs(want).max(), 1.0)


def test_each_step_of_the_gradient_sums_to_minus_one():
    """A consequence of the definition rather than a coincidence: the extended
    positions partition the total probability at every step, so the gradient
    with respect to that step's log probabilities sums to exactly -1. It is the
    single cheapest check that the forward and backward recursions line up."""
    rng = np.random.default_rng(6)
    logs = _random_logs(rng, 9, 3, 6)
    targets = [[1, 2, 3], [2, 2, 0], [5, 1, 0]]
    lengths = [3, 2, 2]
    inputs = [9, 7, 9]
    _, grad = _grad_of(logs, targets, inputs, lengths, reduction="sum")
    for sample, used in enumerate(inputs):
        for step in range(used):
            assert grad[step, sample].sum() == pytest.approx(-1.0, abs=1e-12)


def test_steps_beyond_the_input_length_get_no_gradient():
    """They did not enter the loss, so a non-zero gradient there would be the
    loop running past its own bound."""
    rng = np.random.default_rng(7)
    logs = _random_logs(rng, 10, 2, 4)
    _, grad = _grad_of(logs, [[1, 2], [3, 1]], [10, 6], [2, 2], reduction="sum")
    assert np.all(grad[6:, 1] == 0.0)
    # A zero *inside* the used steps is legitimate -- some class no path can
    # reach there. What separates used from unused is that a used step's
    # gradient sums to -1 and an unused one is entirely zero.
    for step in range(6):
        assert grad[step, 1].sum() == pytest.approx(-1.0, abs=1e-12)
    assert np.all(grad[:, 0].sum(axis=1) == pytest.approx(-1.0, abs=1e-12))


def test_the_gradient_is_negative_everywhere():
    """`-alpha beta / p` is a sum of probabilities, so every entry is a
    non-positive number -- there is no direction in which raising a log
    probability raises the loss."""
    rng = np.random.default_rng(8)
    logs = _random_logs(rng, 7, 2, 5)
    _, grad = _grad_of(logs, [[1, 2, 3], [4, 4, 0]], [7, 7], [3, 2], reduction="sum")
    assert np.all(grad <= 0.0)


def test_nothing_is_asked_of_the_targets():
    """They are indices; there is nothing to differentiate."""
    logs = _random_logs(np.random.default_rng(9), 5, 1, 4)
    tensor = mt.Tensor.from_numpy(np.ascontiguousarray(logs), requires_grad=True)
    out = mt.nn.ctc_loss(
        tensor,
        _t(np.array([[1, 2]], dtype=np.int64)),
        _t(np.array([5], dtype=np.int64)),
        _t(np.array([2], dtype=np.int64)),
    )
    assert out.requires_grad
    assert not _one(logs, [1, 2]).requires_grad


# --------------------------------------------------------------------------
# Reductions
# --------------------------------------------------------------------------


def test_none_gives_one_loss_per_batch_element():
    rng = np.random.default_rng(10)
    logs = _random_logs(rng, 6, 3, 4)
    out = _call(logs, [[1, 2], [3, 1], [2, 2]], [6, 6, 6], [2, 2, 2], reduction="none")
    assert out.numpy().shape == (3,)


def test_sum_is_the_total_of_none():
    rng = np.random.default_rng(11)
    logs = _random_logs(rng, 6, 3, 4)
    args = (logs, [[1, 2], [3, 1], [2, 3]], [6, 5, 6], [2, 2, 2])
    each = _call(*args, reduction="none").numpy()
    assert _call(*args, reduction="sum").item() == pytest.approx(each.sum())


def test_mean_divides_each_loss_by_its_own_target_length():
    """Not the mean of what `none` returns. A long target has more steps to
    get wrong, so without the division a batch's average would drift with the
    target lengths in it rather than with how well the model did."""
    rng = np.random.default_rng(12)
    logs = _random_logs(rng, 8, 3, 5)
    args = (logs, [[1, 2, 3], [4, 1, 0], [2, 0, 0]], [8, 8, 8], [3, 2, 1])
    each = _call(*args, reduction="none").numpy()
    want = (each / np.array([3, 2, 1])).mean()
    assert _call(*args, reduction="mean").item() == pytest.approx(want)


def test_an_empty_target_divides_by_one_rather_than_by_zero():
    """Target lengths [3, 0], not [1, 0]: with [1, 0] the clamped divisor is
    [1, 1], which is the same as not dividing at all, so the test would pass
    against an implementation that skipped the division entirely -- while being
    named after the division."""
    rng = np.random.default_rng(13)
    logs = _random_logs(rng, 6, 2, 3)
    args = (logs, [[1, 2, 1], []], [6, 6], [3, 0])
    each = _call(*args, reduction="none").numpy()
    want = (each / np.array([3, 1])).mean()
    assert _call(*args, reduction="mean").item() == pytest.approx(want)
    # And the divisor really is doing something here.
    assert want != pytest.approx(each.mean())


def test_mean_is_not_the_mean_of_none():
    """The convention stated directly. `mean` divides each loss by its own
    target length first, so the two coincide only when every target has length
    one -- and a batch of unequal targets is exactly when it matters."""
    rng = np.random.default_rng(40)
    logs = _random_logs(rng, 9, 3, 5)
    args = (logs, [[1, 2, 3, 4], [2, 1], [3]], [9, 9, 9], [4, 2, 1])
    each = _call(*args, reduction="none").numpy()
    averaged = _call(*args, reduction="mean").item()
    assert averaged == pytest.approx((each / np.array([4, 2, 1])).mean())
    assert averaged != pytest.approx(each.mean())


def test_the_reduction_scales_the_gradient_the_same_way():
    rng = np.random.default_rng(14)
    logs = _random_logs(rng, 7, 2, 4)
    args = ([[1, 2, 3], [2, 1, 0]], [7, 7], [3, 2])
    _, summed = _grad_of(logs, *args, reduction="sum")
    _, averaged = _grad_of(logs, *args, reduction="mean")
    want = summed.copy()
    want[:, 0] /= 3 * 2
    want[:, 1] /= 2 * 2
    assert np.allclose(averaged, want, rtol=1e-12, atol=0)


def test_an_upstream_gradient_reaches_each_batch_element_separately():
    """`none` returns a vector, so its upstream gradient is one number per
    batch element and has to meet the batch axis of a `(steps, batch, classes)`
    gradient -- not the axis that broadcasting from the right would pick."""
    rng = np.random.default_rng(15)
    logs = _random_logs(rng, 5, 3, 4)
    tensor = mt.Tensor.from_numpy(np.ascontiguousarray(logs), requires_grad=True)
    out = mt.nn.ctc_loss(
        tensor,
        _t(np.array([[1, 2], [3, 1], [2, 3]], dtype=np.int64)),
        _t(np.array([5, 5, 5], dtype=np.int64)),
        _t(np.array([2, 2, 2], dtype=np.int64)),
        reduction="none",
    )
    out.backward(_t(np.array([1.0, 10.0, 100.0])))
    weighted = tensor.grad.numpy()

    _, plain = _grad_of(
        logs,
        [[1, 2], [3, 1], [2, 3]],
        [5, 5, 5],
        [2, 2, 2],
        reduction="sum",
    )
    for sample, factor in enumerate([1.0, 10.0, 100.0]):
        assert np.allclose(weighted[:, sample], factor * plain[:, sample])


# --------------------------------------------------------------------------
# Unreachable targets
# --------------------------------------------------------------------------


def test_zero_infinity_replaces_the_loss_and_the_gradient():
    """One sample the input is too short to spell would otherwise carry the
    whole batch's gradient to infinity."""
    rng = np.random.default_rng(16)
    logs = _random_logs(rng, 4, 2, 5)
    args = ([[1, 2], [1, 2, 3, 4, 1]], [4, 4], [2, 5])
    loose = _call(logs, *args, reduction="none").numpy()
    assert np.isfinite(loose[0]) and not np.isfinite(loose[1])

    fixed = _call(logs, *args, reduction="none", zero_infinity=True).numpy()
    assert fixed[0] == pytest.approx(loose[0])
    assert fixed[1] == 0.0

    _, grad = _grad_of(logs, *args, reduction="sum", zero_infinity=True)
    assert np.all(np.isfinite(grad))
    assert np.all(grad[:, 1] == 0.0)
    assert not np.all(grad[:, 0] == 0.0)


def test_an_unreachable_target_poisons_the_sum_without_it():
    rng = np.random.default_rng(17)
    logs = _random_logs(rng, 3, 2, 5)
    out = _call(logs, [[1, 2], [1, 2, 3, 4]], [3, 3], [2, 4], reduction="sum")
    assert not np.isfinite(out.item())


def test_an_unreachable_target_asks_for_no_gradient_of_its_own():
    """Its loss is infinite, so there is no slope to report -- and the sample
    next to it in the batch still gets the gradient it earned."""
    rng = np.random.default_rng(38)
    logs = _random_logs(rng, 4, 2, 5)
    out, grad = _grad_of(
        logs, [[1, 2], [1, 2, 3, 4, 1]], [4, 4], [2, 5], reduction="sum"
    )
    assert not np.isfinite(out.item())
    assert np.all(np.isfinite(grad))
    assert np.all(grad[:, 1] == 0.0)
    for step in range(4):
        assert grad[step, 0].sum() == pytest.approx(-1.0, abs=1e-12)


def test_an_input_length_of_zero_asks_for_no_gradient_either():
    """The zero-input sample is given a target it cannot spell, so this covers
    the unreachable case too. With a zero-length target its loss would be zero
    either way, and the test would pass against an implementation where an
    empty input admitted any target at all."""
    rng = np.random.default_rng(39)
    logs = _random_logs(rng, 3, 2, 4)
    out, grad = _grad_of(
        logs, [[1], [1]], [0, 3], [1, 1], reduction="sum", zero_infinity=True
    )
    assert np.all(grad[:, 0] == 0.0)
    assert not np.all(grad[:, 1] == 0.0)

    # Without zero_infinity the same sample is infinite, which is the fact the
    # zeroed gradient stands in for.
    loose = _call(logs, [[1], [1]], [0, 3], [1, 1], reduction="none").numpy()
    assert not np.isfinite(loose[0])
    assert np.isfinite(loose[1])


def test_an_input_length_of_zero_admits_only_an_empty_target():
    rng = np.random.default_rng(18)
    logs = _random_logs(rng, 3, 2, 4)
    out = _call(logs, [[0], [1]], [0, 0], [0, 1], reduction="none").numpy()
    assert out[0] == 0.0
    assert not np.isfinite(out[1])


# --------------------------------------------------------------------------
# Shapes, layouts and dtypes
# --------------------------------------------------------------------------


def test_the_two_target_layouts_agree():
    """A padded `(batch, length)` block and the rows concatenated are the same
    question. The second is what a caller with wildly uneven targets wants."""
    rng = np.random.default_rng(19)
    logs = _random_logs(rng, 8, 3, 5)
    padded = np.array([[1, 2, 3], [4, 1, 0], [2, 0, 0]], dtype=np.int64)
    flat = np.array([1, 2, 3, 4, 1, 2], dtype=np.int64)
    lengths = [3, 2, 1]
    assert np.allclose(
        _call(logs, padded, [8, 8, 8], lengths, reduction="none").numpy(),
        _call(logs, flat, [8, 8, 8], lengths, reduction="none").numpy(),
    )


def test_the_padding_beyond_a_target_length_is_ignored():
    rng = np.random.default_rng(20)
    logs = _random_logs(rng, 6, 1, 5)
    short = _call(logs, [[1, 2, 0, 0]], [6], [2], reduction="none").numpy()
    other = _call(logs, [[1, 2, 4, 3]], [6], [2], reduction="none").numpy()
    assert short == pytest.approx(other)


def test_steps_beyond_an_input_length_are_ignored():
    rng = np.random.default_rng(21)
    logs = _random_logs(rng, 9, 1, 4)
    used = _call(logs, [[1, 2]], [5], [2], reduction="none").numpy()[0]
    trimmed = _call(logs[:5], [[1, 2]], [5], [2], reduction="none").numpy()[0]
    assert used == pytest.approx(trimmed)


def test_each_batch_element_is_independent():
    rng = np.random.default_rng(22)
    logs = _random_logs(rng, 7, 4, 5)
    targets = [[1, 2], [3, 4], [2, 2], [4, 1]]
    together = _call(logs, targets, [7] * 4, [2] * 4, reduction="none").numpy()
    for sample, target in enumerate(targets):
        alone = _one(logs[:, sample : sample + 1, :], target, reduction="none").numpy()[
            0
        ]
        assert together[sample] == pytest.approx(alone)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_both_floating_dtypes(dtype):
    rng = np.random.default_rng(23)
    logs = _random_logs(rng, 6, 2, 4).astype(dtype)
    out = _call(logs, [[1, 2], [3, 1]], [6, 6], [2, 2], reduction="none")
    assert out.numpy().dtype == dtype
    exact = _call(
        logs.astype(np.float64), [[1, 2], [3, 1]], [6, 6], [2, 2], reduction="none"
    )
    assert np.allclose(out.numpy(), exact.numpy(), rtol=1e-6)


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_both_integer_dtypes_for_the_indices(index_dtype):
    rng = np.random.default_rng(24)
    logs = _random_logs(rng, 5, 1, 4)
    out = mt.nn.ctc_loss(
        _t(logs),
        _t(np.array([[1, 2]], dtype=index_dtype)),
        _t(np.array([5], dtype=index_dtype)),
        _t(np.array([2], dtype=index_dtype)),
        reduction="none",
    )
    assert out.numpy() == pytest.approx(_one(logs, [1, 2], reduction="none").numpy())


def test_the_blank_need_not_be_class_zero():
    """It is a convention, not a rule, and the recursion has to read it from
    the argument rather than assume the first class."""
    rng = np.random.default_rng(25)
    logs = _random_logs(rng, 5, 1, 4)
    # New class j holds old class `order[j]`, so the old blank 0 lands at 3 and
    # the old symbols 1 and 2 land at 0 and 1.
    order = [1, 2, 3, 0]
    shifted = logs[:, :, order]
    at_zero = _one(logs, [1, 2], reduction="none").numpy()[0]
    at_three = _call(shifted, [[0, 1]], [5], [2], blank=3, reduction="none").numpy()[0]
    assert at_three == pytest.approx(at_zero)


# --------------------------------------------------------------------------
# What log space is for
# --------------------------------------------------------------------------


def test_a_long_sequence_does_not_underflow():
    """A path probability is a product of `steps` numbers below one. Two
    thousand of them underflow `f64` many times over, so an implementation that
    multiplied probabilities would return infinity here -- every path having
    rounded to zero."""
    rng = np.random.default_rng(26)
    logs = _random_logs(rng, 2000, 1, 8)
    out = _one(logs, [1, 2, 3, 4, 5], reduction="none").numpy()[0]
    assert np.isfinite(out)
    assert out > 0.0
    _, grad = _grad_of(logs, [[1, 2, 3, 4, 5]], [2000], [5], reduction="sum")
    assert np.all(np.isfinite(grad))
    assert grad[1000].sum() == pytest.approx(-1.0, abs=1e-10)


def test_a_zero_probability_class_does_not_produce_a_nan():
    """`log 0` is `-inf`, and `-inf` entering the recursion has to stay a
    perfectly ordinary "no path through here" rather than becoming a NaN."""
    with np.errstate(divide="ignore"):
        logs = np.log(
            np.array([[[0.0, 0.5, 0.5]], [[0.5, 0.0, 0.5]], [[0.4, 0.6, 0.0]]])
        )
    out = _one(logs, [1, 2], reduction="none").numpy()[0]
    assert np.isfinite(out)
    _, grad = _grad_of(logs, [[1, 2]], [3], [2], reduction="sum")
    assert not np.any(np.isnan(grad))


# --------------------------------------------------------------------------
# What it refuses
# --------------------------------------------------------------------------


def test_a_target_may_not_contain_the_blank():
    """The blank stands for emitting nothing, so a target asking for one has no
    reading at all -- better an error than a number."""
    logs = _random_logs(np.random.default_rng(27), 5, 1, 4)
    with pytest.raises(Exception, match="blank"):
        _call(logs, [[1, 0]], [5], [2])


def test_a_target_class_must_exist():
    logs = _random_logs(np.random.default_rng(28), 5, 1, 4)
    with pytest.raises(Exception, match="outside"):
        _call(logs, [[1, 9]], [5], [2])


def test_the_blank_class_must_exist():
    logs = _random_logs(np.random.default_rng(29), 5, 1, 4)
    with pytest.raises(Exception, match="outside"):
        _call(logs, [[1, 2]], [5], [2], blank=7)


def test_an_input_length_may_not_exceed_the_steps_provided():
    logs = _random_logs(np.random.default_rng(30), 5, 1, 4)
    with pytest.raises(Exception, match="exceeds"):
        _call(logs, [[1, 2]], [9], [2])


def test_a_target_length_may_not_exceed_the_padded_width():
    logs = _random_logs(np.random.default_rng(31), 5, 1, 4)
    with pytest.raises(Exception, match="exceeds"):
        _call(logs, [[1, 2]], [5], [4])


def test_the_concatenated_targets_must_add_up():
    logs = _random_logs(np.random.default_rng(32), 5, 2, 4)
    with pytest.raises(Exception, match="add up"):
        _call(logs, np.array([1, 2, 3], dtype=np.int64), [5, 5], [2, 2])


def test_log_probs_must_be_three_dimensional():
    logs = _random_logs(np.random.default_rng(33), 5, 1, 4)
    with pytest.raises(Exception, match="steps, batch, classes"):
        _call(logs[:, 0, :], [[1, 2]], [5], [2])


def test_the_lengths_must_be_one_per_batch_element():
    logs = _random_logs(np.random.default_rng(34), 5, 2, 4)
    with pytest.raises(Exception, match="input_lengths"):
        _call(logs, [[1, 2], [2, 1]], [5], [2, 2])
    with pytest.raises(Exception, match="target_lengths"):
        _call(logs, [[1, 2], [2, 1]], [5, 5], [2])


def test_the_lengths_may_not_be_negative():
    logs = _random_logs(np.random.default_rng(35), 5, 1, 4)
    with pytest.raises(Exception, match="negative"):
        _call(logs, [[1, 2]], [-1], [2])


def test_an_unknown_reduction_is_refused():
    logs = _random_logs(np.random.default_rng(36), 5, 1, 4)
    with pytest.raises(Exception, match="reduction"):
        _call(logs, [[1, 2]], [5], [2], reduction="average")


# --------------------------------------------------------------------------
# What it is for
# --------------------------------------------------------------------------


def test_training_moves_the_loss_down():
    """The whole point, checked end to end: gradient descent on the scores
    behind a log-softmax lowers the loss."""
    rng = np.random.default_rng(37)
    scores = rng.normal(0.0, 0.5, (12, 1, 5))
    target = [[1, 2, 3, 1]]
    before = None
    for _ in range(60):
        tensor = mt.Tensor.from_numpy(np.ascontiguousarray(scores), requires_grad=True)
        logs = mt.log_softmax(tensor, -1)
        loss = mt.nn.ctc_loss(
            logs,
            _t(np.array(target, dtype=np.int64)),
            _t(np.array([12], dtype=np.int64)),
            _t(np.array([4], dtype=np.int64)),
            reduction="sum",
        )
        loss.backward()
        if before is None:
            before = loss.item()
        scores = scores - 0.5 * tensor.grad.numpy()
    assert loss.item() < before / 10
