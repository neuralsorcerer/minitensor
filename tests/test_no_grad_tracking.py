# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Under `no_grad()`, an operation's output must not claim to be tracked.

`add_to_graph` already refuses to record under `no_grad`, so no node was ever
created. What leaked was the *flag*: several ops set `requires_grad` on their
output by copying it from the input, either through `view` (which clones the
input wholesale) or through `requires_grad_`, which deliberately ignores grad
mode so that marking a leaf trainable inside `no_grad` still works.

The result was a tensor with `requires_grad=True` and a `grad_fn` but no graph
node -- it looked tracked and back-propagated to nothing. Feeding one into a
later graph made it a spurious leaf that accumulated a `.grad` of its own while
the real input got none. PyTorch returns `requires_grad=False` here.

`reshape`, `flatten`, `ravel`, `squeeze`, `unsqueeze`, `repeat`, `roll`,
`norm`, `mae_loss`, `huber_loss` and `binary_cross_entropy_with_logits` were
affected; `transpose`, `mul`, `sum` and `mse_loss` were already correct, which
is what made the inconsistency easy to miss.
"""

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F


@pytest.fixture(autouse=True)
def _clean_graph():
    mt.clear_autograd_graph()
    yield
    mt.clear_autograd_graph()


def _tracked(shape=(2, 3)):
    values = np.arange(1, int(np.prod(shape)) + 1, dtype=np.float64).reshape(shape)
    return mt.Tensor(values, dtype="float64", requires_grad=True)


# Shapes are chosen so each op does real work; on a 1-D input several of these
# are identity operations that return the input itself and prove nothing.
SHAPE_OPS = [
    ("reshape", (2, 3), lambda t: t.reshape((6,))),
    ("flatten", (2, 3), lambda t: t.flatten()),
    ("ravel", (2, 3), lambda t: t.ravel()),
    ("squeeze", (1, 3), lambda t: t.squeeze()),
    ("unsqueeze", (2, 3), lambda t: t.unsqueeze(0)),
    ("repeat", (2, 3), lambda t: t.repeat([2, 1])),
    ("roll", (6,), lambda t: t.roll(1, 0)),
    ("norm", (2, 3), lambda t: t.norm()),
    ("transpose", (2, 3), lambda t: t.transpose(0, 1)),
    ("mul", (2, 3), lambda t: t * 2.0),
    ("sum", (2, 3), lambda t: t.sum()),
]


@pytest.mark.parametrize("name,shape,op", SHAPE_OPS, ids=[c[0] for c in SHAPE_OPS])
def test_no_grad_outputs_are_untracked(name, shape, op):
    x = _tracked(shape)
    with mt.no_grad():
        out = op(x)
    assert out.requires_grad is False, f"{name} returned a tracked tensor"


LOSSES = [
    ("mae_loss", lambda p, t: F.l1_loss(p, t)),
    ("mse_loss", lambda p, t: F.mse_loss(p, t)),
    ("huber_loss", lambda p, t: F.huber_loss(p, t)),
    (
        "bce_with_logits",
        lambda p, t: F.binary_cross_entropy_with_logits(
            p, mt.Tensor(np.zeros((2, 3)), dtype="float64")
        ),
    ),
]


@pytest.mark.parametrize("name,loss", LOSSES, ids=[c[0] for c in LOSSES])
def test_losses_are_untracked_under_no_grad(name, loss):
    # These gate on their *inputs* rather than on the loss tensor, because the
    # loss itself is computed detached. That is correct, but it has to respect
    # the ambient grad mode as well.
    predictions, targets = _tracked(), _tracked()
    with mt.no_grad():
        out = loss(predictions, targets)
    assert out.requires_grad is False, f"{name} returned a tracked tensor"


def test_the_orphan_this_produced():
    """The concrete damage: a spurious leaf that steals the gradient."""
    x = _tracked((2, 3))
    with mt.no_grad():
        view = x.flatten()

    assert view.requires_grad is False

    # Feeding it into a later graph must not resurrect it as a trainable leaf.
    # Nothing in the chain requires grad now, so backward has no work and says
    # so, where before it silently handed `view` a gradient of its own.
    downstream = view * 2.0
    assert downstream.requires_grad is False
    with pytest.raises(Exception):
        downstream.sum().backward()
    assert view.grad is None
    assert x.grad is None  # no_grad severed the history, as intended


def test_an_op_that_returns_its_input_unchanged_keeps_the_input_flag():
    # `contiguous`, `cpu` and `to(same dtype)` are no-ops on a conforming
    # tensor and hand back the input, so the flag they report is the input's
    # own rather than a leak. A real conversion is untracked.
    x = _tracked()
    with mt.no_grad():
        assert x.contiguous().requires_grad is True
        assert x.to("float64").requires_grad is True
        assert x.to("float32").requires_grad is False


# Ops that hand back the input itself when it already conforms, so the flag they
# report is the input's own rather than a new tracked tensor. The exemption is
# not taken on trust: `test_an_op_that_returns_its_input_unchanged...` asserts
# that when these ops actually convert something, the result is untracked.
_RETURNS_INPUT_UNCHANGED = {
    "contiguous",
    "cpu",
    "to",
    "clone",
    "detach",
    # In-place mutators return the receiver, so there is no new tensor whose
    # tracking could be wrong -- the flag is the caller's own.
    "fill_",
    "copy_",
    "requires_grad_",
    "detach_",
}


def test_no_op_under_no_grad_records_a_graph_node():
    """Sweep the surface: nothing may become reachable by backward."""
    offenders = []
    source = mt.Tensor(np.full(6, 2.0), dtype="float64")

    for name in sorted(n for n in dir(mt.Tensor) if not n.startswith("_")):
        probe = mt.Tensor(np.arange(1, 7, dtype=np.float64), dtype="float64")
        if not callable(getattr(probe, name, None)):
            continue
        for args in ((), (0,), (1.0,), (source,)):
            x = _tracked((6,))
            try:
                with mt.no_grad():
                    out = getattr(x, name)(*args)
            except TypeError:
                continue  # wrong arity, try the next
            except Exception:
                break  # rejected for another reason; nothing was produced
            outputs = out if isinstance(out, tuple) else (out,)
            for produced in outputs:
                if name in _RETURNS_INPUT_UNCHANGED:
                    continue
                if not getattr(produced, "requires_grad", False):
                    continue
                try:
                    produced.sum().backward()
                    reachable = x.grad is not None
                except Exception:
                    reachable = False
                finally:
                    mt.clear_autograd_graph()
                if reachable:
                    offenders.append(name)
            break

    assert not offenders, (
        "these ops built a live backward edge despite no_grad: "
        + ", ".join(sorted(set(offenders)))
    )
