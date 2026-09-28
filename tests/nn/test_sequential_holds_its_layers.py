# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A `Sequential` holds the layers it is given, not copies of them.

It held a clone of each. The clone shared the parameters' storage but had its
own flags and buffers, so the layer the caller kept was half of the same one:

- `model = Sequential([backbone, head]); backbone.requires_grad_(False)`
  froze the caller's `backbone` and left `model` training it;
- `backbone.eval()` left its dropout running inside `model`;
- `backbone`'s running statistics stopped at the values they had when it was
  added, so evaluating it alone used statistics `model` had moved on from.

Now a change made through either one is made to both, because there is only
one. That makes a module's place in a model a tree, which is checked: a module
belongs to at most one `Sequential`, once, and never to one inside itself.
"""

from __future__ import annotations

import copy
import gc

import numpy as np
import pytest

import minitensor as mt

nn = mt.nn
optim = mt.optim


def _snapshot(module):
    return {name: t.numpy().copy() for name, t in module.state_dict().items()}


def _changed(before, after):
    return sorted(n for n in before if not np.array_equal(before[n], after[n]))


def _inputs(rows=8, seed=0):
    rng = np.random.default_rng(seed)
    return mt.from_numpy(rng.standard_normal((rows, 4)).astype(np.float32))


def _model():
    mt.manual_seed(0)
    backbone = nn.Sequential([nn.DenseLayer(4, 6), nn.BatchNorm1d(6), nn.Dropout(0.5)])
    head = nn.DenseLayer(6, 2)
    return nn.Sequential([backbone, head]), backbone, head


def _train_step(optimizer, model):
    optimizer.zero_grad()
    (model(_inputs()) ** 2).sum().backward()
    optimizer.step()


# --- one layer, reached two ways --------------------------------------------


def test_freezing_a_part_after_building_the_model_freezes_it_in_the_model():
    model, backbone, head = _model()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.1)
    backbone.requires_grad_(False)

    assert [p.requires_grad for p in model.parameters()] == [False] * 4 + [True] * 2
    before_backbone, before_head = _snapshot(backbone), _snapshot(head)
    _train_step(optimizer, model)
    assert _changed(before_backbone, _snapshot(backbone)) == [
        "1.running_mean",
        "1.running_var",
    ]
    assert _changed(before_head, _snapshot(head)) == ["bias", "weight"]


def test_switching_a_part_to_eval_switches_it_in_the_model():
    model, backbone, _ = _model()
    backbone.eval()
    x = _inputs()
    np.testing.assert_array_equal(model(x).numpy(), model(x).numpy())


def test_running_statistics_moved_by_the_model_are_the_parts():
    model, backbone, _ = _model()
    model(_inputs() + 3.0)
    assert np.any(backbone.state_dict()["1.running_mean"].numpy())


def test_a_part_trained_through_the_model_runs_alone_with_those_weights():
    model, backbone, head = _model()
    model.eval()
    x = _inputs()
    np.testing.assert_allclose(head(backbone(x)).numpy(), model(x).numpy(), rtol=1e-6)


def test_loading_into_a_part_is_loading_into_the_model():
    model, backbone, _ = _model()
    mt.manual_seed(5)
    other = nn.Sequential([nn.DenseLayer(4, 6), nn.BatchNorm1d(6), nn.Dropout(0.5)])
    backbone.load_state_dict(other.state_dict())

    for name, values in _snapshot(other).items():
        np.testing.assert_array_equal(_snapshot(model)[f"0.{name}"], values)


def test_add_module_holds_the_layer_too():
    model = nn.Sequential([nn.DenseLayer(4, 3)])
    extra = nn.DenseLayer(3, 2)
    model.add_module("extra", extra)
    extra.requires_grad_(False)
    assert [p.requires_grad for p in model.parameters()] == [True, True, False, False]


def test_three_levels_deep():
    inner = nn.Sequential([nn.DenseLayer(4, 4)])
    middle = nn.Sequential([inner, nn.ReLU()])
    outer = nn.Sequential([middle, nn.DenseLayer(4, 2)])
    inner.requires_grad_(False)
    assert [p.requires_grad for p in outer.parameters()] == [False, False, True, True]


# --- a tree, checked ---------------------------------------------------------


def test_a_module_belongs_to_one_sequential():
    layer = nn.DenseLayer(4, 3)
    holder = nn.Sequential([layer])  # noqa: F841 -- kept alive, it holds `layer`
    with pytest.raises(ValueError, match="already belongs to a Sequential"):
        nn.Sequential([layer])
    with pytest.raises(ValueError, match="already belongs to a Sequential"):
        nn.Sequential([]).add_module("again", layer)


def test_a_module_belongs_to_a_sequential_once():
    relu = nn.ReLU()
    with pytest.raises(ValueError, match="only once"):
        nn.Sequential([relu, relu])
    nn.Sequential([relu])  # the refused one left it free


def test_a_sequential_cannot_hold_itself_or_what_holds_it():
    inner = nn.Sequential([nn.ReLU()])
    outer = nn.Sequential([inner])
    with pytest.raises(ValueError, match="cannot hold itself"):
        inner.add_module("loop", inner)
    with pytest.raises(ValueError, match="cannot hold itself"):
        inner.add_module("loop", outer)


def test_a_refused_constructor_leaves_every_layer_free():
    first, taken = nn.DenseLayer(4, 3), nn.ReLU()
    holder = nn.Sequential([taken])
    with pytest.raises(ValueError):
        nn.Sequential([first, taken])
    nn.Sequential([first])


def test_a_dropped_sequential_lets_its_layers_go():
    layer = nn.DenseLayer(4, 3)
    model = nn.Sequential([layer])
    del model
    gc.collect()
    nn.Sequential([layer])


# --- copies are copies -------------------------------------------------------


def test_a_deep_copy_of_the_model_is_linked_to_nothing():
    model, backbone, _ = _model()
    copied = copy.deepcopy(model)
    optimizer = optim.SGD(copied.parameters(), lr=0.1)
    before = _snapshot(backbone)

    backbone.requires_grad_(False)
    assert all(p.requires_grad for p in copied.parameters())
    _train_step(optimizer, copied)
    for name, values in before.items():
        np.testing.assert_array_equal(_snapshot(backbone)[name], values)


def test_a_part_of_a_copied_model_can_be_added_elsewhere():
    """The copy's parts are its own, so the originals are still in `model`
    and a copy of a part is free."""
    _, backbone, _ = _model()
    nn.Sequential([copy.deepcopy(backbone)])
