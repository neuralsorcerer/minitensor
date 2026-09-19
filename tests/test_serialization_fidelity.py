# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Saving and loading must return the same bits, not merely the same numbers.

`test_serialization_roundtrip.py` checks that a dense MLP and a conv/batchnorm
stack survive a save and load. This widens that in the three directions where a
serialization bug hides without failing anything:

**Bit-for-bit rather than equal.** `np.array_equal` cannot see the difference
between `-0.0` and `0.0`, and reports every NaN as unequal to itself, so a
format that dropped a sign bit or normalised a NaN would pass a value
comparison. These tests compare the raw bytes.

**Values a text format cannot spell.** JSON has no NaN and no Infinity in its
grammar, which is the classic way weights come back as `null` or zero. A
subnormal and the largest finite float are the other two an encoder is liable
to round.

**Layers the existing tests do not reach.** Recurrent layers carry four
parameter tensors per direction, embeddings carry one large one, and the
existing coverage is dense and conv. A layer whose parameters are stored in a
different order, or whose count differs on reload, is a corrupted model that
still loads.

Nothing here was failing when it was written; it is the kind of thing that
fails quietly later.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn

FORMATS = ["json", "bin", "msgpack", "extension"]

# NaN and both infinities, both zeros, the smallest subnormal and the largest
# finite value -- every float a format is liable to mangle, in one row.
#
# The subnormal is float32's (about 1.4e-45), not float64's 5e-324, which
# underflows to zero on the way into a float32 tensor and would have made the
# subnormal case a test of nothing.
SMALLEST_SUBNORMAL = np.float32(1.401298464324817e-45)
LARGEST_FINITE = np.float32(3.4028234663852886e38)
AWKWARD = np.array(
    [np.nan, np.inf, -np.inf, -0.0, 0.0, SMALLEST_SUBNORMAL, LARGEST_FINITE, 1.5],
    dtype=np.float32,
)


def _bits(array):
    """The raw bytes, which is the only view that sees `-0.0` and NaN."""
    return np.asarray(array).copy().view(np.uint8)


def _roundtrip(build, path, fmt, seed=0):
    """Save a model built one way, load it into one built differently."""
    mt.manual_seed(seed)
    model = build()
    saved = [np.asarray(p).copy() for p in model.parameters()]

    model.save(str(path), fmt)
    mt.manual_seed(seed + 999)  # different initial parameters
    fresh = build()
    fresh.load_state_dict(type(model).load_state_from(str(path), fmt))
    return saved, [np.asarray(p).copy() for p in fresh.parameters()]


LAYERS = [
    ("dense", lambda: nn.Sequential([nn.DenseLayer(4, 3)])),
    ("embedding", lambda: nn.Sequential([nn.Embedding(10, 4)])),
    ("layer_norm", lambda: nn.Sequential([nn.LayerNorm(8)])),
    ("conv1d", lambda: nn.Sequential([nn.Conv1d(3, 4, 3)])),
    ("conv_transpose2d", lambda: nn.Sequential([nn.ConvTranspose2d(3, 4, 3)])),
    ("lstm", lambda: nn.LSTM(4, 8)),
    ("gru", lambda: nn.GRU(4, 8)),
    ("lstm_bidirectional", lambda: nn.LSTM(4, 8, bidirectional=True)),
]


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("name,build", LAYERS, ids=[n for n, _ in LAYERS])
def test_every_parameter_comes_back_bit_for_bit(tmp_path, fmt, name, build):
    saved, loaded = _roundtrip(build, tmp_path / f"{name}.{fmt}", fmt)

    assert len(saved) == len(loaded), (
        f"{name} saved {len(saved)} parameters and loaded {len(loaded)}; a model "
        f"that loses one still loads"
    )
    for index, (before, after) in enumerate(zip(saved, loaded)):
        assert before.shape == after.shape, f"{name} parameter {index}"
        np.testing.assert_array_equal(
            _bits(before), _bits(after), err_msg=f"{name} parameter {index}"
        )


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_dtype_survives_as_well_as_the_values(tmp_path, fmt, dtype):
    """A model reloaded at another precision is silently a different model."""
    with mt.default_dtype(dtype):
        mt.manual_seed(0)
        model = nn.Sequential([nn.DenseLayer(4, 3)])
        saved = [np.asarray(p).copy() for p in model.parameters()]

        model.save(str(tmp_path / f"m.{fmt}"), fmt)
        mt.manual_seed(999)
        fresh = nn.Sequential([nn.DenseLayer(4, 3)])
        fresh.load_state_dict(
            type(model).load_state_from(str(tmp_path / f"m.{fmt}"), fmt)
        )

        assert [p.dtype for p in fresh.parameters()] == [dtype] * len(saved)
        for before, after in zip(saved, fresh.parameters()):
            np.testing.assert_array_equal(_bits(before), _bits(np.asarray(after)))


@pytest.mark.parametrize("fmt", FORMATS)
def test_values_a_text_format_cannot_spell_survive(tmp_path, fmt):
    """NaN, both infinities, both zeros, a subnormal and the largest finite.

    JSON's grammar has no NaN and no Infinity, so this is where a weight comes
    back as `null`, or as zero, without anything raising.
    """
    mt.manual_seed(0)
    model = nn.Sequential([nn.DenseLayer(8, 8)])
    weight = list(model.parameters())[0]
    values = np.asarray(weight).copy()
    values[0, :] = AWKWARD
    weight.copy_(mt.Tensor(values, dtype="float32"))
    before = np.asarray(weight).copy()

    path = tmp_path / f"awkward.{fmt}"
    model.save(str(path), fmt)
    mt.manual_seed(999)
    fresh = nn.Sequential([nn.DenseLayer(8, 8)])
    fresh.load_state_dict(type(model).load_state_from(str(path), fmt))
    after = np.asarray(list(fresh.parameters())[0])

    np.testing.assert_array_equal(_bits(before), _bits(after))
    # Spelled out, because the bit comparison above would also pass if both
    # sides were mangled the same way by a shared encode/decode bug.
    row = after[0, :]
    assert np.isnan(row[0])
    assert row[1] == np.inf and row[2] == -np.inf
    assert np.signbit(row[3]) and not np.signbit(row[4])
    assert row[5] == SMALLEST_SUBNORMAL and row[5] > 0.0
    assert row[6] == LARGEST_FINITE


def test_a_nested_sequential_is_refused_rather_than_half_saved():
    """The one shape that is not supported says so, instead of losing layers."""
    with pytest.raises(TypeError, match="Nested Sequential"):
        nn.Sequential([nn.DenseLayer(4, 4), nn.Sequential([nn.DenseLayer(4, 4)])])
