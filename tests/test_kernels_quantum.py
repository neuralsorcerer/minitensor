# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`kernels.apply_gate` and friends, against complex arithmetic done in NumPy.

The kernel is a strided butterfly over pairs of amplitudes, written directly
because there is no elementwise way to say it. So the things worth checking are
the ones a butterfly gets wrong: which pairs it forms (the striding), whether
the complex multiply is a complex multiply, whether a batch stays separate, and
whether the backward really is the adjoint gate rather than the gate.

The reference throughout is NumPy complex arithmetic on the same amplitudes,
which is the independent implementation this has to agree with.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import minitensor as mt
from minitensor.kernels import apply_gate, expect_z, prefix_trace, probabilities

_HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2.0)
_PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


@pytest.fixture(autouse=True)
def _clean_graph():
    yield
    mt.clear_autograd_graph()


def _state(amplitudes, requires_grad=False):
    """`(..., 2**q, 2)` from an array of `(real, imag)` pairs."""
    return mt.Tensor(
        np.ascontiguousarray(np.asarray(amplitudes, dtype=np.float64)),
        dtype="float64",
        requires_grad=requires_grad,
    )


def _complex(tensor):
    values = np.asarray(tensor)
    return values[..., 0] + 1j * values[..., 1]


def _pairs(complex_state):
    return np.stack([complex_state.real, complex_state.imag], axis=-1)


def _reference_gate(complex_state, matrix, qubit):
    """Apply a one-qubit gate the slow, obvious way."""
    out = complex_state.copy()
    stride = 1 << qubit
    for index in range(complex_state.size):
        if index & stride:
            continue
        lo, hi = index, index + stride
        out[lo], out[hi] = matrix @ np.array([complex_state[lo], complex_state[hi]])
    return out


# --- the value ----------------------------------------------------------------


def test_hadamard_makes_an_equal_superposition():
    out = apply_gate(_state([[1.0, 0.0], [0.0, 0.0]]), "h", 0)
    h = 1.0 / math.sqrt(2.0)
    np.testing.assert_allclose(np.asarray(out), [[h, 0.0], [h, 0.0]], atol=1e-15)


def test_hadamard_is_its_own_inverse():
    start = _state([[0.6, 0.0], [0.0, 0.8]])
    twice = apply_gate(apply_gate(start, "h", 0), "h", 0)
    np.testing.assert_allclose(np.asarray(twice), np.asarray(start), atol=1e-15)


@pytest.mark.parametrize("qubit", [0, 1, 2])
@pytest.mark.parametrize(
    "name,matrix", [("h", _HADAMARD), ("x", _PAULI_X), ("z", _PAULI_Z)]
)
def test_a_named_gate_matches_complex_arithmetic(qubit, name, matrix):
    rng = np.random.default_rng(qubit)
    amplitudes = rng.standard_normal((8, 2))
    got = _complex(apply_gate(_state(amplitudes), name, qubit))
    want = _reference_gate(amplitudes[:, 0] + 1j * amplitudes[:, 1], matrix, qubit)
    np.testing.assert_allclose(got, want, atol=1e-14)


def test_an_arbitrary_complex_gate_matches():
    rng = np.random.default_rng(7)
    entries = rng.standard_normal(8)
    matrix = np.array(
        [
            [entries[0] + 1j * entries[1], entries[2] + 1j * entries[3]],
            [entries[4] + 1j * entries[5], entries[6] + 1j * entries[7]],
        ]
    )
    amplitudes = rng.standard_normal((4, 2))
    got = _complex(apply_gate(_state(amplitudes), list(entries), 1))
    want = _reference_gate(amplitudes[:, 0] + 1j * amplitudes[:, 1], matrix, 1)
    np.testing.assert_allclose(got, want, atol=1e-14)


def test_the_phase_gate_multiplies_by_i():
    """A gate whose only content is imaginary, so a dropped term shows."""
    s_gate = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    out = apply_gate(_state([[0.0, 0.0], [1.0, 0.0]]), s_gate, 0)
    np.testing.assert_allclose(np.asarray(out), [[0.0, 0.0], [0.0, 1.0]], atol=1e-15)


def test_a_high_qubit_pairs_distant_amplitudes():
    """The striding: qubit 2 of three pairs index 0 with index 4."""
    amplitudes = np.zeros((8, 2))
    amplitudes[0, 0] = 1.0
    out = np.asarray(apply_gate(_state(amplitudes), "x", 2))
    assert abs(out[4, 0] - 1.0) < 1e-15
    assert abs(out[0, 0]) < 1e-15


def test_a_unitary_gate_preserves_the_norm():
    rng = np.random.default_rng(11)
    state = _state(rng.standard_normal((16, 2)))
    before = float(mt.sum(probabilities(state)).item())
    after = float(mt.sum(probabilities(apply_gate(state, "h", 3))).item())
    assert abs(before - after) < 1e-12


def test_a_batch_of_states_evolves_independently():
    rng = np.random.default_rng(13)
    amplitudes = rng.standard_normal((5, 4, 2))
    out = _complex(apply_gate(_state(amplitudes), "h", 1))
    for index in range(5):
        want = _reference_gate(
            amplitudes[index, :, 0] + 1j * amplitudes[index, :, 1], _HADAMARD, 1
        )
        np.testing.assert_allclose(out[index], want, atol=1e-14)


# --- measurement --------------------------------------------------------------


def test_probabilities_follow_the_born_rule():
    rng = np.random.default_rng(17)
    amplitudes = rng.standard_normal((8, 2))
    got = np.asarray(probabilities(_state(amplitudes)))
    want = np.abs(amplitudes[:, 0] + 1j * amplitudes[:, 1]) ** 2
    np.testing.assert_allclose(got, want, atol=1e-14)


def test_expect_z_is_the_probability_difference():
    rng = np.random.default_rng(19)
    amplitudes = rng.standard_normal((4, 2))
    probs = np.abs(amplitudes[:, 0] + 1j * amplitudes[:, 1]) ** 2
    # Qubit 0 is the low bit: 0 and 2 have it clear, 1 and 3 set.
    want = probs[0] + probs[2] - probs[1] - probs[3]
    assert abs(float(expect_z(_state(amplitudes), 0).item()) - want) < 1e-13


def test_expect_z_is_plus_one_on_the_zero_state():
    assert (
        abs(float(expect_z(_state([[1.0, 0.0], [0.0, 0.0]]), 0).item()) - 1.0) < 1e-15
    )


def test_expect_z_is_zero_on_a_superposition():
    plus = apply_gate(_state([[1.0, 0.0], [0.0, 0.0]]), "h", 0)
    assert abs(float(expect_z(plus, 0).item())) < 1e-15


def test_prefix_trace_marginalises_the_low_qubits():
    rng = np.random.default_rng(23)
    amplitudes = rng.standard_normal((8, 2))
    probs = np.abs(amplitudes[:, 0] + 1j * amplitudes[:, 1]) ** 2
    got = np.asarray(prefix_trace(_state(amplitudes), 2))
    np.testing.assert_allclose(got, probs.reshape(4, 2).sum(-1), atol=1e-14)


def test_prefix_trace_of_everything_is_the_probabilities():
    rng = np.random.default_rng(29)
    state = _state(rng.standard_normal((4, 2)))
    np.testing.assert_allclose(
        np.asarray(prefix_trace(state, 2)), np.asarray(probabilities(state)), atol=1e-15
    )


def test_prefix_trace_of_nothing_is_the_total_probability():
    rng = np.random.default_rng(31)
    state = _state(rng.standard_normal((8, 2)))
    total = float(mt.sum(probabilities(state)).item())
    np.testing.assert_allclose(np.asarray(prefix_trace(state, 0)), [total], atol=1e-13)


# --- the gradient -------------------------------------------------------------


def _numeric_jacobian_vector(function, base, cotangent, step=1e-6):
    """`d (out . cotangent) / d base`, one central difference per element."""
    values = np.array(base, dtype=np.float64)
    out = np.empty_like(values)
    flat = values.reshape(-1)
    for index in range(flat.size):
        original = flat[index]
        flat[index] = original + step
        high = float((function(values) * cotangent).sum())
        flat[index] = original - step
        low = float((function(values) * cotangent).sum())
        flat[index] = original
        out.reshape(-1)[index] = (high - low) / (2.0 * step)
    return out


def _analytic(function, base, cotangent):
    state = _state(base, requires_grad=True)
    output = function(state)
    weights = mt.Tensor(
        np.ascontiguousarray(np.asarray(cotangent, dtype=np.float64)), dtype="float64"
    )
    (output * weights).sum().backward()
    grad = np.asarray(state.grad).copy()
    mt.clear_autograd_graph()
    return grad


def test_the_gradient_through_a_gate_matches_finite_differences():
    rng = np.random.default_rng(37)
    base = rng.standard_normal((4, 2))
    cotangent = rng.standard_normal(4)

    def forward(values):
        return np.asarray(probabilities(apply_gate(_state(values), "h", 1)))

    np.testing.assert_allclose(
        _analytic(lambda s: probabilities(apply_gate(s, "h", 1)), base, cotangent),
        _numeric_jacobian_vector(forward, base, cotangent),
        atol=1e-7,
    )


def test_the_gradient_through_a_two_gate_circuit_matches():
    rng = np.random.default_rng(41)
    base = rng.standard_normal((4, 2))
    cotangent = np.ones(())

    def circuit(state):
        return expect_z(apply_gate(apply_gate(state, "h", 0), "x", 1), 0)

    def forward(values):
        return np.asarray(circuit(_state(values)))

    np.testing.assert_allclose(
        _analytic(circuit, base, cotangent),
        _numeric_jacobian_vector(forward, base, cotangent),
        atol=1e-7,
    )


def test_the_gradient_through_a_prefix_trace_matches():
    rng = np.random.default_rng(43)
    base = rng.standard_normal((8, 2))
    cotangent = rng.standard_normal(2)

    def forward(values):
        return np.asarray(prefix_trace(_state(values), 1))

    np.testing.assert_allclose(
        _analytic(lambda s: prefix_trace(s, 1), base, cotangent),
        _numeric_jacobian_vector(forward, base, cotangent),
        atol=1e-7,
    )


def test_the_backward_is_the_adjoint_and_not_the_gate():
    """A non-self-adjoint gate, where using `U` instead of `U†` would show.

    Hadamard and Pauli-X are their own adjoints, so a backward that applied the
    gate itself would pass every test above. The phase gate is not.
    """
    s_gate = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    rng = np.random.default_rng(47)
    base = rng.standard_normal((2, 2))
    cotangent = rng.standard_normal(2)

    def forward(values):
        return np.asarray(probabilities(apply_gate(_state(values), s_gate, 0)))

    np.testing.assert_allclose(
        _analytic(lambda s: probabilities(apply_gate(s, s_gate, 0)), base, cotangent),
        _numeric_jacobian_vector(forward, base, cotangent),
        atol=1e-7,
    )


# --- what is refused ----------------------------------------------------------


def test_a_qubit_out_of_range_is_rejected():
    with pytest.raises(Exception):
        apply_gate(_state([[1.0, 0.0], [0.0, 0.0]]), "h", 3)


def test_an_unknown_gate_name_is_rejected():
    with pytest.raises(ValueError, match="unknown gate"):
        apply_gate(_state([[1.0, 0.0], [0.0, 0.0]]), "flurb", 0)


def test_a_gate_of_the_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="eight reals"):
        apply_gate(_state([[1.0, 0.0], [0.0, 0.0]]), [1.0, 2.0], 0)


def test_a_state_that_is_not_a_power_of_two_is_rejected():
    with pytest.raises(Exception):
        apply_gate(_state([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), "h", 0)


def test_a_state_without_the_component_axis_is_rejected():
    with pytest.raises(Exception):
        probabilities(mt.Tensor([1.0, 0.0, 0.0, 0.0], dtype="float64"))
