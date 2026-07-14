# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

import minitensor as mt
from minitensor.tensor import Tensor


def test_eq_broadcasting():
    a = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = mt.Tensor([[1.0, 4.0]])
    result = a.eq(b)
    expected = np.array([[True, False], [False, True]])
    np.testing.assert_array_equal(result.numpy(), expected)


def test_lt_bool_error():
    a = mt.Tensor([True, False], dtype="bool")
    b = mt.Tensor([False, True], dtype="bool")
    with pytest.raises(ValueError):
        a.lt(b)


def test_gt_incompatible_shapes_error():
    a = mt.Tensor([1.0, 2.0, 3.0])
    b = mt.Tensor([1.0, 2.0])
    with pytest.raises(ValueError):
        a.gt(b)


def test_eq_promotes_mixed_dtypes():
    a = mt.Tensor([1.0, 2.0], dtype="float32")
    b = mt.Tensor([1, 3], dtype="int32")
    result = a.eq(b)
    expected = np.array([True, False])
    np.testing.assert_array_equal(result.numpy(), expected)


def test_bool_numeric_comparisons():
    bools = mt.Tensor([True, False], dtype="bool")
    ints = mt.Tensor([1, 0], dtype="int32")
    floats = mt.Tensor([1.0, 0.5], dtype="float32")

    eq_res = bools.eq(ints)
    np.testing.assert_array_equal(eq_res.numpy(), np.array([True, True]))

    lt_res = bools.lt(floats)
    np.testing.assert_array_equal(lt_res.numpy(), np.array([False, True]))


def test_comparison_invalid_operand_type():
    a = mt.Tensor([1.0, 2.0])
    with pytest.raises(TypeError):
        a.eq("foo")


def test_nan_and_inf_comparisons():
    a = mt.Tensor([float("nan"), float("inf")])
    b = mt.Tensor([0.0, 1.0])
    eq_res = a.eq(a)
    lt_res = a.lt(b)
    gt_res = a.gt(b)
    assert not eq_res.numpy()[0]
    assert not lt_res.numpy()[0]
    assert gt_res.numpy()[1] and not lt_res.numpy()[1]


def test_eq_with_scalar():
    a = mt.Tensor([1.0, 2.0, 3.0])
    res = (a == 2.0).numpy()
    expected = np.array([False, True, False])
    np.testing.assert_array_equal(res, expected)


def test_functional_minimum_maximum_broadcast_and_scalars():
    a = mt.Tensor([[1.0, -2.0, 3.0], [4.0, 0.5, -6.0]])
    b = mt.Tensor([[0.0, 2.0, 2.5]])

    max_result = mt.maximum(a, b)
    min_result = mt.functional.minimum(a, -1.0)
    int_result = mt.maximum(mt.Tensor([1, 4], dtype="int32"), 3)

    np.testing.assert_allclose(max_result.numpy(), np.maximum(a.numpy(), b.numpy()))
    np.testing.assert_allclose(min_result.numpy(), np.minimum(a.numpy(), -1.0))
    np.testing.assert_array_equal(
        int_result.numpy(), np.maximum(np.array([1, 4], dtype=np.int32), 3)
    )


def test_functional_minimum_maximum_bool_and_nan_edges():
    bool_a = mt.Tensor([True, False, True], dtype="bool")
    bool_b = mt.Tensor([False, False, True], dtype="bool")

    np.testing.assert_array_equal(
        mt.maximum(bool_a, bool_b).numpy(),
        np.maximum(bool_a.numpy(), bool_b.numpy()),
    )
    np.testing.assert_array_equal(
        mt.functional.minimum(bool_a, bool_b).numpy(),
        np.minimum(bool_a.numpy(), bool_b.numpy()),
    )

    left_nan = mt.Tensor([float("nan"), 1.0])
    right_nan = mt.Tensor([0.0, float("nan")])

    max_result = mt.maximum(left_nan, right_nan).numpy()
    min_result = mt.functional.minimum(left_nan, right_nan).numpy()

    assert np.isnan(max_result[0])
    assert np.isnan(max_result[1])
    assert np.isnan(min_result[0])
    assert np.isnan(min_result[1])


def test_functional_minimum_maximum_reject_incompatible_shapes():
    a = mt.Tensor([1.0, 2.0, 3.0])
    b = mt.Tensor([1.0, 2.0])

    with pytest.raises(ValueError):
        mt.maximum(a, b)

    with pytest.raises(ValueError):
        mt.functional.minimum(a, b)


def test_lt_with_list():
    a = mt.Tensor([1, 2, 3])
    res = a < [2, 2, 4]
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(res.numpy(), expected)


def test_gt_with_numpy_array():
    a = mt.Tensor([1.0, 2.0, 3.0])
    b = np.array([1.5, 1.5, 4.0])
    res = a.gt(b)
    expected = np.array([False, True, False])
    np.testing.assert_array_equal(res.numpy(), expected)


def test_negation_uses_backend():
    a = mt.Tensor([1.0, -2.0, 3.0])
    res = (-a).numpy()
    expected = np.array([-1.0, 2.0, -3.0])
    np.testing.assert_array_equal(res, expected)


def test_allclose_top_level_accepts_tensor_like_inputs_and_tolerances():
    assert mt.allclose([1.0, 2.0], np.array([1.0, 2.0 + 1e-6], dtype=np.float32))
    assert not mt.allclose([1.0, 2.0], [1.0, 2.1], rtol=1e-6, atol=1e-6)


def test_allclose_equal_nan_matches_numpy_style_option():
    left = mt.tensor([1.0, float("nan"), float("inf")])
    right = mt.tensor([1.0, float("nan"), float("inf")])

    assert not left.allclose(right)
    assert left.allclose(right, equal_nan=True)
    assert mt.allclose(left, right, equal_nan=True)


def test_allclose_handles_signed_zero_and_infinities():
    assert mt.allclose(
        [0.0, float("inf"), -float("inf")], [-0.0, float("inf"), -float("inf")]
    )
    assert not mt.allclose([float("inf")], [-float("inf")])


def test_equality_helpers_promote_compatible_numeric_dtypes():
    ints = mt.tensor([1, 2, 3], dtype="int64")
    floats = mt.tensor([1.0, 2.0, 3.0], dtype="float32")

    assert ints.allclose(floats)
    assert mt.allclose(ints, floats)
    assert mt.array_equal(ints, floats)


def test_allclose_rejects_invalid_tolerances():
    tensor = mt.tensor([1.0])

    with pytest.raises(ValueError, match="rtol and atol"):
        tensor.allclose(tensor, rtol=-1.0)

    with pytest.raises(ValueError, match="rtol and atol"):
        mt.allclose(tensor, tensor, atol=float("nan"))


def test_array_equal_top_level_accepts_tensor_like_inputs():
    assert mt.array_equal([1, 2, 3], np.array([1, 2, 3], dtype=np.int64))
    assert not mt.array_equal([1, 2, 3], [1, 2, 4])


def test_where_basic_selection():
    condition = mt.Tensor([[True, False], [False, True]], dtype="bool")
    input_tensor = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    other_tensor = mt.Tensor([[10.0, 20.0], [30.0, 40.0]])

    result = input_tensor.where(condition, other_tensor)
    expected = np.array([[1.0, 20.0], [30.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(result.numpy(), expected)


def test_where_broadcasting():
    condition = mt.Tensor([[True], [False]], dtype="bool")
    input_tensor = mt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    other_tensor = mt.Tensor([10.0, 20.0, 30.0])

    result = input_tensor.where(condition, other_tensor)
    expected = np.where(
        np.array([[True], [False]]),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]]),
    )
    np.testing.assert_allclose(result.numpy(), expected)


def test_where_requires_bool_condition():
    condition = mt.Tensor([0, 1])
    input_tensor = mt.Tensor([1.0, 2.0])
    other_tensor = mt.Tensor([3.0, 4.0])

    with pytest.raises(TypeError):
        input_tensor.where(condition, other_tensor)


def test_where_promotes_to_common_dtype():
    condition = mt.Tensor([True, False, True], dtype="bool")
    int_input = mt.Tensor([1, 2, 3], dtype="int64")
    float_other = mt.Tensor([0.5, 1.5, 2.5], dtype="float32")

    method_result = int_input.where(condition, float_other)
    assert method_result.dtype == "float32"
    np.testing.assert_allclose(
        method_result.numpy(), np.array([1.0, 1.5, 3.0], dtype=np.float32)
    )

    functional_result = mt.functional.where(condition, int_input, [0.5, 1.5, 2.5])
    assert functional_result.dtype == "float32"
    np.testing.assert_allclose(
        functional_result.numpy(), np.array([1.0, 1.5, 3.0], dtype=np.float32)
    )


def test_where_autograd_masks_gradients():
    condition = mt.Tensor([[True, False], [False, True]], dtype="bool")
    input_tensor = mt.Tensor(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True, dtype="float32"
    )
    other_tensor = mt.Tensor(
        [[10.0, 20.0], [30.0, 40.0]], requires_grad=True, dtype="float32"
    )

    result = input_tensor.where(condition, other_tensor)
    loss = result.sum()
    loss.backward()

    np.testing.assert_allclose(
        input_tensor.grad.numpy(),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        other_tensor.grad.numpy(),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )


def test_where_functional_and_top_level_match():
    condition = [[True, False], [False, True]]
    input_data = [[1.0, 2.0], [3.0, 4.0]]
    other_data = [[10.0, 20.0], [30.0, 40.0]]

    via_method = mt.Tensor(input_data).where(
        mt.Tensor(condition, dtype="bool"), mt.Tensor(other_data)
    )
    via_functional = mt.functional.where(condition, input_data, other_data)
    via_top_level = mt.where(condition, input_data, other_data)
    core_where = mt.numpy_compat.where(
        mt.Tensor(condition, dtype="bool")._tensor,
        mt.Tensor(input_data)._tensor,
        mt.Tensor(other_data)._tensor,
    )
    via_numpy_compat = mt.Tensor.__new__(mt.Tensor)
    via_numpy_compat._tensor = core_where

    expected = np.array([[1.0, 20.0], [30.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(via_method.numpy(), expected)
    np.testing.assert_allclose(via_functional.numpy(), expected)
    np.testing.assert_allclose(via_top_level.numpy(), expected)
    np.testing.assert_allclose(via_numpy_compat.numpy(), expected)


def test_scalar_broadcasting_addition():
    a = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = mt.Tensor(1.0)
    c = a + b
    expected = np.array([[2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_allclose(c.numpy(), expected)


def test_scalar_broadcasting_subtraction():
    a = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = mt.Tensor(1.0)
    c = a - b
    expected = np.array([[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(c.numpy(), expected)


def test_scalar_broadcasting_division():
    a = mt.Tensor([[2.0, 4.0], [6.0, 8.0]])
    b = mt.Tensor(2.0)
    c = a / b
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(c.numpy(), expected)


def test_broadcast_incompatible_shapes_error():
    a = mt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = mt.Tensor([1.0, 2.0])
    with pytest.raises(ValueError):
        _ = a + b


def test_clone_creates_independent_tensor():
    a = mt.Tensor([1.0, 2.0, 3.0])
    b = a.clone()
    b = b + mt.Tensor([1.0, 1.0, 1.0])
    np.testing.assert_allclose(a.numpy(), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(b.numpy(), np.array([2.0, 3.0, 4.0]))


def test_multi_dimensional_broadcasting():
    a = mt.Tensor(np.ones((2, 1, 3), dtype=np.float32))
    b = mt.Tensor(np.ones((1, 4, 1), dtype=np.float32) * 2)
    c = a + b
    expected = np.ones((2, 4, 3), dtype=np.float32) * 3
    np.testing.assert_allclose(c.numpy(), expected)


def test_high_dim_broadcast_mismatch_error():
    a = mt.Tensor(np.ones((2, 3, 1), dtype=np.float32))
    b = mt.Tensor(np.ones((4, 3, 1), dtype=np.float32))
    with pytest.raises(ValueError):
        _ = a + b


def test_broadcast_zero_dimension():
    a = mt.Tensor(np.empty((0, 3), dtype=np.float32))
    b = mt.Tensor(np.ones((3,), dtype=np.float32))
    result = a + b
    assert result.shape == (0, 3)


def test_conv_bias_broadcasting():
    inp = mt.Tensor(np.ones((2, 3, 4, 4), dtype=np.float32))
    bias = mt.Tensor(np.arange(3, dtype=np.float32).reshape(3, 1, 1))
    out = inp + bias
    expected = np.ones((2, 3, 4, 4), dtype=np.float32) + np.arange(
        3, dtype=np.float32
    ).reshape(1, 3, 1, 1)
    np.testing.assert_allclose(out.numpy(), expected)


def test_conv_bias_broadcasting_broadcasting():
    inp = mt.Tensor(np.ones((2, 3, 4, 4), dtype=np.float32))
    bias = mt.Tensor(np.arange(3, dtype=np.float32).reshape(3, 1, 1))
    out = inp + bias
    expected = np.ones((2, 3, 4, 4), dtype=np.float32) + np.arange(
        3, dtype=np.float32
    ).reshape(1, 3, 1, 1)
    np.testing.assert_allclose(out.numpy(), expected)


def test_broadcast_backward_add():
    dev = mt.device("cpu")
    a = mt.Tensor(
        np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True, device=dev
    )
    b = mt.Tensor(np.arange(3, dtype=np.float32), requires_grad=True, device=dev)
    c = a + b
    grad = mt.Tensor(1.0, device=dev)
    c.sum().backward(grad)
    np.testing.assert_allclose(a.grad.numpy(), np.ones((2, 3), dtype=np.float32))
    np.testing.assert_allclose(
        b.grad.numpy(), np.array([2.0, 2.0, 2.0], dtype=np.float32)
    )


def test_broadcast_backward_mul():
    dev = mt.device("cpu")
    a = mt.Tensor(
        np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True, device=dev
    )
    b = mt.Tensor(np.arange(3, dtype=np.float32) + 1, requires_grad=True, device=dev)
    c = a * b
    grad = mt.Tensor(1.0, device=dev)
    c.sum().backward(grad)
    np.testing.assert_allclose(
        a.grad.numpy(), np.tile(np.arange(1, 4, dtype=np.float32), (2, 1))
    )
    np.testing.assert_allclose(
        b.grad.numpy(), np.array([3.0, 5.0, 7.0], dtype=np.float32)
    )


class IndexLike:
    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


def test_broadcast_shapes_matches_numpy_style_rules() -> None:
    assert mt.broadcast_shapes() == ()
    assert mt.broadcast_shapes((), (2, 3)) == (2, 3)
    assert mt.broadcast_shapes((5, 1, 4), (1, 3, 1), (3, 4)) == (5, 3, 4)
    assert mt.broadcast_shapes(3, (2, 1)) == (2, 3)
    assert mt.broadcast_shapes(np.int64(3), (2, 1)) == (2, 3)
    assert mt.broadcast_shapes((IndexLike(1), IndexLike(4)), (3, 1)) == (3, 4)


def test_broadcast_shapes_matches_numpy_edge_cases() -> None:
    cases = [
        ((),),
        ((0,), (1,)),
        ((1, 0), (3, 0)),
        ((1, 1, 0), (2, 3, 0), (3, 0)),
        ((1,) * 8, (2, 1, 3, 1, 4, 1, 5, 1)),
    ]

    for shapes in cases:
        assert mt.broadcast_shapes(*shapes) == np.broadcast_shapes(*shapes)


def test_broadcast_shapes_accepts_tensor_shape_objects() -> None:
    tensor = mt.zeros(2, 1, 4)
    assert mt.broadcast_shapes(tensor.shape, (3, 4)) == (2, 3, 4)


def test_broadcast_shapes_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="cannot be broadcast"):
        mt.broadcast_shapes((2, 3), (4, 3))


def test_broadcast_shapes_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mt.broadcast_shapes((2, -1))
    with pytest.raises(ValueError, match="non-negative"):
        mt.broadcast_shapes(IndexLike(-1))
    with pytest.raises(TypeError, match="integers"):
        mt.broadcast_shapes((2, 1.5))
    with pytest.raises(TypeError, match="not bool"):
        mt.broadcast_shapes(True)


def test_can_broadcast_returns_boolean_without_raising() -> None:
    assert mt.can_broadcast((1, 3), (2, 3)) is True
    assert mt.can_broadcast((2, 3), (4, 3)) is False
    assert mt.can_broadcast((2, "bad")) is False


def test_broadcast_to_matches_numpy_and_reuses_existing_tensor() -> None:
    tensor = mt.Tensor([[1.0], [2.0]])

    result = mt.broadcast_to(tensor, (2, 3))

    assert result.shape == (2, 3)
    np.testing.assert_array_equal(
        result.numpy(), np.broadcast_to(tensor.numpy(), (2, 3))
    )
    assert mt.broadcast_to(result, (2, 3)) is result


def test_broadcast_to_accepts_tensor_like_inputs_and_scalar_shape() -> None:
    result = mt.broadcast_to([7.0], 4)

    assert result.shape == (4,)
    np.testing.assert_array_equal(result.numpy(), np.full((4,), 7.0, dtype=np.float32))


def test_broadcast_to_supports_scalar_targets_and_leading_zero_axes() -> None:
    scalar = mt.broadcast_to(mt.Tensor(3.0), ())
    assert scalar.shape == ()
    np.testing.assert_array_equal(scalar.numpy(), np.array(3.0, dtype=np.float32))

    empty_row = mt.broadcast_to(mt.ones(0), (1, 0))
    assert empty_row.shape == (1, 0)
    np.testing.assert_array_equal(empty_row.numpy(), np.empty((1, 0), dtype=np.float32))


def test_broadcast_to_handles_zero_sized_target_metadata() -> None:
    tensor = mt.Tensor.ones((1,), dtype="float32", requires_grad=True)

    result = mt.broadcast_to(tensor, (0,))

    assert result.shape == (0,)
    assert result.dtype == tensor.dtype
    assert result.device == tensor.device
    assert result.requires_grad is True
    np.testing.assert_array_equal(result.numpy(), np.empty((0,), dtype=np.float32))


def test_broadcast_to_rejects_invalid_or_incompatible_targets() -> None:
    with pytest.raises(ValueError, match="cannot be broadcast"):
        mt.broadcast_to(mt.ones(2, 3), (3, 2))
    with pytest.raises(ValueError, match="cannot be broadcast"):
        mt.broadcast_to(mt.ones(1, 2), (2,))
    with pytest.raises(ValueError, match="non-negative"):
        mt.broadcast_to(mt.ones(1), (2, -1))
    with pytest.raises(TypeError, match="not bool"):
        mt.broadcast_to(mt.ones(1), True)


def test_broadcast_helpers_are_public_api_entries() -> None:
    top_level = mt.list_public_api()["top_level"]
    assert "broadcast_to" in top_level
    assert "broadcast_shapes" in top_level
    assert "broadcast_tensors" in top_level
    assert "can_broadcast" in top_level


def test_matmul_basic():
    a = mt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = mt.Tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    result = a.matmul(b)
    np.testing.assert_allclose(result.numpy(), np.array([[58.0, 64.0], [139.0, 154.0]]))


def test_solve_vector_matches_numpy():
    a = mt.Tensor([[3.0, 1.0], [1.0, 2.0]])
    b = mt.Tensor([9.0, 8.0])
    result = a.solve(b)
    expected = np.linalg.solve(a.numpy(), b.numpy())
    np.testing.assert_allclose(result.numpy(), expected)


def test_functional_solve_matches_tensor():
    a = mt.Tensor([[4.0, 2.0], [1.0, 3.0]])
    b = mt.Tensor([10.0, 7.0])
    tensor_result = a.solve(b)
    functional_result = mt.solve(a, b)
    np.testing.assert_allclose(functional_result.numpy(), tensor_result.numpy())


def test_solve_multiple_rhs():
    a = mt.Tensor(np.array([[2.0, 1.0], [5.0, 7.0]], dtype=np.float32))
    b = mt.Tensor(np.array([[11.0, 5.0], [13.0, 6.0]], dtype=np.float32))
    result = a.solve(b)
    expected = np.linalg.solve(a.numpy(), b.numpy())
    np.testing.assert_allclose(result.numpy(), expected)


def test_solve_batched_systems():
    a = mt.Tensor(
        np.array(
            [
                [[3.0, 1.0], [1.0, 2.0]],
                [[4.0, 2.0], [2.0, 5.0]],
            ],
            dtype=np.float32,
        )
    )
    b = mt.Tensor(
        np.array(
            [
                [9.0, 8.0],
                [13.0, 17.0],
            ],
            dtype=np.float32,
        )
    )
    result = a.solve(b)
    a_np = a.numpy()
    b_np = b.numpy()
    expected = np.stack(
        [np.linalg.solve(a_np[i], b_np[i]) for i in range(a_np.shape[0])]
    )
    np.testing.assert_allclose(result.numpy(), expected)


def test_solve_handles_empty_rhs_columns():
    a = mt.Tensor(np.eye(3, dtype=np.float32))
    b = mt.Tensor(np.empty((3, 0), dtype=np.float32))

    result = a.solve(b)

    assert result.shape == (3, 0)
    assert result.numpy().size == 0


def test_solve_backward_gradients():
    a = mt.Tensor([[3.0, 1.0], [1.0, 2.0]], requires_grad=True)
    b = mt.Tensor([9.0, 8.0], requires_grad=True)
    solution = a.solve(b)
    solution.sum().backward()

    a_np = a.numpy()
    x_np = solution.detach().numpy()
    grad_out = np.ones_like(x_np)
    expected_grad_b = np.linalg.solve(a_np.T, grad_out)
    grad_matrix = np.outer(grad_out, x_np)
    expected_grad_a = -np.linalg.solve(a_np.T, grad_matrix)

    np.testing.assert_allclose(b.grad.numpy(), expected_grad_b, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(a.grad.numpy(), expected_grad_a, rtol=1e-5, atol=1e-6)


def test_solve_requires_square_matrix():
    a = mt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = mt.Tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        a.solve(b)


def test_matmul_shape_mismatch():
    a = mt.Tensor([[1.0, 2.0]])
    b = mt.Tensor([[3.0, 4.0, 5.0]])
    with pytest.raises(ValueError):
        a.matmul(b)


def test_matmul_dtype_mismatch():
    a = mt.Tensor([[1.0, 2.0]], dtype="float32")
    b = mt.Tensor([[3.0], [4.0]], dtype="float64")
    with pytest.raises(TypeError):
        a.matmul(b)


def test_matmul_bool_error():
    a = mt.Tensor([[True, False], [False, True]], dtype="bool")
    b = mt.Tensor([[True, True], [False, False]], dtype="bool")
    with pytest.raises(ValueError):
        a.matmul(b)


def test_matmul_vector_operands():
    # For 1-D operands, vec@vec -> scalar and mat@vec/vec@mat -> vec.
    a = mt.Tensor([1.0, 2.0])
    b = mt.Tensor([3.0, 4.0])
    np.testing.assert_allclose(a.matmul(b).numpy(), np.dot([1.0, 2.0], [3.0, 4.0]))
    assert a.matmul(b).numpy().shape == ()

    m = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(
        m.matmul(a).numpy(), np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2) @ [1.0, 2.0]
    )
    np.testing.assert_allclose(
        (a @ m).numpy(), np.array([1.0, 2.0]) @ np.array([[1.0, 2.0], [3.0, 4.0]])
    )

    # Scalars are still not valid matmul operands.
    with pytest.raises((ValueError, RuntimeError)):
        mt.Tensor(2.0).matmul(mt.Tensor(3.0))


def test_matmul_batch_dimensions():
    a = mt.Tensor(np.arange(12, dtype=np.float32).reshape(2, 2, 3))
    b = mt.Tensor(np.arange(12, dtype=np.float32).reshape(2, 3, 2))
    result = a.matmul(b)
    expected = np.matmul(
        np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        np.arange(12, dtype=np.float32).reshape(2, 3, 2),
    )
    np.testing.assert_allclose(result.numpy(), expected)


def test_bmm_matches_numpy():
    a_data = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    b_data = np.linspace(0.5, 6.0, 12, dtype=np.float32).reshape(2, 3, 2)
    a = mt.Tensor(a_data)
    b = mt.Tensor(b_data)
    result = a.bmm(b)
    expected = np.matmul(a_data, b_data)
    np.testing.assert_allclose(result.numpy(), expected)


def test_top_level_bmm_matches_method():
    a = mt.Tensor(np.arange(12, dtype=np.float32).reshape(2, 2, 3))
    b = mt.Tensor(np.arange(12, dtype=np.float32).reshape(2, 3, 2))
    method_result = a.bmm(b)
    functional_result = mt.bmm(a, b)
    np.testing.assert_allclose(functional_result.numpy(), method_result.numpy())


def test_bmm_requires_three_dimensions():
    a = mt.Tensor(np.ones((2, 3), dtype=np.float32))
    b = mt.Tensor(np.ones((2, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        a.bmm(b)


def test_bmm_batch_mismatch_error():
    a = mt.Tensor(np.arange(12, dtype=np.float32).reshape(2, 2, 3))
    b = mt.Tensor(np.arange(18, dtype=np.float32).reshape(3, 3, 2))
    with pytest.raises(ValueError):
        a.bmm(b)


def test_matmul_batch_mismatch_error():
    a = mt.Tensor(np.arange(12, dtype=np.float32).reshape(2, 2, 3))
    b = mt.Tensor(np.arange(18, dtype=np.float32).reshape(3, 3, 2))
    with pytest.raises(ValueError):
        a.matmul(b)


def test_matmul_zero_dimension():
    a = mt.Tensor(np.empty((2, 0), dtype=np.float32))
    b = mt.Tensor(np.empty((0, 3), dtype=np.float32))
    result = a.matmul(b)
    np.testing.assert_allclose(result.numpy(), np.zeros((2, 3), dtype=np.float32))


def test_matmul_inf_nan_propagation():
    a = mt.Tensor([[np.inf, np.nan], [1.0, 2.0]], dtype="float32")
    b = mt.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    result = a.matmul(b).numpy()
    assert np.isnan(result[0]).all()


def test_matmul_backward_batched():
    dev = mt.device("cpu")
    a = mt.Tensor(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4), requires_grad=True, device=dev
    )
    b = mt.Tensor(
        np.arange(32, dtype=np.float32).reshape(2, 4, 4), requires_grad=True, device=dev
    )
    c = a.matmul(b)
    c.sum().backward()
    expected_a = np.matmul(np.ones_like(c.numpy()), np.swapaxes(b.numpy(), -1, -2))
    expected_b = np.matmul(np.swapaxes(a.numpy(), -1, -2), np.ones_like(c.numpy()))
    np.testing.assert_allclose(a.grad.numpy(), expected_a)
    np.testing.assert_allclose(b.grad.numpy(), expected_b)


def test_matmul_backward_requires_grad_flags():
    dev = mt.device("cpu")
    a = mt.Tensor(
        np.arange(4, dtype=np.float32).reshape(2, 2), requires_grad=True, device=dev
    )
    b = mt.Tensor(
        np.arange(4, dtype=np.float32).reshape(2, 2), requires_grad=False, device=dev
    )
    c = a.matmul(b)
    c.sum().backward()
    assert a.grad is not None
    assert b.grad is None


def test_transpose_backward_permutation():
    dev = mt.device("cpu")
    x = mt.Tensor(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4), requires_grad=True, device=dev
    )
    y = x.permute(1, 2, 0)
    grad = mt.Tensor(np.ones((3, 4, 2), dtype=np.float32), device=dev)
    y.backward(grad)
    expected = np.transpose(grad.numpy(), (2, 0, 1))
    np.testing.assert_allclose(x.grad.numpy(), expected)


def test_triu_matches_numpy():
    data = np.arange(9, dtype=np.float32).reshape(3, 3)
    tensor = mt.Tensor(data)
    result = tensor.triu()
    np.testing.assert_allclose(result.numpy(), np.triu(data))


def test_triu_diagonal_offset():
    data = np.arange(9, dtype=np.float32).reshape(3, 3)
    tensor = mt.Tensor(data)
    result = tensor.triu(diagonal=1)
    np.testing.assert_allclose(result.numpy(), np.triu(data, k=1))


def test_triu_gradient_mask():
    data = np.arange(9, dtype=np.float32).reshape(3, 3)
    tensor = mt.Tensor(data, requires_grad=True)
    tensor.triu(diagonal=-1).sum().backward()
    expected_grad = np.triu(np.ones_like(data), k=-1)
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad)


def test_triu_requires_minimum_rank():
    tensor = mt.Tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        tensor.triu()


def test_tril_matches_numpy_batched():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    tensor = mt.Tensor(data)
    result = tensor.tril(diagonal=-1)
    np.testing.assert_allclose(result.numpy(), np.tril(data, k=-1))


def test_tril_gradient_mask():
    data = np.arange(9, dtype=np.float32).reshape(3, 3)
    tensor = mt.Tensor(data, requires_grad=True)
    tensor.tril(diagonal=0).sum().backward()
    expected_grad = np.tril(np.ones_like(data), k=0)
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad)


def test_tril_requires_minimum_rank():
    tensor = mt.Tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        tensor.tril()


def test_dot_matches_numpy_float():
    a = mt.Tensor([1.0, 2.0, 3.0], dtype="float32")
    b = mt.Tensor([4.0, 5.0, 6.0], dtype="float32")
    result = a.dot(b)
    assert result.dtype == "float32"
    np.testing.assert_allclose(result.numpy(), np.dot(a.numpy(), b.numpy()))


def test_top_level_dot_matches_tensor_method():
    a = mt.Tensor([1.0, 2.0, 3.0], dtype="float32")
    b = mt.Tensor([4.0, 5.0, 6.0], dtype="float32")
    tensor_result = a.dot(b)
    functional_result = mt.dot(a, b)
    assert tensor_result.dtype == "float32"
    assert functional_result.dtype == "float32"
    np.testing.assert_allclose(functional_result.numpy(), tensor_result.numpy())


def test_dot_matches_numpy_int():
    a = mt.Tensor([2, 3, 4], dtype="int64")
    b = mt.Tensor([5, 6, 7], dtype="int64")
    result = a.dot(b)
    assert result.dtype == "int64"
    assert result.item() == int(np.dot(a.numpy(), b.numpy()))


def test_dot_requires_1d_inputs():
    a = mt.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = mt.Tensor([1.0, 2.0])
    with pytest.raises(ValueError):
        a.dot(b)


def test_dot_mismatched_lengths_raise():
    a = mt.Tensor([1.0, 2.0, 3.0])
    b = mt.Tensor([4.0, 5.0])
    with pytest.raises(ValueError):
        a.dot(b)


def test_dot_backward_gradients():
    a = mt.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = mt.Tensor([4.0, 5.0, 6.0], requires_grad=True)
    result = a.dot(b)
    result.backward()
    np.testing.assert_allclose(a.grad.numpy(), b.numpy())
    np.testing.assert_allclose(b.grad.numpy(), a.numpy())


def test_dot_bool_not_supported():
    a = mt.Tensor([True, False, True], dtype="bool")
    b = mt.Tensor([True, True, False], dtype="bool")
    with pytest.raises(ValueError):
        a.dot(b)


def test_cross_product_matches_numpy():
    a = mt.Tensor([1.0, 2.0, 3.0])
    b = mt.Tensor([4.0, 5.0, 6.0])
    c = mt.cross(a, b)
    expected = np.cross(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    np.testing.assert_allclose(c.numpy(), expected)


def test_cross_product_batch_axis():
    a = mt.Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b = mt.Tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    c = mt.cross(a, b, axis=-1)
    expected = np.cross(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        axis=-1,
    )
    np.testing.assert_allclose(c.numpy(), expected)


def test_cross_product_non_last_axis():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    b_np = np.array([[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]])
    a = mt.Tensor(a_np.tolist())
    b = mt.Tensor(b_np.tolist())
    c = mt.cross(a, b, axis=0)
    expected = np.cross(a_np, b_np, axis=0)
    np.testing.assert_allclose(c.numpy(), expected)


def test_cross_product_broadcasting():
    a_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b_np = np.array([0.0, 0.0, 1.0])
    a = mt.Tensor(a_np.tolist())
    b = mt.Tensor(b_np.tolist())
    c = mt.cross(a, b)
    expected = np.cross(a_np, b_np)
    np.testing.assert_allclose(c.numpy(), expected)


def test_cross_product_invalid_axis():
    a = mt.Tensor([1.0, 2.0, 3.0])
    b = mt.Tensor([4.0, 5.0, 6.0])
    with pytest.raises(ValueError):
        mt.cross(a, b, axis=1)


def test_cross_product_invalid_dimension():
    a = mt.Tensor([1.0, 2.0, 3.0, 4.0])
    b = mt.Tensor([5.0, 6.0, 7.0, 8.0])
    with pytest.raises(ValueError):
        mt.cross(a, b)


def test_cross_product_broadcast_mismatch():
    a_np = np.ones((2, 3))
    b_np = np.ones((3, 3))
    a = mt.Tensor(a_np.tolist())
    b = mt.Tensor(b_np.tolist())
    with pytest.raises(ValueError):
        mt.cross(a, b)


def test_cross_product_negative_axis_equivalent():
    a_np = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
        ]
    )
    b_np = np.array(
        [
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ]
    )
    a = mt.Tensor(a_np.tolist())
    b = mt.Tensor(b_np.tolist())
    c = mt.cross(a, b, axis=-2)
    expected = np.cross(a_np, b_np, axis=-2)
    np.testing.assert_allclose(c.numpy(), expected)


def test_cross_product_high_dimensional_broadcasting():
    a_np = np.arange(24.0).reshape(2, 1, 4, 3)
    b_np = np.array([1.0, 0.0, 0.0])
    a = mt.Tensor(a_np.tolist())
    b = mt.Tensor(b_np.tolist())
    c = mt.cross(a, b, axis=-1)
    expected = np.cross(a_np, b_np, axis=-1)
    np.testing.assert_allclose(c.numpy(), expected)


def test_cross_product_dtype_mismatch():
    a = mt.Tensor([1.0, 2.0, 3.0], dtype="float32")
    b = mt.Tensor([4.0, 5.0, 6.0], dtype="float64")
    with pytest.raises(ValueError):
        mt.cross(a, b)


def test_cross_product_negative_axis_out_of_range():
    a = mt.Tensor([[1.0, 0.0, 0.0]])
    b = mt.Tensor([[0.0, 1.0, 0.0]])
    with pytest.raises(ValueError):
        mt.cross(a, b, axis=-3)


def test_cross_product_anti_commutativity():
    a_np = np.array([1.0, 2.0, 3.0])
    b_np = np.array([0.5, -1.0, 2.0])
    a = mt.Tensor(a_np.tolist())
    b = mt.Tensor(b_np.tolist())
    ab = mt.cross(a, b)
    ba = mt.cross(b, a)
    np.testing.assert_allclose(ab.numpy(), -ba.numpy())


def test_tensor_cross_matches_numpy():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a.cross(b)
    expected = np.cross(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    np.testing.assert_allclose(c.numpy(), expected)


def test_tensor_cross_axis_parameter():
    a_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b_np = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    a = Tensor(a_np.tolist())
    b = Tensor(b_np.tolist())
    c = a.cross(b, axis=-1)
    expected = np.cross(a_np, b_np, axis=-1)
    np.testing.assert_allclose(c.numpy(), expected)


def test_tensor_cross_invalid_axis():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    with pytest.raises(ValueError):
        a.cross(b, axis=1)


def test_broadcast_tensors_expands_inputs_without_eager_materialization() -> None:
    row = mt.Tensor([[1.0, 2.0, 3.0]], dtype="float32", requires_grad=True)
    column = mt.Tensor([[10.0], [20.0]], dtype="float32")
    scalar = 5.0

    row_b, column_b, scalar_b = mt.broadcast_tensors(row, column, scalar)

    assert row_b.shape == (2, 3)
    assert column_b.shape == (2, 3)
    assert scalar_b.shape == (2, 3)
    assert row_b.requires_grad is True
    np.testing.assert_allclose(
        row_b.numpy(), np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        column_b.numpy(),
        np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(scalar_b.numpy(), np.full((2, 3), 5.0, dtype=np.float32))


def test_broadcast_tensors_validates_inputs_and_reuses_matching_tensors() -> None:
    tensor = mt.ones((2, 3))

    (same,) = mt.broadcast_tensors(tensor)
    assert same is tensor

    with pytest.raises(TypeError):
        mt.broadcast_tensors()

    with pytest.raises(ValueError):
        mt.broadcast_tensors(mt.ones((2, 3)), mt.ones((4, 3)))


def test_broadcast_tensors_handles_zero_sized_broadcast_edges() -> None:
    one = mt.ones((1,), dtype="float64", requires_grad=True)
    empty = mt.ones((0,), dtype="float32")

    one_b, empty_b = mt.broadcast_tensors(one, empty)

    assert one_b.shape == (0,)
    assert one_b.dtype == "float64"
    assert one_b.device == one.device
    assert one_b.requires_grad is True
    assert one_b.numpy().shape == (0,)
    assert empty_b is empty

    leading, matrix_empty = mt.broadcast_tensors(mt.ones((1, 3)), mt.ones((0, 3)))
    assert leading.shape == (0, 3)
    assert matrix_empty.shape == (0, 3)


def test_isclose_returns_elementwise_mask_with_broadcasting_and_special_values():
    left = mt.tensor([[1.0, 2.0, float("nan"), float("inf")]], dtype="float32")
    right = mt.tensor([1.0 + 1e-6, 3.0, float("nan"), float("inf")], dtype="float32")

    result = mt.isclose(left, right, rtol=1e-5, atol=1e-8, equal_nan=True)

    expected = np.isclose(
        left.numpy(), right.numpy(), rtol=1e-5, atol=1e-8, equal_nan=True
    )
    np.testing.assert_array_equal(result.numpy(), expected)


def test_isclose_matches_methods_and_rejects_invalid_tolerances():
    tensor = mt.tensor([1, 2, 3], dtype="int64")
    other = [1.0, 2.0, 4.0]

    functional_result = mt.functional.isclose(tensor, other)
    method_result = tensor.isclose(other)

    np.testing.assert_array_equal(
        functional_result.numpy(), np.array([True, True, False])
    )
    np.testing.assert_array_equal(method_result.numpy(), functional_result.numpy())

    with pytest.raises(ValueError):
        mt.isclose(tensor, tensor, rtol=-1.0)


def test_matmul_vector_gradients():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((4, 3))
    v = rng.standard_normal((3,))
    N = rng.standard_normal((3, 5))

    # matrix @ vector: d/dM (w·(M@v)) = outer(w, v), d/dv = M^T @ w
    mM = mt.Tensor(M.tolist(), dtype="float64", requires_grad=True)
    mv = mt.Tensor(v.tolist(), dtype="float64", requires_grad=True)
    out = mM.matmul(mv)
    w = rng.standard_normal(out.numpy().shape)
    (out * mt.Tensor(w.tolist(), dtype="float64")).sum().backward()
    np.testing.assert_allclose(mM.grad.numpy(), np.outer(w, v), rtol=1e-6)
    np.testing.assert_allclose(mv.grad.numpy(), M.T @ w, rtol=1e-6)

    # vector @ matrix
    mv2 = mt.Tensor(v.tolist(), dtype="float64", requires_grad=True)
    mN = mt.Tensor(N.tolist(), dtype="float64", requires_grad=True)
    out2 = mv2.matmul(mN)
    w2 = rng.standard_normal(out2.numpy().shape)
    (out2 * mt.Tensor(w2.tolist(), dtype="float64")).sum().backward()
    np.testing.assert_allclose(mv2.grad.numpy(), N @ w2, rtol=1e-6)
    np.testing.assert_allclose(mN.grad.numpy(), np.outer(v, w2), rtol=1e-6)

    # vector @ vector (dot): d/da (a·b) = b, d/db = a
    u = rng.standard_normal((3,))
    a = mt.Tensor(v.tolist(), dtype="float64", requires_grad=True)
    b = mt.Tensor(u.tolist(), dtype="float64", requires_grad=True)
    a.matmul(b).backward()
    np.testing.assert_allclose(a.grad.numpy(), u, rtol=1e-6)
    np.testing.assert_allclose(b.grad.numpy(), v, rtol=1e-6)
