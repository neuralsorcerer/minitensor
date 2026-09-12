# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Differentiable cluster assignment.

A hard assignment -- each point to its nearest centroid -- has a derivative of
zero almost everywhere and no derivative at all on the boundaries, so a loss
computed through one tells an optimizer nothing about where the centroids
should move. The standard repair is to soften it: assign each point to every
centroid with a weight that falls off with distance, and let the temperature
say how close that is to the hard answer.

    responsibility[n, k] = softmax_k(-||x[n] - c[k]||^2 / (2 * temperature))

The forward is one distance matrix and one softmax; what makes this worth a
`Function` of its own is the backward, which is written against the distances
rather than composed out of the pieces. Composing it would keep the `(n, k, d)`
difference tensor alive for the backward pass -- `d` times the memory of the
answer -- where the derivative needs only the `(n, k)` responsibilities and the
inputs, because

    d/dc[k] sum_n L = sum_n r[n, k] (x[n] - c[k]) / temperature * (dL/dr row term)

collapses back to a matrix product. That is the shape of every kernel worth
adding here: the same value, a smaller footprint on the tape.
"""

from __future__ import annotations

from typing import Any

from .. import _core as _C
from ..autograd import Function, FunctionCtx

__all__ = ["soft_assignment"]

_F = _C.functional


def _squared_distances(points: Any, centroids: Any) -> Any:
    """`[n, k]` squared euclidean distances, without forming `[n, k, d]`.

    `||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2`, which is two reductions and a
    matrix product. The expanded difference tensor is the obvious way to write
    it and costs `d` times as much memory.
    """

    point_norms = (points * points).sum(1, True)
    centroid_norms = (centroids * centroids).sum(1, False)
    cross = _F.matmul(points, centroids.transpose(0, 1))
    return point_norms - cross * 2.0 + centroid_norms


class SoftAssignment(Function):
    """`softmax(-d^2 / 2T)` over centroids, with a backward over `[n, k]`."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: FunctionCtx, points: Any, centroids: Any, temperature: float
    ) -> Any:
        logits = _squared_distances(points, centroids) / (-2.0 * temperature)
        responsibilities = _F.softmax(logits, 1)
        ctx.save_for_backward(points, centroids, responsibilities)
        ctx.temperature = temperature
        return responsibilities

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Any) -> Any:  # type: ignore[override]
        points, centroids, responsibilities = ctx.saved_tensors
        temperature = ctx.temperature

        # Through the softmax: `r * (g - sum_k g r)`, the usual Jacobian-vector
        # product, which needs the responsibilities and nothing else.
        weighted = responsibilities * grad_output
        through_softmax = weighted - responsibilities * weighted.sum(1, True)

        # Through `logits = -d^2 / 2T`, where `d(d^2)/dx[n] = 2(x[n] - c[k])`.
        # Summed over `k` this is a matrix product against the centroids, so
        # the `[n, k, d]` differences never exist.
        scaled = through_softmax / (-1.0 * temperature)
        row_totals = scaled.sum(1, True)
        column_totals = scaled.sum(0, False).reshape([-1, 1])

        grad_points = points * row_totals - _F.matmul(scaled, centroids)
        grad_centroids = centroids * column_totals - _F.matmul(
            scaled.transpose(0, 1), points
        )
        return grad_points, grad_centroids, None


def soft_assignment(points: Any, centroids: Any, temperature: float = 1.0) -> Any:
    """Responsibilities of each centroid for each point, differentiably.

    `points` is `[n, d]`, `centroids` is `[k, d]`, and the answer is `[n, k]`
    with each row summing to one. As `temperature` falls the rows approach the
    one-hot nearest-centroid assignment; at any positive temperature the
    gradient reaches both arguments.
    """

    # Checked here rather than in the forward: an argument the operation
    # cannot use is the caller's mistake, and saying so before a graph node
    # exists gives them their own traceback instead of one wrapped in a report
    # that the forward raised.
    if not temperature > 0.0:
        raise ValueError(
            f"soft_assignment needs a positive temperature, got {temperature}"
        )
    return SoftAssignment.apply(points, centroids, temperature)
