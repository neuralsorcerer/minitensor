# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def callable_attribute(tensor, name):
    """The named attribute of `tensor` if it is callable, else `None`.

    The sweeps that enumerate the tensor surface want its methods, and reach
    them with `getattr` over `dir(Tensor)`. A property is *evaluated* by that
    lookup, so one that refuses the probe's rank -- `mT` on a vector -- raises
    from the lookup itself rather than returning something uncallable. It is
    still not a method to sweep, so it is skipped like any other non-callable.
    """

    try:
        attribute = getattr(tensor, name, None)
    except Exception:
        return None
    return attribute if callable(attribute) else None


@pytest.fixture(scope="session")
def numeric_grad():
    """Central-difference derivative of a NumPy-valued function.

    Nine test modules had grown their own copy of this, in three spellings that
    differed in local variable names and in whether the callable took the array
    back. It is the NumPy-level counterpart to the public `minitensor.gradcheck`:
    that one takes tensors and checks a whole backward pass, while this one
    produces the expected array for a test that has already built its own loss
    out of NumPy.

    The helpers it does *not* replace are the ones that are not this function:
    the symmetric perturbation `test_cholesky` and `test_eigh` need, the extra
    weight argument in `test_autograd_shape_ops`, and the tensor-valued probes
    in `test_autograd_function`. Those are purpose-built, not duplicates.

    `f` is called with `arr` itself, mutated in place and restored, so a caller
    that closes over `arr` sees each perturbation either way.
    """

    def compute(f, arr, eps=1e-6):
        grad = np.zeros_like(arr)
        flat, gflat = arr.reshape(-1), grad.reshape(-1)
        for i in range(flat.size):
            original = flat[i]
            flat[i] = original + eps
            high = f(arr)
            flat[i] = original - eps
            low = f(arr)
            flat[i] = original
            gflat[i] = (high - low) / (2 * eps)
        return grad

    return compute
