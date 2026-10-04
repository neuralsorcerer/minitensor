# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Each public namespace's `__all__` lists everything public it holds.

The compiled submodules list what the extension registered, and the Python
package attaches more to them at import. None of that joined `__all__`, so
`from minitensor.nn import *` left out `conv3d`, `embedding` and the other
Python-level layer functions, and `functional` left out over half of itself.
"""

import types

import pytest

import minitensor as mt
import minitensor.functional as F
import minitensor.nn as nn

NAMESPACES = [nn, F, mt.numpy_compat]


def _public(module):
    return {
        name
        for name in dir(module)
        if not name.startswith("_")
        and not isinstance(getattr(module, name), types.ModuleType)
    }


@pytest.mark.parametrize("module", NAMESPACES, ids=lambda m: m.__name__)
def test_every_public_name_is_listed(module):
    assert sorted(_public(module) - set(module.__all__)) == []


@pytest.mark.parametrize("module", NAMESPACES, ids=lambda m: m.__name__)
def test_every_listed_name_exists_once(module):
    assert [name for name in module.__all__ if not hasattr(module, name)] == []
    assert len(module.__all__) == len(set(module.__all__))


def test_star_import_carries_the_python_level_functions():
    namespace = {}
    exec("from minitensor.nn import *", namespace)
    for name in ("conv3d", "embedding", "group_norm", "Conv2d", "LSTM"):
        assert name in namespace
    namespace = {}
    exec("from minitensor.functional import *", namespace)
    for name in ("cross_entropy", "conv1d", "add", "partition"):
        assert name in namespace
