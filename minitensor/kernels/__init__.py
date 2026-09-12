# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Domain kernels written as `autograd.Function`s.

Each of these is an operation a general array library has no reason to carry
and a specialised one cannot do without. They are here as much to be read as to
be used: every one is a worked example of the extension path, in the same
handful of lines a user would write for their own.
"""

from __future__ import annotations

from ._clustering import soft_assignment

__all__ = ["soft_assignment"]
