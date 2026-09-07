# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

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
