# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`empty_cache()` hands the allocator's kept blocks back to the system.

Blocks of a megabyte or more that a tensor frees are kept for the next tensor
of the same size, which spares a loop over same-shaped batches the page faults
of fresh memory. The price is that freeing a large tensor no longer lowers the
process's resident memory -- with the system allocator alone, a block that
size is unmapped as it is freed. Nothing could ask for that memory back, so a
process done with its large tensors held up to 256 MiB for good.
"""

import os
import sys

import pytest

import minitensor as mt


def _resident_bytes():
    with open("/proc/self/statm") as statm:
        return int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="reads /proc/self/statm"
)
def test_a_freed_large_tensor_stays_resident_until_the_cache_is_emptied():
    size = 64 << 20
    mt.empty_cache()
    before = _resident_bytes()
    tensor = mt.ones([size // 4])
    del tensor
    held = _resident_bytes()
    assert held - before >= size // 2, "the freed block was not kept"

    mt.empty_cache()
    assert held - _resident_bytes() >= size // 2, "empty_cache kept the block"


def test_emptying_the_cache_leaves_live_tensors_alone():
    kept = mt.full([1 << 20], 3.0)
    freed = mt.full([1 << 20], 5.0)
    del freed
    mt.empty_cache()
    assert kept.sum().item() == 3.0 * (1 << 20)
    again = mt.full([1 << 20], 7.0)
    assert again.sum().item() == 7.0 * (1 << 20)
