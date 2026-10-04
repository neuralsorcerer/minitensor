# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`empty_cache()` hands the allocator's kept blocks back to the system.

Blocks of 64 KiB or more that a tensor frees are kept for the next tensor
of the same size, which spares a loop over same-shaped batches the page faults
of fresh memory. The price is that freeing a large tensor no longer lowers the
process's resident memory -- with the system allocator alone, a block that
size is unmapped as it is freed. Nothing could ask for that memory back, so a
process done with its large tensors held up to 256 MiB for good.
"""

import subprocess
import sys

import pytest

import minitensor as mt

# In a fresh process: resident memory is only a clean measure of one block
# before the heap has history. Late in a test run the system allocator can hold
# a large free region that is already resident, and carve the tensor from it
# without the count moving at all.
_MEASURE = """
import os
import minitensor as mt

def resident():
    with open("/proc/self/statm") as statm:
        return int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")

size = 64 << 20
before = resident()
tensor = mt.ones([size // 4])
del tensor
held = resident()
mt.empty_cache()
print(held - before, held - resident())
"""


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="reads /proc/self/statm"
)
def test_a_freed_large_tensor_stays_resident_until_the_cache_is_emptied():
    size = 64 << 20
    result = subprocess.run(
        [sys.executable, "-c", _MEASURE], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-2000:]
    kept, released = (int(value) for value in result.stdout.split())
    assert kept >= size // 2, "the freed block was not kept"
    assert released >= size // 2, "empty_cache kept the block"


def test_emptying_the_cache_leaves_live_tensors_alone():
    kept = mt.full([1 << 20], 3.0)
    freed = mt.full([1 << 20], 5.0)
    del freed
    mt.empty_cache()
    assert kept.sum().item() == 3.0 * (1 << 20)
    again = mt.full([1 << 20], 7.0)
    assert again.sum().item() == 7.0 * (1 << 20)


# A chain over half-megabyte tensors: each call frees several temporaries
# together, which is what had the system allocator return them and fault them
# back in on the next call while the floor of the cache was a megabyte.
_FAULTS = """
import resource
import minitensor as mt

w = mt.randn(128 << 10)
g = mt.randn(128 << 10)

def chain():
    return ((w * 2 + g) * 3 - g) * 0.5 + w.exp()

for _ in range(20):
    chain()
before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
for _ in range(100):
    chain()
print((resource.getrusage(resource.RUSAGE_SELF).ru_minflt - before) / 100)
"""


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="counts the process's page faults"
)
def test_a_loop_over_sub_megabyte_tensors_does_not_fault_every_call():
    result = subprocess.run(
        [sys.executable, "-c", _FAULTS], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-2000:]
    # Each fresh 512 KiB block is 128 pages; uncached, the chain made ~350.
    assert float(result.stdout) < 32
