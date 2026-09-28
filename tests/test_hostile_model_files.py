# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A damaged or hostile model file is refused when it is read.

Two ways that failed:

**A deployment model decoded without a limit.** Every other load goes through
one reader that bounds what a binary file can make the decoder allocate, but
`DeploymentModel.load` called the decoder directly. An 81-byte file whose first
length prefix was edited to 2^62 asked for 4.6e18 bytes, and the allocation
failure aborted the interpreter -- no exception, no traceback, the process gone.

**Tensors checked only when first used.** A file whose tensor bytes did not
match the shape beside them loaded without complaint and failed later, at
whatever first touched that tensor, with a message that named neither the
tensor nor the file and suggested checking the disk. Every tensor is now
checked as the file is read, and the error names it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

import minitensor as mt
from minitensor import nn


def test_a_corrupt_length_prefix_in_a_deployment_file_raises(tmp_path):
    model = nn.DenseLayer(2, 3)
    model.save(str(tmp_path / "model.bin"), "bin")
    serialized = mt.serialization.ModelSerializer.load(str(tmp_path / "model.bin"))
    serialized.to_deployment_model().save(str(tmp_path / "deploy.bin"))

    data = (tmp_path / "deploy.bin").read_bytes()
    # The name comes first, behind a one-byte varint length. 0xFD introduces an
    # eight-byte length, here 2^62.
    hostile = bytes([0xFD]) + (1 << 62).to_bytes(8, "little") + data[1:]
    (tmp_path / "hostile.bin").write_bytes(hostile)

    # In a child process: the failure this guards against is an abort, which
    # would take the test runner down with it.
    script = textwrap.dedent(f"""
        import minitensor as mt
        try:
            mt.serialization.DeploymentModel.load({str(tmp_path / "hostile.bin")!r})
        except OSError:
            print("refused")
        """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "refused"


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(lambda w: w.update(data=w["data"][:4]), id="truncated"),
        pytest.param(lambda w: w["shape"].update(dims=[1000, 1000]), id="inflated"),
        pytest.param(lambda w: w.update(data=w["data"][:-1]), id="odd-bytes"),
        pytest.param(lambda w: w.update(dtype="Float64"), id="wrong-dtype"),
    ],
)
def test_a_tensor_whose_bytes_do_not_match_its_shape_is_refused_by_name(
    tmp_path, tamper
):
    path = tmp_path / "model.json"
    nn.DenseLayer(2, 3).save(str(path), "json")
    saved = json.loads(path.read_text())
    tamper(saved["state_dict"]["parameters"]["weight"])
    path.write_text(json.dumps(saved))

    with pytest.raises(OSError, match="parameter `weight`"):
        nn.DenseLayer.load_state_from(str(path), "json")


def test_an_intact_file_still_loads(tmp_path):
    path = tmp_path / "model.json"
    model = nn.DenseLayer(2, 3)
    model.save(str(path), "json")
    state = nn.DenseLayer.load_state_from(str(path), "json")
    assert tuple(state["weight"].shape) == tuple(model.weight.shape)
