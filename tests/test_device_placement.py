# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device placement is refused where it cannot be honoured.

Every kernel in the engine reads host memory, so CPU is the only device a
tensor can live on. Asking for another one used to succeed: the tensor was
built out of host memory, reported ``device=cuda:0``, and then failed inside
the first operation applied to it with "Internal error ... please report it
with a minimal reproduction case". These tests pin the replacement behaviour --
the request fails where the device was named, with a message naming it.
"""

import numpy as np
import pytest

import minitensor as mt

UNAVAILABLE = ["cuda", "cuda:1", "metal", "opencl", "opencl:2"]


def _unavailable_devices():
    return [mt.Device(spec) for spec in UNAVAILABLE]


def test_cpu_is_the_only_available_device():
    assert mt.Device("cpu").is_available()
    for device in _unavailable_devices():
        assert not device.is_available(), device


# Every module-level entry point that accepts ``device=``. Each entry is
# (name, positional args) -- the callable is looked up on the module so a
# renamed or dropped constructor fails here loudly instead of silently
# dropping out of the sweep.
_CONSTRUCTORS = [
    ("empty", ((2, 2),)),
    ("zeros", ((2, 2),)),
    ("ones", ((2, 2),)),
    ("full", ((2, 2), 3.0)),
    ("eye", (3,)),
    ("arange", (0, 4)),
    ("linspace", (0.0, 1.0, 4)),
    ("logspace", (0.0, 1.0, 4)),
    ("rand", ((2, 2),)),
    ("randn", ((2, 2),)),
    ("randint", (0, 4, (2, 2))),
    ("randperm", (4,)),
    ("tensor", ([1.0, 2.0],)),
    ("as_tensor", ([1.0, 2.0],)),
    ("xavier_uniform", ((2, 2),)),
    ("xavier_normal", ((2, 2),)),
    ("he_uniform", ((2, 2),)),
    ("he_normal", ((2, 2),)),
    ("lecun_uniform", ((2, 2),)),
    ("lecun_normal", ((2, 2),)),
    ("uniform", ((2, 2),)),
    ("truncated_normal", ((2, 2),)),
]

_LIKE_CONSTRUCTORS = [
    "empty_like",
    "zeros_like",
    "ones_like",
    "rand_like",
    "randn_like",
    "xavier_uniform_like",
    "xavier_normal_like",
    "he_uniform_like",
    "he_normal_like",
    "lecun_uniform_like",
    "lecun_normal_like",
    "uniform_like",
    "truncated_normal_like",
]


@pytest.mark.parametrize("name,args", _CONSTRUCTORS)
def test_constructors_reject_unavailable_devices(name, args):
    fn = getattr(mt, name)
    for device in _unavailable_devices():
        with pytest.raises(RuntimeError, match="is not available"):
            fn(*args, device=device)


@pytest.mark.parametrize("name,args", _CONSTRUCTORS)
def test_constructors_accept_cpu_and_the_default(name, args):
    fn = getattr(mt, name)
    assert fn(*args).device == "cpu"
    assert fn(*args, device=mt.Device("cpu")).device == "cpu"


@pytest.mark.parametrize("name", _LIKE_CONSTRUCTORS)
def test_like_constructors_reject_unavailable_devices(name):
    fn = getattr(mt, name)
    reference = mt.ones((2, 2))
    for device in _unavailable_devices():
        with pytest.raises(RuntimeError, match="is not available"):
            fn(reference, device=device)
    assert fn(reference).device == "cpu"


def test_full_like_and_randint_like_reject_unavailable_devices():
    reference = mt.ones((2, 2))
    for device in _unavailable_devices():
        with pytest.raises(RuntimeError, match="is not available"):
            mt.full_like(reference, 2.0, device=device)
        with pytest.raises(RuntimeError, match="is not available"):
            mt.randint_like(reference, 0, 4, device=device)


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param("device-object", id="device-object"),
        pytest.param("string", id="string"),
        pytest.param("keyword", id="keyword"),
    ],
)
def test_to_rejects_unavailable_devices(spec):
    tensor = mt.ones((2, 2))
    for name in UNAVAILABLE:
        with pytest.raises(RuntimeError, match="is not available"):
            if spec == "device-object":
                tensor.to(mt.Device(name))
            elif spec == "string":
                tensor.to(name)
            else:
                tensor.to(device=name)


def test_to_cpu_still_works_and_dtype_conversion_is_unaffected():
    tensor = mt.ones((2, 2))
    assert tensor.to("cpu").device == "cpu"
    assert tensor.to(mt.Device("cpu")).device == "cpu"
    assert tensor.to("float64").dtype == "float64"
    assert tensor.to(device="cpu", dtype="int32").dtype == "int32"


def test_tensor_constructor_rejects_unavailable_devices():
    for device in _unavailable_devices():
        with pytest.raises(RuntimeError, match="is not available"):
            mt.Tensor([1.0, 2.0], device=device)
        with pytest.raises(RuntimeError, match="is not available"):
            mt.Tensor(np.zeros((2, 2), dtype=np.float32), device=device)


def test_layers_reject_unavailable_devices():
    for device in _unavailable_devices():
        with pytest.raises(RuntimeError, match="is not available"):
            mt.nn.DenseLayer(2, 2, device=device)


def test_error_message_names_the_device_that_was_asked_for():
    with pytest.raises(RuntimeError) as excinfo:
        mt.zeros((2, 2), device=mt.Device("opencl:3"))
    message = str(excinfo.value)
    assert "opencl:3" in message
    assert "cpu" in message


def test_rejection_happens_before_any_tensor_exists():
    # The old failure mode was a tensor that could be created and inspected but
    # not computed with. Nothing that reports a GPU device should be reachable.
    for device in _unavailable_devices():
        with pytest.raises(RuntimeError):
            created = mt.ones((2, 2), device=device)
            pytest.fail(f"created a tensor on {device}: {created.device}")
