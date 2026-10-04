# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Betas are spelled the same way for every optimizer that takes them.

`NAdam` took only `beta1`/`beta2`, so `NAdam(params, betas=(0.9, 0.99))` was a
TypeError where every other Adam-family optimizer accepted it. And naming one
beta alone was refused everywhere else, though `NAdam` allowed it; it now
keeps the default for the other, as any keyword left out does.
"""

import pytest

import minitensor as mt
import minitensor.optim as optim

FAMILY = {
    "Adam": (0.9, 0.999),
    "AdamW": (0.9, 0.999),
    "Adamax": (0.9, 0.999),
    "NAdam": (0.9, 0.999),
    "RAdam": (0.9, 0.999),
    "Lion": (0.9, 0.99),
}


def _params():
    return [mt.tensor([1.0, 2.0], requires_grad=True)]


@pytest.mark.parametrize("name", list(FAMILY))
def test_betas_pair_is_accepted(name):
    opt = getattr(optim, name)(_params(), betas=(0.8, 0.95))
    assert (opt.beta1, opt.beta2) == pytest.approx((0.8, 0.95))


@pytest.mark.parametrize("name", list(FAMILY))
def test_one_beta_keeps_the_default_for_the_other(name):
    default = FAMILY[name]
    first = getattr(optim, name)(_params(), beta1=0.7)
    assert (first.beta1, first.beta2) == pytest.approx((0.7, default[1]))
    second = getattr(optim, name)(_params(), beta2=0.9)
    assert (second.beta1, second.beta2) == pytest.approx((default[0], 0.9))


@pytest.mark.parametrize("name", list(FAMILY))
def test_the_pair_and_a_single_beta_together_are_refused(name):
    with pytest.raises(TypeError, match="not both"):
        getattr(optim, name)(_params(), betas=(0.8, 0.9), beta1=0.5)


@pytest.mark.parametrize("name", list(FAMILY))
@pytest.mark.parametrize("bad", [1.0, -0.1, float("nan")])
def test_a_beta_outside_the_unit_interval_is_refused(name, bad):
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        getattr(optim, name)(_params(), betas=(bad, 0.9))
