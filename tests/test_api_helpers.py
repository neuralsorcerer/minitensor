# Copyright (c) 2026 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import minitensor as mt


def test_available_submodules_contains_expected_keys():
    flags = mt.available_submodules()
    assert set(flags.keys()) == {
        "functional",
        "nn",
        "optim",
        "numpy_compat",
        "plugins",
        "serialization",
    }


def test_list_public_api_has_top_level_tensor():
    api = mt.list_public_api()
    assert "top_level" in api
    assert "Tensor" in api["top_level"]


def test_api_summary_counts_match_list_public_api():
    api = mt.list_public_api()
    summary = mt.api_summary()
    assert summary["version"] == mt.__version__
    assert summary["available_submodules"] == mt.available_submodules()
    assert summary["counts"] == {module: len(names) for module, names in api.items()}


def test_search_api_filters_and_scopes():
    results = mt.search_api("tensor")
    assert any(item.endswith("Tensor") for item in results)

    top_level_hits = mt.search_api("tensor", module="top_level")
    assert "Tensor" in top_level_hits


def test_search_api_rejects_unknown_module():
    try:
        mt.search_api("tensor", module="unknown")
    except ValueError as exc:
        assert "Unknown module" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown module")


def test_help_prints_and_returns(capsys):
    output = mt.help()
    captured = capsys.readouterr()
    assert "MiniTensor" in output
    assert "MiniTensor" in captured.out
