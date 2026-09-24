# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A Python plugin's callbacks run when its registry says they do.

They used to be stored and never called: `register` ran no initialize code and
`unregister` no cleanup, while the documentation's own example printed from
both.
"""

from __future__ import annotations

import pytest

import minitensor.plugins as plugins


def _plugin(name="callbacks"):
    return (
        plugins.PluginBuilder()
        .name(name)
        .version(plugins.VersionInfo(1, 0, 0))
        .description("records its callbacks")
        .author("tests")
        .min_minitensor_version(plugins.VersionInfo(0, 1, 0))
        .build()
    )


def test_register_initializes_and_unregister_cleans_up():
    calls = []
    plugin = _plugin()
    plugin.set_initialize_fn(lambda registry: calls.append(("init", registry)))
    plugin.set_cleanup_fn(lambda registry: calls.append(("cleanup", registry)))

    registry = plugins.PluginRegistry()
    registry.register(plugin)
    assert [kind for kind, _ in calls] == ["init"]
    assert calls[0][1] is registry

    registry.unregister("callbacks")
    assert [kind for kind, _ in calls] == ["init", "cleanup"]
    assert calls[1][1] is registry


def test_the_plugin_is_registered_by_the_time_it_initializes():
    seen = []
    plugin = _plugin()
    plugin.set_initialize_fn(
        lambda registry: seen.append(registry.is_registered("callbacks"))
    )
    plugins.PluginRegistry().register(plugin)
    assert seen == [True]


def test_a_failing_initialize_undoes_the_registration():
    def fail(registry):
        raise RuntimeError("cannot start")

    plugin = _plugin()
    plugin.set_initialize_fn(fail)
    registry = plugins.PluginRegistry()
    with pytest.raises(RuntimeError, match="cannot start"):
        registry.register(plugin)
    assert not registry.is_registered("callbacks")
    # And it can be registered again once the callback is fixed.
    plugin.set_initialize_fn(lambda registry: None)
    registry.register(plugin)
    assert registry.is_registered("callbacks")


def test_a_callback_may_use_the_registry_it_is_handed():
    helper = _plugin("helper")
    main = _plugin("main")
    main.set_initialize_fn(lambda registry: registry.register(helper))
    main.set_cleanup_fn(lambda registry: registry.unregister("helper"))

    registry = plugins.PluginRegistry()
    registry.register(main)
    assert registry.is_registered("helper")
    registry.unregister("main")
    assert not registry.is_registered("helper")


def test_plugins_without_callbacks_still_register():
    registry = plugins.PluginRegistry()
    registry.register(_plugin())
    registry.unregister("callbacks")
    assert not registry.is_registered("callbacks")
