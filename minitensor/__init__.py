# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public Python API surface that directly re-exports the Rust backend."""

from __future__ import annotations

import collections.abc as _collections_abc
import copyreg as _copyreg
import inspect as _inspect
import os as _os
import sys as _sys
import types as _types
from contextlib import contextmanager as _contextmanager

# Dense products go to the BLAS NumPy loads, and after each one OpenBLAS keeps
# its threads busy-waiting for the next -- by default for 2^28 cycles, about
# 130ms on a 2.1GHz core, without yielding. The engine's own pool runs the
# operations between those products on the same cores, so its workers waited
# for a scheduler tick behind the spinning threads: an MLP training step
# averaged 1.72ms with a 99th percentile of 13-21ms. Spinning for 2^22 cycles,
# 2ms, averages 1.39ms. Shorter still was faster here but put a wake-up on
# NumPy's own products a millisecond apart, which this length does not.
# OpenBLAS reads the variable once, as it loads, so it only takes effect when
# NumPy has not been imported yet, and never over a value already set.
_os.environ.setdefault("OPENBLAS_THREAD_TIMEOUT", "22")

from . import _api as _api_helpers
from . import _core as _C
from . import _elementwise, _matrix, _nn_extras
from ._api import _CORE_API_MODULES, _OPTIONAL_API_MODULES
from ._calculus import gradient
from ._derived import (
    average,
    cdist,
    corrcoef,
    cov,
    diff,
    digitize,
    dist,
    ediff1d,
    histogram2d,
    histogramdd,
    interp,
    kron,
    nancumprod,
    nancumsum,
    nanpercentile,
    normalize,
    outer,
    pairwise_distance,
    pdist,
    percentile,
    ptp,
    trapezoid,
    trapz,
    vdot,
)
from ._exports import (
    _FUNCTIONAL_FORWARDERS,
    _bind_functional_forwarders,
    _ensure_unique_names,
)
from ._indexing import (
    argwhere,
    block_diag,
    cartesian_prod,
    choose,
    compress,
    diag_indices,
    diagflat,
    diagonal_scatter,
    extract,
    flatnonzero,
    index_add,
    index_copy,
    index_fill,
    intersect1d,
    isin,
    masked_scatter,
    put,
    put_along_axis,
    ravel_multi_index,
    select,
    select_scatter,
    setdiff1d,
    setxor1d,
    slice_scatter,
    take,
    take_along_axis,
    take_along_dim,
    tril_indices,
    trim_zeros,
    triu_indices,
    union1d,
    unique,
    unique_all,
    unique_counts,
    unique_inverse,
    unique_values,
    unravel_index,
)
from ._sampling import bernoulli, multinomial, normal
from ._shape import (
    append,
    argpartition,
    array_split,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    block,
    broadcast_arrays,
    broadcast_shapes,
    broadcast_tensors,
    broadcast_to,
    can_broadcast,
    column_stack,
    combinations,
    cumulative_sum,
    delete,
    dsplit,
    dstack,
    expand_dims,
    fliplr,
    flipud,
    fromfunction,
    geomspace,
    hsplit,
    hstack,
    indices,
    insert,
    ix_,
    kthvalue,
    lexsort,
    matrix_transpose,
    meshgrid,
    msort,
    packbits,
    partition,
    permute_dims,
    resize,
    rot90,
    tensor_split,
    tile,
    tri,
    unbind,
    unflatten,
    unpackbits,
    unstack,
    vsplit,
    vstack,
)
from ._signal import (
    bartlett_window,
    blackman_window,
    convolve,
    correlate,
    hamming_window,
    hann_window,
    kaiser_window,
)

_api_namespace = globals()
available_submodules = _api_helpers._bind_namespace(
    _api_helpers.available_submodules, namespace=_api_namespace
)
list_public_api = _api_helpers._bind_namespace(
    _api_helpers.list_public_api, namespace=_api_namespace
)
api_summary = _api_helpers._bind_namespace(
    _api_helpers.api_summary, namespace=_api_namespace
)
search_api = _api_helpers._bind_namespace(
    _api_helpers.search_api, namespace=_api_namespace
)
_module_public_names = _api_helpers._bind_namespace(
    _api_helpers._module_public_names, namespace=_api_namespace
)
describe_api = _api_helpers._bind_namespace(
    _api_helpers.describe_api, namespace=_api_namespace
)
help = _api_helpers._bind_namespace(_api_helpers.help, namespace=_api_namespace)
_iter_public_names = _api_helpers._iter_public_names
_api_module_names = _api_helpers._bind_namespace(
    _api_helpers._api_module_names, namespace=_api_namespace
)
_api_module_namespace = _api_helpers._bind_namespace(
    _api_helpers._api_module_namespace, namespace=_api_namespace
)
_api_module_unavailable_error = _api_helpers._api_module_unavailable_error
_api_module_title = _api_helpers._api_module_title
_resolve_symbol = _api_helpers._bind_namespace(
    _api_helpers._resolve_symbol, namespace=_api_namespace
)

try:  # pragma: no cover - fallback when version metadata missing
    from ._version import __version__, __version_tuple__
except ImportError:  # pragma: no cover
    __version__ = "0.1.0"
    __version_tuple__ = (0, 1, 0)

_rust_version = getattr(_C, "__version__", None)
if _rust_version:
    __version__ = _rust_version

Tensor = _C.Tensor
tensor = Tensor

Device = _C.Device
device = Device
cpu = Device.cpu
cuda = Device.cuda

zeros = Tensor.zeros
ones = Tensor.ones
empty = Tensor.empty
rand = Tensor.rand
randn = Tensor.randn
truncated_normal = Tensor.truncated_normal
rand_like = Tensor.rand_like
randn_like = Tensor.randn_like
truncated_normal_like = Tensor.truncated_normal_like
randint = Tensor.randint
randint_like = Tensor.randint_like
randperm = Tensor.randperm
eye = Tensor.eye


def identity(n, dtype=None, device=None, requires_grad=False):
    """The `n` by `n` identity matrix, a square `eye`."""

    return Tensor.eye(n, n, dtype=dtype, device=device, requires_grad=requires_grad)


full = Tensor.full
full_like = Tensor.full_like
uniform = Tensor.uniform
uniform_like = Tensor.uniform_like
xavier_uniform = Tensor.xavier_uniform
xavier_uniform_like = Tensor.xavier_uniform_like
xavier_normal = Tensor.xavier_normal
xavier_normal_like = Tensor.xavier_normal_like
he_uniform = Tensor.he_uniform
he_uniform_like = Tensor.he_uniform_like
he_normal = Tensor.he_normal
he_normal_like = Tensor.he_normal_like
lecun_uniform = Tensor.lecun_uniform
lecun_uniform_like = Tensor.lecun_uniform_like
lecun_normal = Tensor.lecun_normal
lecun_normal_like = Tensor.lecun_normal_like
empty_like = Tensor.empty_like
zeros_like = Tensor.zeros_like
ones_like = Tensor.ones_like
linspace = Tensor.linspace
logspace = Tensor.logspace
arange = Tensor.arange
from_numpy = Tensor.from_numpy
from_numpy_shared = Tensor.from_numpy_shared
as_tensor = Tensor.as_tensor

get_default_dtype = _C.get_default_dtype
set_default_dtype = _C.set_default_dtype
manual_seed = _C.manual_seed
empty_cache = _C.empty_cache
get_gradient = _C.get_gradient
clear_autograd_graph = _C.clear_autograd_graph
autograd_graph_size = _C.autograd_graph_size
is_autograd_graph_consumed = _C.is_autograd_graph_consumed
mark_autograd_graph_consumed = _C.mark_autograd_graph_consumed
no_grad = _C.no_grad
enable_grad = _C.enable_grad
is_grad_enabled = _C.is_grad_enabled
set_grad_enabled = _C.set_grad_enabled

functional = _C.functional
_sys.modules[__name__ + ".functional"] = functional

nn = _C.nn
_sys.modules[__name__ + ".nn"] = nn

# The free-function forms of the operators, and their second spellings. Each
# is written once, in `_elementwise`, and put into `functional` here -- before
# the forwarder pass below, which is then the single mechanism that carries
# every one of them to the top level, exactly as it does for the ops that come
# out of the extension. They go onto `Tensor` as well, since each takes its
# tensor first, so `mt.add(a, b)` and `a.add(b)` are one definition.
for _module, _names in (
    (_elementwise, _elementwise._ELEMENTWISE),
    (_matrix, _matrix._MATRIX),
):
    for _elementwise_name in _names:
        _member = getattr(_module, _elementwise_name)
        setattr(functional, _elementwise_name, _member)
        # A name the extension already implements as a method keeps it: the
        # free function above delegates to that method, so overwriting it here
        # would make it call itself.
        if not hasattr(Tensor, _elementwise_name):
            setattr(Tensor, _elementwise_name, _member)

for _alias, _target in _elementwise._ALIASES.items():
    setattr(functional, _alias, getattr(functional, _target))
    if hasattr(Tensor, _target):
        setattr(Tensor, _alias, getattr(Tensor, _target))

# The Python-level pieces of `nn`. They are attached before the mirror below
# copies `nn` into `functional`, so both namespaces carry them and neither has
# to name them twice.
for _extra_name in _nn_extras._NN_EXTRAS:
    setattr(nn, _extra_name, getattr(_nn_extras, _extra_name))


def _reduce_to_constructor(self):
    """Rebuild a loss from the keywords that built it.

    A loss holds its configuration and nothing else, and each piece is
    readable under the name of the keyword that set it -- so that is all
    `pickle`, `copy` and `deepcopy` need. Without it none of the three could
    handle a loss, which left a training configuration holding one
    uncopyable as a whole.
    """
    keywords = {
        name: getattr(self, name) for name in _inspect.signature(type(self)).parameters
    }
    return (_copyreg.__newobj_ex__, (type(self), (), keywords))


for _loss_name in dir(nn):
    if _loss_name.endswith("Loss") and isinstance(getattr(nn, _loss_name), type):
        getattr(nn, _loss_name).__reduce__ = _reduce_to_constructor
del _loss_name


def _rebuild_layer(cls, keywords, state, training, frozen):
    """The other half of `_reduce_layer`."""
    layer = cls(**keywords)
    if state:
        layer.load_state_dict(state)
    if frozen:
        layer.requires_grad_(False)
    layer.train(training)
    return layer


def _rebuild_sequential(named, training, modes):
    """The other half of `_reduce_layer` for a `Sequential`.

    The children arrive rebuilt, each with its own state, freezing and mode.
    The container's mode is set first and the children's after it, since
    setting a container's mode sets its children's too.
    """
    model = nn.Sequential()
    for name, child in named:
        # A module added unnamed reports its position as its name, and a
        # position is not a name `add_module` takes.
        if name.isdigit():
            model.append(child)
        else:
            model.add_module(name, child)
    model.train(training)
    for child, mode in zip(model, modes):
        child.train(mode)
    return model


def _reduce_layer(self):
    """Rebuild a built-in layer from its constructor keywords and its state.

    Every keyword a layer's constructor takes reads back under its own name,
    so the configuration is the constructor call that made it; the trained
    values travel as its state dict, which holds tensors and so pickles. The
    training mode and whether the layer is frozen come along, so the copy is
    the same layer in every respect anything can observe.

    `pickle`, `copy.copy` and `copy.deepcopy` all go through this. A copy made
    by `copy.copy` is therefore independent too: a layer's parameters belong
    to it, and two layers cannot share them.
    """
    cls = type(self)
    if cls is nn.Sequential:
        children = self.named_children()
        return (
            _rebuild_sequential,
            (children, self.training, [child.training for child in self]),
        )
    keywords = {}
    parameters = list(self.parameters())
    for name in _inspect.signature(cls).parameters:
        if name == "device":
            continue
        if name == "dtype":
            if parameters:
                keywords[name] = str(parameters[0].dtype)
            continue
        value = getattr(self, name)
        if name == "bias" and not isinstance(value, bool):
            # Layers with one bias tensor report the tensor; the constructor
            # asks whether there is one.
            value = value is not None
        keywords[name] = value
    state = dict(self.state_dict())
    # Not `any`: this module's own `any` is the tensor reduction.
    frozen = bool(parameters) and not [p for p in parameters if p.requires_grad]
    return (_rebuild_layer, (cls, keywords, state, self.training, frozen))


for _layer_name in dir(nn):
    _layer = getattr(nn, _layer_name)
    if (
        isinstance(_layer, type)
        and issubclass(_layer, nn.Module)
        and _layer is not nn.Module
        and not _layer_name.endswith("Loss")
    ):
        _layer.__reduce__ = _reduce_layer
del _layer_name, _layer


def _copy_sequential(self):
    """`copy.copy` of a `Sequential` is its deep copy.

    Its layers can belong to only one container, so a copy cannot hold the
    same ones; and every layer's copy is independent anyway.
    """
    import copy

    return copy.deepcopy(self)


nn.Sequential.__copy__ = _copy_sequential

optim = _C.optim
_sys.modules[__name__ + ".optim"] = optim

from . import autograd  # noqa: E402  (after `_C`, which it imports)
from . import kernels  # noqa: E402  (after `autograd`, which it builds on)

_sys.modules[__name__ + ".autograd"] = autograd
_sys.modules[__name__ + ".kernels"] = kernels

from .gradcheck import gradcheck  # noqa: E402  (after `_C`, which it imports)

numpy_compat = getattr(_C, "numpy_compat", None)
if numpy_compat is not None:
    _sys.modules[__name__ + ".numpy_compat"] = numpy_compat
    cross = getattr(numpy_compat, "cross", None)
    # These four are the same operations the top level already provides, and
    # what makes them themselves is the rank promotion: `vstack` of two vectors
    # is two rows, `hstack` of two vectors is one longer one, and `hsplit` cuts
    # a vector along the only axis it has. The compiled module used to carry
    # its own `concatenate`-on-a-fixed-axis versions with none of that, so
    # `numpy_compat.vstack([v, v])` gave one long vector where it should give
    # two rows, and `hstack` and `hsplit` raised `IndexError` outright. Installing
    # the real ones keeps one implementation rather than a second, worse one.
    for _name in ("vstack", "hstack", "hsplit", "vsplit"):
        setattr(numpy_compat, _name, globals()[_name])
    del _name
else:
    cross = None

plugins = getattr(_C, "plugins", None)
if plugins is not None:
    _sys.modules[__name__ + ".plugins"] = plugins

serialization = getattr(_C, "serialization", None)
if serialization is not None:
    _sys.modules[__name__ + ".serialization"] = serialization
    # A state dict has the whole read side of a mapping -- subscripting,
    # `len`, `in`, iteration, `keys`, `values`, `items`, `get` -- so code that
    # checks for one before reading it should find one.
    _collections_abc.Mapping.register(serialization.StateDict)

_OPTIONAL_TOP_LEVEL_EXPORTS = (
    "register_custom_op",
    "execute_custom_op_py",
    "is_custom_op_registered_py",
    "list_custom_ops_py",
    "register_example_custom_ops",
    "unregister_custom_op_py",
)

for _name in _OPTIONAL_TOP_LEVEL_EXPORTS:
    _member = getattr(_C, _name, None)
    if _member is not None:
        globals()[_name] = _member


@_contextmanager
def default_dtype(dtype: str):
    """Temporarily switch the global default dtype within a ``with`` block.

    This helper restores the previous default dtype even if an exception is
    raised inside the managed block. It relies on the Rust backend for
    validation so any invalid ``dtype`` values will propagate the backend
    ``ValueError`` after ensuring the prior dtype is reinstated.

    Parameters
    ----------
    dtype:
        The name of the dtype to activate (for example ``"float64"``).
    """

    previous = get_default_dtype()
    if isinstance(dtype, str) and dtype == previous:
        yield
        return

    try:
        set_default_dtype(dtype)
        yield
    finally:
        set_default_dtype(previous)


# `functional.partition` was the raw two-output selection kernel, which is not
# what `mt.partition` is: it took its positions as a sequence and returned a
# pair, so `F.partition(x, 2)` -- the same call that works at the top level --
# answered "'int' object is not an instance of 'Sequence'". One name meant two
# things across two public namespaces. It now means the wrapper in both, and
# `argpartition` joins it, which is the arrangement every other Python-level
# op here already has.
for _selection_name in ("partition", "argpartition", "unique"):
    setattr(functional, _selection_name, globals()[_selection_name])

_bind_functional_forwarders(_FUNCTIONAL_FORWARDERS, globals())

for _name in dir(nn):
    if _name.startswith("_") or not _name:
        continue
    _member = getattr(nn, _name)
    if callable(_member) and _name[0].islower():
        setattr(functional, _name, _member)

dot = getattr(functional, "dot")
bmm = getattr(functional, "bmm")

_TENSOR_EXPORTS = (
    "Tensor",
    "tensor",
    "zeros",
    "ones",
    "empty",
    "rand",
    "randn",
    "rand_like",
    "randn_like",
    "truncated_normal",
    "truncated_normal_like",
    "uniform",
    "uniform_like",
    "xavier_uniform",
    "xavier_uniform_like",
    "xavier_normal",
    "xavier_normal_like",
    "he_uniform",
    "he_uniform_like",
    "he_normal",
    "he_normal_like",
    "lecun_uniform",
    "lecun_uniform_like",
    "lecun_normal",
    "lecun_normal_like",
    "randint",
    "randint_like",
    "randperm",
    "eye",
    "full",
    "full_like",
    "empty_like",
    "zeros_like",
    "ones_like",
    "linspace",
    "logspace",
    "arange",
    "from_numpy",
    "from_numpy_shared",
    "as_tensor",
    "get_default_dtype",
    "set_default_dtype",
    "manual_seed",
    "empty_cache",
    "default_dtype",
)
_ensure_unique_names(_TENSOR_EXPORTS, "tensor exports")

_tensor_module = _types.ModuleType(__name__ + ".tensor")
for _name in _TENSOR_EXPORTS:
    setattr(_tensor_module, _name, globals()[_name])

_sys.modules[_tensor_module.__name__] = _tensor_module


_BASE_EXPORTS = (
    *_TENSOR_EXPORTS,
    "Device",
    "device",
    "cpu",
    "cuda",
    "available_submodules",
    "list_public_api",
    "api_summary",
    "broadcast_to",
    "broadcast_shapes",
    "broadcast_tensors",
    "can_broadcast",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "meshgrid",
    "hstack",
    "vstack",
    "dstack",
    "column_stack",
    "tile",
    "unbind",
    "tensor_split",
    "fliplr",
    "flipud",
    "rot90",
    "outer",
    "vdot",
    "kron",
    "dist",
    "cdist",
    "diff",
    "trapezoid",
    "trapz",
    "cov",
    "corrcoef",
    "histogramdd",
    "histogram2d",
    "average",
    "ptp",
    "percentile",
    "nanpercentile",
    "interp",
    "digitize",
    "nancumsum",
    "nancumprod",
    "ediff1d",
    "normalize",
    "pairwise_distance",
    "pdist",
    "take",
    "take_along_dim",
    "union1d",
    "unique",
    "intersect1d",
    "setdiff1d",
    "setxor1d",
    "trim_zeros",
    "geomspace",
    "tri",
    "indices",
    "ix_",
    "fromfunction",
    "hann_window",
    "hamming_window",
    "blackman_window",
    "bartlett_window",
    "kaiser_window",
    "correlate",
    "convolve",
    "packbits",
    "unpackbits",
    "expand_dims",
    "permute_dims",
    "matrix_transpose",
    "unstack",
    "array_split",
    "append",
    "delete",
    "insert",
    "resize",
    "block",
    "cumulative_sum",
    "identity",
    "broadcast_arrays",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "take_along_axis",
    "put_along_axis",
    "compress",
    "extract",
    "choose",
    "partition",
    "argpartition",
    "lexsort",
    "index_add",
    "index_copy",
    "index_fill",
    "masked_scatter",
    "select",
    "slice_scatter",
    "select_scatter",
    "diagonal_scatter",
    "flatnonzero",
    "argwhere",
    "isin",
    "tril_indices",
    "triu_indices",
    "diag_indices",
    "put",
    "unravel_index",
    "ravel_multi_index",
    "diagflat",
    "block_diag",
    "cartesian_prod",
    "unflatten",
    "msort",
    "hsplit",
    "vsplit",
    "dsplit",
    "kthvalue",
    "combinations",
    "gradient",
    "bernoulli",
    "normal",
    "multinomial",
    "search_api",
    "describe_api",
    "help",
    "get_gradient",
    "clear_autograd_graph",
    "autograd_graph_size",
    "is_autograd_graph_consumed",
    "mark_autograd_graph_consumed",
    "no_grad",
    "enable_grad",
    "is_grad_enabled",
    "set_grad_enabled",
    "functional",
    "nn",
    "optim",
    "autograd",
    "kernels",
    "gradcheck",
    "numpy_compat",
    "cross",
    "plugins",
    "serialization",
    "dot",
    "bmm",
)
_ensure_unique_names(_BASE_EXPORTS, "base exports")

_ALL_EXPORT_CANDIDATES = (
    *_BASE_EXPORTS,
    *_OPTIONAL_TOP_LEVEL_EXPORTS,
    *_FUNCTIONAL_FORWARDERS,
)
_ensure_unique_names(_ALL_EXPORT_CANDIDATES, "top-level public exports")

__all__ = [name for name in _ALL_EXPORT_CANDIDATES if name in globals()]

# A compiled submodule's `__all__` lists what the extension registered, and
# nothing attached to it from here joined the list -- so `from minitensor.nn
# import *` left out `conv3d`, `embedding` and the rest of the Python-level
# layer functions, and `functional` left out more than half of itself. Each
# extended namespace now advertises everything public it holds, in its own
# order first.
for _extended in (functional, nn, numpy_compat):
    if _extended is None:
        continue
    _listed = getattr(_extended, "__all__", ())
    _listed = list(_listed) if isinstance(_listed, (list, tuple)) else []
    _attached = sorted(
        _name
        for _name in dir(_extended)
        if not _name.startswith("_")
        and _name not in _listed
        and not isinstance(getattr(_extended, _name), _types.ModuleType)
    )
    _extended.__all__ = _listed + _attached
del _extended, _listed, _attached
