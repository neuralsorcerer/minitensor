# MiniTensor Documentation

Welcome to the MiniTensor documentation. This directory contains focused guides
for installing, using, extending, benchmarking, and contributing to MiniTensor.

```{toctree}
:hidden:
:maxdepth: 2
:caption: User guides

installation
api_reference
custom_operations
plugin_system
performance
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Contributor guides

development
```

## Start here

- [Installation guide](installation.md) -- supported installation paths for PyPI,
  source builds, editable installs, and contributor environments.
- [API reference](api_reference.md) -- consolidated public Python API surface,
  including top-level helpers, tensors, functional operations, neural-network
  layers, optimizers, NumPy compatibility, serialization, plugins, and debug
  utilities.
- [Development guide](development.md) -- repository layout, environment setup,
  validation commands, documentation workflow, and release checks.

## Feature guides

- [Custom operations](custom_operations.md) -- how the Rust custom-operation
  trait, registry, builder, and Python helpers fit together.
- [Plugin system](plugin_system.md) -- plugin metadata, compatibility checks,
  Python plugin registries, dynamic loading, and best practices.
- [Performance benchmarks](performance.md) -- benchmark commands,
  interpretation guidance, profiling tips, and optimization checklist.

## Examples and notebooks

The repository also includes executable examples under [`../examples`](../examples)
and notebooks under [`../examples/notebooks`](../examples/notebooks). Rebuild the
Rust extension before running examples if you changed code in `engine/` or
`bindings/`.

## Citation and paper

If you use MiniTensor in academic or research work, please cite the accompanying
paper:

> Soumyadip Sarkar. *MiniTensor: A Lightweight, High-Performance Tensor
> Operations Library*. arXiv:2602.00125, 2026.

- Paper: [arXiv:2602.00125](https://arxiv.org/abs/2602.00125)
- DOI: [10.48550/arXiv.2602.00125](https://doi.org/10.48550/arXiv.2602.00125)
- Citation metadata: [`CITATION.cff`](../CITATION.cff)

```bibtex
@misc{sarkar2026minitensorlightweighthighperformancetensor,
      title={MiniTensor: A Lightweight, High-Performance Tensor Operations Library},
      author={Soumyadip Sarkar},
      year={2026},
      eprint={2602.00125},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.00125},
}
```

## Documentation maintenance checklist

When adding or changing public functionality:

1. Update [the API reference](api_reference.md) for new symbols, parameters, or
   behavior changes.
2. Add or update a focused guide when the feature requires more than a short API
   entry.
3. Add an executable example or test for important user-facing behavior.
4. Run the validation commands in [the development guide](development.md#validation-commands).
