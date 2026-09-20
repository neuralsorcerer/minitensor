// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
#[pymethods]
impl PyTensor {
    /// Join tensors along an existing axis.
    ///
    /// `dim`, not `axis`, and defaulted the way `stack` below defaults it:
    /// these two are the same pair of static constructors, and one of them
    /// wanting NumPy's keyword made the choice depend on which you reached
    /// for. The free spellings -- `cat`, `concat` -- have always said `dim`.
    /// `numpy_compat.concatenate` keeps `axis`, as everything in that module
    /// does.
    #[staticmethod]
    #[pyo3(signature = (tensors, dim=0))]
    pub fn concatenate(tensors: &Bound<PyList>, dim: isize) -> PyResult<PyTensor> {
        if tensors.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot concatenate empty list of tensors",
            ));
        }

        let tensor_vec: Vec<Tensor> = tensors
            .iter()
            .map(|obj| PyTensor::from_python_value(&obj).map(|t| t.inner.clone()))
            .collect::<PyResult<_>>()?;

        let tensor_refs: Vec<&Tensor> = tensor_vec.iter().collect();
        let result =
            engine::ops::shape_ops::concatenate(&tensor_refs, dim).map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Stack tensors along a new axis.
    ///
    /// `dim`, not `axis`: the free `stack` already said `dim`, so the two
    /// spellings of one operation wanted different keywords.
    #[staticmethod]
    #[pyo3(signature = (tensors, dim=0))]
    pub fn stack(tensors: &Bound<PyList>, dim: isize) -> PyResult<PyTensor> {
        if tensors.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot stack empty list of tensors",
            ));
        }

        let axis = dim;

        let unsqueezed: Vec<Tensor> = tensors
            .iter()
            .map(|obj| {
                let t = PyTensor::from_python_value(&obj)?;
                engine::ops::shape_ops::unsqueeze(&t.inner, axis).map_err(_convert_error)
            })
            .collect::<PyResult<_>>()?;

        let refs: Vec<&Tensor> = unsqueezed.iter().collect();
        let result = engine::ops::shape_ops::concatenate(&refs, axis).map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Select elements along a dimension using integer indices
    /// (a Python sequence or an integer tensor)
    pub fn index_select(&self, dim: isize, indices: &Bound<PyAny>) -> PyResult<PyTensor> {
        with_index_vector(indices, |idx| {
            engine::ops::shape_ops::index_select(&self.inner, dim, idx)
                .map(PyTensor::from_tensor)
                .map_err(_convert_error)
        })
    }

    /// Gather elements along a dimension using an index tensor
    pub fn gather(&self, dim: isize, index: &PyTensor) -> PyResult<PyTensor> {
        let result = engine::ops::shape_ops::gather(&self.inner, dim, &index.inner)
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Write ``src`` into a copy of this tensor at the positions in ``index``
    pub fn scatter(&self, dim: isize, index: &PyTensor, src: &PyTensor) -> PyResult<PyTensor> {
        let result = engine::ops::shape_ops::scatter(&self.inner, dim, &index.inner, &src.inner)
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Add ``src`` into a copy of this tensor at the positions in ``index``
    pub fn scatter_add(&self, dim: isize, index: &PyTensor, src: &PyTensor) -> PyResult<PyTensor> {
        let result =
            engine::ops::shape_ops::scatter_add(&self.inner, dim, &index.inner, &src.inner)
                .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Split tensor into multiple sub-tensors of equal size (``chunk``)
    #[pyo3(signature = (sections, dim=0))]
    pub fn chunk(&self, sections: usize, dim: isize) -> PyResult<Vec<PyTensor>> {
        if sections == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Sections must be greater than zero",
            ));
        }

        let axis = engine::ops::normalize_dim(dim, self.inner.ndim()).map_err(_convert_error)?;

        let dim_size = self.inner.shape().dims()[axis];
        if !dim_size.is_multiple_of(sections) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Tensor cannot be evenly split along the given axis",
            ));
        }

        let chunk_size = dim_size / sections;
        self.split_sections_at(&vec![chunk_size; sections], axis)
    }

    /// Split tensor by chunk size or explicit sections along an axis
    #[pyo3(signature = (split_size_or_sections, dim=0))]
    pub fn split(
        &self,
        split_size_or_sections: &Bound<PyAny>,
        dim: Option<isize>,
    ) -> PyResult<Vec<PyTensor>> {
        let dim = dim.unwrap_or(0);
        let axis = engine::ops::normalize_dim(dim, self.inner.ndim()).map_err(_convert_error)?;
        let dim_size = self.inner.shape().dims()[axis];

        let mut sections: Vec<usize> = Vec::new();

        if let Ok(split_size) = split_size_or_sections.extract::<usize>() {
            if split_size == 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "split_size must be greater than zero",
                ));
            }
            if dim_size == 0 {
                // An empty axis is one empty piece, not no pieces. The loop
                // below never runs for it, which left `cat(t.split(n, d), d)`
                // -- the round trip these two exist for -- failing on "cannot
                // concatenate empty list" instead of rebuilding the empty
                // tensor. `chunk` and `split_with_sections` both yield the
                // single empty piece here.
                sections.push(0);
            }
            let mut remaining = dim_size;
            while remaining > 0 {
                let chunk = split_size.min(remaining);
                sections.push(chunk);
                remaining -= chunk;
            }
        } else if let Ok(sizes) = split_size_or_sections.extract::<Vec<isize>>() {
            // Any sequence of sizes, rather than a list arm and a tuple arm
            // holding the same body twice. A negative entry used to reach
            // pyo3's `usize` conversion and come back as "can't convert
            // negative int to unsigned"; it is a section size like any other.
            for size in sizes {
                if size <= 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "section size must be greater than zero",
                    ));
                }
                sections.push(size as usize);
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "split_size_or_sections must be int or sequence",
            ));
        }

        self.split_sections_at(&sections, axis)
    }

    /// Split into pieces of the given sizes along `dim`.
    ///
    /// `dim`, like every other axis argument here -- the reference has always
    /// documented it that way, and this was the one signature that said
    /// `axis` without a name asking for it. It counts from the end like every
    /// other one too: taking a `usize` made this the only dim in the library
    /// that answered a negative value with "can't convert negative int to
    /// unsigned", because the type rejected it before any of the dim handling
    /// ran.
    #[pyo3(signature = (sections, dim))]
    fn split_with_sections(&self, sections: Vec<usize>, dim: isize) -> PyResult<Vec<PyTensor>> {
        let axis = engine::ops::normalize_dim(dim, self.inner.ndim()).map_err(_convert_error)?;
        self.split_sections_at(&sections, axis)
    }
}

impl PyTensor {
    /// Cut `sections` consecutive pieces out of an already-resolved axis.
    ///
    /// The three public spellings all land here, so the sizes are checked
    /// against the axis once rather than in each of them. They have to cover
    /// it exactly, which is what makes the round trip through `cat` in the
    /// reference true: a short list used to return a prefix and drop the rest
    /// without a word -- `t.split_with_sections([1], 1)` on an axis of three
    /// gave one piece -- and a long one reported the slice bounds it had
    /// computed rather than the mistake that produced them.
    fn split_sections_at(&self, sections: &[usize], axis: usize) -> PyResult<Vec<PyTensor>> {
        let dim_size = self.inner.shape().dims()[axis];
        let total: usize = sections.iter().sum();
        if total != dim_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "split sizes do not sum to dimension size: {total} against an axis of {dim_size}"
            )));
        }

        let mut outputs = Vec::with_capacity(sections.len());
        let mut start = 0;
        for size in sections {
            let end = start + size;
            let slice = engine::ops::shape_ops::slice(&self.inner, axis as isize, start, end, 1)
                .map_err(_convert_error)?;
            outputs.push(PyTensor::from_tensor(slice));
            start = end;
        }
        Ok(outputs)
    }
}
