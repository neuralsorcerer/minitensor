// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
#[pymethods]
impl PyTensor {
    /// Concatenate tensors along an axis
    #[staticmethod]
    #[pyo3(signature = (tensors, axis=None))]
    pub fn concatenate(tensors: &Bound<PyList>, axis: Option<isize>) -> PyResult<PyTensor> {
        if tensors.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot concatenate empty list of tensors",
            ));
        }

        let axis = axis.unwrap_or(0);

        let tensor_vec: Vec<Tensor> = tensors
            .iter()
            .map(|obj| PyTensor::from_python_value(&obj).map(|t| t.inner.clone()))
            .collect::<PyResult<_>>()?;

        let tensor_refs: Vec<&Tensor> = tensor_vec.iter().collect();
        let result =
            engine::ops::shape_ops::concatenate(&tensor_refs, axis).map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Stack tensors along a new axis
    #[staticmethod]
    #[pyo3(signature = (tensors, axis=None))]
    pub fn stack(tensors: &Bound<PyList>, axis: Option<isize>) -> PyResult<PyTensor> {
        if tensors.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot stack empty list of tensors",
            ));
        }

        let axis = axis.unwrap_or(0);

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
        let idx_vec = extract_index_vector(indices)?;
        let result = engine::ops::shape_ops::index_select(&self.inner, dim, &idx_vec)
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(result))
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
        let section_vec = vec![chunk_size; sections];
        self.split_with_sections(section_vec, axis)
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
        } else if let Ok(list) = split_size_or_sections.cast::<PyList>() {
            for obj in list.iter() {
                let size: usize = obj.extract()?;
                if size == 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "section size must be greater than zero",
                    ));
                }
                sections.push(size);
            }
            let total: usize = sections.iter().sum();
            if total != dim_size {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "split sizes do not sum to dimension size",
                ));
            }
        } else if let Ok(tuple) = split_size_or_sections.cast::<PyTuple>() {
            for obj in tuple.iter() {
                let size: usize = obj.extract()?;
                if size == 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "section size must be greater than zero",
                    ));
                }
                sections.push(size);
            }
            let total: usize = sections.iter().sum();
            if total != dim_size {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "split sizes do not sum to dimension size",
                ));
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "split_size_or_sections must be int or sequence",
            ));
        }

        self.split_with_sections(sections, axis)
    }

    fn split_with_sections(&self, sections: Vec<usize>, axis: usize) -> PyResult<Vec<PyTensor>> {
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
