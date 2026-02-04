// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[pymethods]
impl PyTensor {
    #[staticmethod]
    #[pyo3(signature = (input, dtype=None, device=None, requires_grad=None))]
    fn rand_like(
        input: &Bound<PyAny>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let reference = PyTensor::from_python_value(input)?;
        let reference_tensor = reference.tensor();

        let dtype = match dtype {
            Some(name) => dtype::parse_dtype(name)?,
            None => match reference_tensor.dtype() {
                DataType::Float32 | DataType::Float64 => reference_tensor.dtype(),
                _ => dtype::default_float_dtype(),
            },
        };

        match dtype {
            DataType::Float32 | DataType::Float64 => {}
            _ => {
                return Err(PyValueError::new_err(
                    "rand_like only supports float32 or float64 dtypes",
                ));
            }
        }

        let device = device
            .map(|d| d.device())
            .unwrap_or_else(|| reference_tensor.device());
        let requires_grad = requires_grad.unwrap_or(reference_tensor.requires_grad());
        let shape = Shape::new(reference.shape_vec());
        let tensor = create_random_tensor(shape, dtype, device, requires_grad, false)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (input, dtype=None, device=None, requires_grad=None))]
    fn randn_like(
        input: &Bound<PyAny>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let reference = PyTensor::from_python_value(input)?;
        let reference_tensor = reference.tensor();

        let dtype = match dtype {
            Some(name) => dtype::parse_dtype(name)?,
            None => match reference_tensor.dtype() {
                DataType::Float32 | DataType::Float64 => reference_tensor.dtype(),
                _ => dtype::default_float_dtype(),
            },
        };

        match dtype {
            DataType::Float32 | DataType::Float64 => {}
            _ => {
                return Err(PyValueError::new_err(
                    "randn_like only supports float32 or float64 dtypes",
                ));
            }
        }

        let device = device
            .map(|d| d.device())
            .unwrap_or_else(|| reference_tensor.device());
        let requires_grad = requires_grad.unwrap_or(reference_tensor.requires_grad());
        let shape = Shape::new(reference.shape_vec());
        let tensor = create_random_tensor(shape, dtype, device, requires_grad, true)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (input, mean=0.0, std=1.0, lower=None, upper=None, dtype=None, device=None, requires_grad=None))]
    #[allow(clippy::too_many_arguments)]
    fn truncated_normal_like(
        input: &Bound<PyAny>,
        mean: f64,
        std: f64,
        lower: Option<f64>,
        upper: Option<f64>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let reference = PyTensor::from_python_value(input)?;
        let reference_tensor = reference.tensor();

        let dtype = match dtype {
            Some(name) => dtype::parse_dtype(name)?,
            None => match reference_tensor.dtype() {
                DataType::Float32 | DataType::Float64 => reference_tensor.dtype(),
                _ => dtype::default_float_dtype(),
            },
        };

        let device = device
            .map(|d| d.device())
            .unwrap_or_else(|| reference_tensor.device());
        let requires_grad = requires_grad.unwrap_or(reference_tensor.requires_grad());
        let shape = Shape::new(reference.shape_vec());
        let tensor = create_truncated_normal_tensor(
            shape,
            dtype,
            device,
            requires_grad,
            mean,
            std,
            lower,
            upper,
            "truncated_normal_like",
        )?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (input, dtype=None, device=None, requires_grad=None))]
    fn empty_like(
        input: &Bound<PyAny>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let reference = PyTensor::from_python_value(input)?;
        let reference_tensor = reference.tensor();

        let dtype = match dtype {
            Some(name) => dtype::parse_dtype(name)?,
            None => reference_tensor.dtype(),
        };
        let device = device
            .map(|d| d.device())
            .unwrap_or_else(|| reference_tensor.device());
        let requires_grad = requires_grad.unwrap_or(reference_tensor.requires_grad());
        let shape = Shape::new(reference.shape_vec());
        let tensor = Tensor::empty(shape, dtype, device, requires_grad);
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (input, dtype=None, device=None, requires_grad=None))]
    fn zeros_like(
        input: &Bound<PyAny>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let reference = PyTensor::from_python_value(input)?;
        let reference_tensor = reference.tensor();

        let dtype = match dtype {
            Some(name) => dtype::parse_dtype(name)?,
            None => reference_tensor.dtype(),
        };
        let device = device
            .map(|d| d.device())
            .unwrap_or_else(|| reference_tensor.device());
        let requires_grad = requires_grad.unwrap_or(reference_tensor.requires_grad());
        let shape = Shape::new(reference.shape_vec());
        let tensor = Tensor::zeros(shape, dtype, device, requires_grad);
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (input, dtype=None, device=None, requires_grad=None))]
    fn ones_like(
        input: &Bound<PyAny>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let reference = PyTensor::from_python_value(input)?;
        let reference_tensor = reference.tensor();

        let dtype = match dtype {
            Some(name) => dtype::parse_dtype(name)?,
            None => reference_tensor.dtype(),
        };
        let device = device
            .map(|d| d.device())
            .unwrap_or_else(|| reference_tensor.device());
        let requires_grad = requires_grad.unwrap_or(reference_tensor.requires_grad());
        let shape = Shape::new(reference.shape_vec());
        let tensor = Tensor::ones(shape, dtype, device, requires_grad);
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (input, fill_value, dtype=None, device=None, requires_grad=None))]
    fn full_like(
        input: &Bound<PyAny>,
        fill_value: f64,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let reference = PyTensor::from_python_value(input)?;
        let reference_tensor = reference.tensor();

        let dtype = match dtype {
            Some(name) => dtype::parse_dtype(name)?,
            None => reference_tensor.dtype(),
        };

        let device = device
            .map(|d| d.device())
            .unwrap_or_else(|| reference_tensor.device());
        let requires_grad = requires_grad.unwrap_or(reference_tensor.requires_grad());
        let shape = reference.shape_vec();
        let tensor = create_full_tensor(shape, fill_value, dtype, device, requires_grad)?;
        Ok(Self::from_tensor(tensor))
    }

    #[pyo3(signature = (shape, dtype=None, device=None, requires_grad=None))]
    fn new_empty(
        &self,
        shape: &Bound<PyAny>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_like(shape, "shape")?;
        let dtype = match dtype {
            Some(name) => dtype::parse_dtype(name)?,
            None => self.inner.dtype(),
        };
        let device = device
            .map(|d| d.device())
            .unwrap_or_else(|| self.inner.device());
        let requires_grad = requires_grad.unwrap_or(self.inner.requires_grad());
        let tensor = Tensor::empty(Shape::new(dims), dtype, device, requires_grad);
        Ok(Self::from_tensor(tensor))
    }

    #[pyo3(signature = (shape, dtype=None, device=None, requires_grad=None))]
    fn new_zeros(
        &self,
        shape: &Bound<PyAny>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_like(shape, "shape")?;
        let dtype = match dtype {
            Some(name) => dtype::parse_dtype(name)?,
            None => self.inner.dtype(),
        };
        let device = device
            .map(|d| d.device())
            .unwrap_or_else(|| self.inner.device());
        let requires_grad = requires_grad.unwrap_or(self.inner.requires_grad());
        let tensor = Tensor::zeros(Shape::new(dims), dtype, device, requires_grad);
        Ok(Self::from_tensor(tensor))
    }

    #[pyo3(signature = (shape, dtype=None, device=None, requires_grad=None))]
    fn new_ones(
        &self,
        shape: &Bound<PyAny>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_like(shape, "shape")?;
        let dtype = match dtype {
            Some(name) => dtype::parse_dtype(name)?,
            None => self.inner.dtype(),
        };
        let device = device
            .map(|d| d.device())
            .unwrap_or_else(|| self.inner.device());
        let requires_grad = requires_grad.unwrap_or(self.inner.requires_grad());
        let tensor = Tensor::ones(Shape::new(dims), dtype, device, requires_grad);
        Ok(Self::from_tensor(tensor))
    }

    #[pyo3(signature = (shape, fill_value, dtype=None, device=None, requires_grad=None))]
    fn new_full(
        &self,
        shape: &Bound<PyAny>,
        fill_value: f64,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_like(shape, "shape")?;
        let dtype = match dtype {
            Some(name) => dtype::parse_dtype(name)?,
            None => self.inner.dtype(),
        };
        let device = device
            .map(|d| d.device())
            .unwrap_or_else(|| self.inner.device());
        let requires_grad = requires_grad.unwrap_or(self.inner.requires_grad());
        let tensor = create_full_tensor(dims, fill_value, dtype, device, requires_grad)?;
        Ok(Self::from_tensor(tensor))
    }

    #[pyo3(signature = (data, dtype=None, device=None, requires_grad=None))]
    fn new_tensor(
        &self,
        data: &Bound<PyAny>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dtype = match dtype {
            Some(name) => dtype::parse_dtype(name)?,
            None => self.inner.dtype(),
        };
        let device = device
            .map(|d| d.device())
            .unwrap_or_else(|| self.inner.device());
        let requires_grad = requires_grad.unwrap_or(self.inner.requires_grad());

        if let Ok(py_tensor) = data.extract::<PyRef<PyTensor>>() {
            let tensor =
                prepare_new_tensor_from_existing(py_tensor.tensor(), dtype, device, requires_grad)?;
            return Ok(Self::from_tensor(tensor));
        }

        if let Ok(inner_attr) = data.getattr(intern!(data.py(), "_tensor"))
            && let Ok(py_tensor) = inner_attr.extract::<PyRef<PyTensor>>()
        {
            let tensor =
                prepare_new_tensor_from_existing(py_tensor.tensor(), dtype, device, requires_grad)?;
            return Ok(Self::from_tensor(tensor));
        }

        let tensor = convert_python_data_to_tensor(data, dtype, device, requires_grad)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (input, low, high=None, dtype=None, device=None, requires_grad=None))]
    fn randint_like(
        input: &Bound<PyAny>,
        low: i64,
        high: Option<i64>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let reference = PyTensor::from_python_value(input)?;
        let reference_tensor = reference.tensor();

        let (low, high) = match high {
            Some(high) => (low, high),
            None => (0, low),
        };

        if low >= high {
            return Err(PyValueError::new_err(
                "randint_like requires that low < high",
            ));
        }

        let dtype = match dtype {
            Some(name) => dtype::parse_dtype(name)?,
            None => match reference_tensor.dtype() {
                DataType::Int32 => DataType::Int32,
                DataType::Int64 => DataType::Int64,
                _ => DataType::Int64,
            },
        };

        match dtype {
            DataType::Int32 | DataType::Int64 => {}
            _ => {
                return Err(PyValueError::new_err(
                    "randint_like only supports int32 or int64 dtypes",
                ));
            }
        }

        let device = device
            .map(|d| d.device())
            .unwrap_or_else(|| reference_tensor.device());
        let requires_grad = requires_grad.unwrap_or(reference_tensor.requires_grad());
        let shape = Shape::new(reference.shape_vec());
        let tensor = create_randint_tensor(shape, dtype, device, requires_grad, low, high)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (low, high=None, *shape, dtype=None, device=None, requires_grad=false))]
    fn randint(
        low: i64,
        high: Option<i64>,
        shape: &Bound<PyTuple>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let (low, high) = match high {
            Some(high) => (low, high),
            None => (0, low),
        };

        if low >= high {
            return Err(PyValueError::new_err("randint requires that low < high"));
        }

        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = match dtype {
            Some(name) => dtype::parse_dtype(name)?,
            None => DataType::Int64,
        };

        match dtype {
            DataType::Int32 | DataType::Int64 => {}
            _ => {
                return Err(PyValueError::new_err(
                    "randint only supports int32 or int64 dtypes",
                ));
            }
        }

        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = create_randint_tensor(shape, dtype, device, requires_grad, low, high)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (n, dtype=None, device=None, requires_grad=false))]
    fn randperm(
        n: usize,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dtype = match dtype {
            Some(name) => dtype::parse_dtype(name)?,
            None => DataType::Int64,
        };

        match dtype {
            DataType::Int32 | DataType::Int64 => {}
            _ => {
                return Err(PyValueError::new_err(
                    "randperm only supports int32 or int64 dtypes",
                ));
            }
        }

        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let tensor = create_randperm_tensor(n, dtype, device, requires_grad)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (n, m=None, dtype=None, device=None, requires_grad=false))]
    fn eye(
        n: usize,
        m: Option<usize>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let m = m.unwrap_or(n);
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let tensor = create_eye_tensor(n, m, dtype, device, requires_grad)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (shape, fill_value, dtype=None, device=None, requires_grad=false))]
    pub fn full(
        shape: &Bound<PyAny>,
        fill_value: f64,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let dims = parse_shape_like(shape, "shape")?;
        let tensor = create_full_tensor(dims, fill_value, dtype, device, requires_grad)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (data, dtype=None, device=None, requires_grad=None, copy=false))]
    fn as_tensor(
        data: &Bound<PyAny>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
        copy: Option<bool>,
    ) -> PyResult<Self> {
        let copy = copy.unwrap_or(false);

        if let Ok(py_tensor) = data.extract::<PyRef<PyTensor>>() {
            let source = py_tensor.tensor();
            let target_dtype = match dtype {
                Some(name) => dtype::parse_dtype(name)?,
                None => source.dtype(),
            };
            let target_device = device
                .map(|d| d.device())
                .unwrap_or_else(|| source.device());
            let target_requires_grad = requires_grad.unwrap_or(source.requires_grad());
            let tensor = adapt_tensor_for_as_tensor(
                source,
                target_dtype,
                target_device,
                target_requires_grad,
                copy,
            )?;
            return Ok(Self::from_tensor(tensor));
        }

        if let Ok(inner_attr) = data.getattr(intern!(data.py(), "_tensor"))
            && let Ok(py_tensor) = inner_attr.extract::<PyRef<PyTensor>>()
        {
            let source = py_tensor.tensor();
            let target_dtype = match dtype {
                Some(name) => dtype::parse_dtype(name)?,
                None => source.dtype(),
            };
            let target_device = device
                .map(|d| d.device())
                .unwrap_or_else(|| source.device());
            let target_requires_grad = requires_grad.unwrap_or(source.requires_grad());
            let tensor = adapt_tensor_for_as_tensor(
                source,
                target_dtype,
                target_device,
                target_requires_grad,
                copy,
            )?;
            return Ok(Self::from_tensor(tensor));
        }

        let target_dtype = match dtype {
            Some(name) => dtype::parse_dtype(name)?,
            None => infer_python_value_dtype(data).unwrap_or_else(dtype::default_dtype),
        };

        let target_device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let target_requires_grad = requires_grad.unwrap_or(false);

        let tensor =
            convert_python_data_to_tensor(data, target_dtype, target_device, target_requires_grad)?;
        Ok(Self::from_tensor(tensor))
    }

}
