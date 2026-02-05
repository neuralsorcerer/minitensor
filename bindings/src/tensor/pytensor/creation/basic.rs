// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[pymethods]
impl PyTensor {
    // Static tensor creation methods
    #[staticmethod]
    #[pyo3(signature = (*shape, dtype=None, device=None, requires_grad=false))]
    pub fn empty(
        shape: &Bound<PyTuple>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = Tensor::empty(shape, dtype, device, requires_grad);
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (*shape, dtype=None, device=None, requires_grad=false))]
    pub fn zeros(
        shape: &Bound<PyTuple>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = Tensor::zeros(shape, dtype, device, requires_grad);
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (*shape, dtype=None, device=None, requires_grad=false))]
    pub fn ones(
        shape: &Bound<PyTuple>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = Tensor::ones(shape, dtype, device, requires_grad);
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (*shape, low=0.0, high=1.0, dtype=None, device=None, requires_grad=false))]
    fn uniform(
        shape: &Bound<PyTuple>,
        low: f64,
        high: f64,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = create_uniform_tensor(shape, dtype, device, requires_grad, low, high)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (*shape, dtype=None, device=None, requires_grad=false))]
    fn xavier_uniform(
        shape: &Bound<PyTuple>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = create_fan_init_tensor(
            shape,
            dtype,
            device,
            requires_grad,
            FanInitKind::XavierUniform,
            "xavier_uniform",
        )?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (*shape, dtype=None, device=None, requires_grad=false))]
    fn xavier_normal(
        shape: &Bound<PyTuple>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = create_fan_init_tensor(
            shape,
            dtype,
            device,
            requires_grad,
            FanInitKind::XavierNormal,
            "xavier_normal",
        )?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (*shape, dtype=None, device=None, requires_grad=false))]
    fn he_uniform(
        shape: &Bound<PyTuple>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = create_fan_init_tensor(
            shape,
            dtype,
            device,
            requires_grad,
            FanInitKind::HeUniform,
            "he_uniform",
        )?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (*shape, dtype=None, device=None, requires_grad=false))]
    fn he_normal(
        shape: &Bound<PyTuple>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = create_fan_init_tensor(
            shape,
            dtype,
            device,
            requires_grad,
            FanInitKind::HeNormal,
            "he_normal",
        )?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (*shape, dtype=None, device=None, requires_grad=false))]
    fn lecun_uniform(
        shape: &Bound<PyTuple>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = create_fan_init_tensor(
            shape,
            dtype,
            device,
            requires_grad,
            FanInitKind::LecunUniform,
            "lecun_uniform",
        )?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (*shape, dtype=None, device=None, requires_grad=false))]
    fn lecun_normal(
        shape: &Bound<PyTuple>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = create_fan_init_tensor(
            shape,
            dtype,
            device,
            requires_grad,
            FanInitKind::LecunNormal,
            "lecun_normal",
        )?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (*shape, dtype=None, device=None, requires_grad=false))]
    fn rand(
        shape: &Bound<PyTuple>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = create_random_tensor(shape, dtype, device, requires_grad, false)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (*shape, dtype=None, device=None, requires_grad=false))]
    fn randn(
        shape: &Bound<PyTuple>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = create_random_tensor(shape, dtype, device, requires_grad, true)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (*shape, mean=0.0, std=1.0, lower=None, upper=None, dtype=None, device=None, requires_grad=false))]
    #[allow(clippy::too_many_arguments)]
    fn truncated_normal(
        shape: &Bound<PyTuple>,
        mean: f64,
        std: f64,
        lower: Option<f64>,
        upper: Option<f64>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dims = parse_shape_tuple(shape, "shape")?;
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let shape = Shape::new(dims);
        let tensor = create_truncated_normal_tensor(
            shape,
            dtype,
            device,
            requires_grad,
            mean,
            std,
            lower,
            upper,
            "truncated_normal",
        )?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (input, low=0.0, high=1.0, dtype=None, device=None, requires_grad=None))]
    fn uniform_like(
        input: &Bound<PyAny>,
        low: f64,
        high: f64,
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
        let tensor = create_uniform_tensor(shape, dtype, device, requires_grad, low, high)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (input, dtype=None, device=None, requires_grad=None))]
    fn xavier_uniform_like(
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
        let tensor = create_fan_init_tensor(
            shape,
            dtype,
            device,
            requires_grad,
            FanInitKind::XavierUniform,
            "xavier_uniform_like",
        )?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (input, dtype=None, device=None, requires_grad=None))]
    fn xavier_normal_like(
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
        let tensor = create_fan_init_tensor(
            shape,
            dtype,
            device,
            requires_grad,
            FanInitKind::XavierNormal,
            "xavier_normal_like",
        )?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (input, dtype=None, device=None, requires_grad=None))]
    fn he_uniform_like(
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
        let tensor = create_fan_init_tensor(
            shape,
            dtype,
            device,
            requires_grad,
            FanInitKind::HeUniform,
            "he_uniform_like",
        )?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (input, dtype=None, device=None, requires_grad=None))]
    fn he_normal_like(
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
        let tensor = create_fan_init_tensor(
            shape,
            dtype,
            device,
            requires_grad,
            FanInitKind::HeNormal,
            "he_normal_like",
        )?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (input, dtype=None, device=None, requires_grad=None))]
    fn lecun_uniform_like(
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
        let tensor = create_fan_init_tensor(
            shape,
            dtype,
            device,
            requires_grad,
            FanInitKind::LecunUniform,
            "lecun_uniform_like",
        )?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (input, dtype=None, device=None, requires_grad=None))]
    fn lecun_normal_like(
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
        let tensor = create_fan_init_tensor(
            shape,
            dtype,
            device,
            requires_grad,
            FanInitKind::LecunNormal,
            "lecun_normal_like",
        )?;
        Ok(Self::from_tensor(tensor))
    }

}
