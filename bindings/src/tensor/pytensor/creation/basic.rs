// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
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
}

/// The fan-in/fan-out initialisers differ only in which [`FanInitKind`] they
/// pass through; every one of them otherwise repeated the same argument
/// resolution and the same call, once for the `*shape` form and once for the
/// `_like` form. Twelve constructors were twelve verbatim copies of two bodies.
///
/// The macro emits the whole `#[pymethods]` block rather than individual
/// methods: pyo3's attribute macro reads the impl block's tokens, so a
/// `macro_rules!` call sitting *inside* one would never be expanded in time.
macro_rules! fan_init_constructors {
    ($(($name:ident, $like:ident, $kind:ident)),* $(,)?) => {
        #[pymethods]
        impl PyTensor {
            $(
                #[staticmethod]
                #[pyo3(signature = (*shape, dtype=None, device=None, requires_grad=false))]
                fn $name(
                    shape: &Bound<PyTuple>,
                    dtype: Option<&str>,
                    device: Option<&PyDevice>,
                    requires_grad: Option<bool>,
                ) -> PyResult<Self> {
                    let dims = parse_shape_tuple(shape, "shape")?;
                    let dtype = dtype::resolve_dtype_arg(dtype)?;
                    let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
                    let tensor = create_fan_init_tensor(
                        Shape::new(dims),
                        dtype,
                        device,
                        requires_grad.unwrap_or(false),
                        FanInitKind::$kind,
                        stringify!($name),
                    )?;
                    Ok(Self::from_tensor(tensor))
                }

                #[staticmethod]
                #[pyo3(signature = (input, dtype=None, device=None, requires_grad=None))]
                fn $like(
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
                    let requires_grad =
                        requires_grad.unwrap_or(reference_tensor.requires_grad());
                    let tensor = create_fan_init_tensor(
                        Shape::new(reference.shape_vec()),
                        dtype,
                        device,
                        requires_grad,
                        FanInitKind::$kind,
                        stringify!($like),
                    )?;
                    Ok(Self::from_tensor(tensor))
                }
            )*
        }
    };
}

fan_init_constructors!(
    (xavier_uniform, xavier_uniform_like, XavierUniform),
    (xavier_normal, xavier_normal_like, XavierNormal),
    (he_uniform, he_uniform_like, HeUniform),
    (he_normal, he_normal_like, HeNormal),
    (lecun_uniform, lecun_uniform_like, LecunUniform),
    (lecun_normal, lecun_normal_like, LecunNormal),
);
