// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[pymethods]
impl PyTensor {
    // Tensor operations
    fn clone(&self) -> PyResult<Self> {
        let result = self.inner.deep_clone().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn detach(&self) -> Self {
        Self {
            inner: self.inner.detach(),
        }
    }

    fn detach_(&mut self) {
        self.inner.detach_inplace();
    }

    fn contiguous(&self) -> PyResult<Self> {
        let result = self.inner.contiguous().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn to(&self, args: &Bound<PyTuple>, kwargs: Option<&Bound<PyDict>>) -> PyResult<Self> {
        let mut dtype_spec: Option<DataType> = None;
        let mut device_spec: Option<Device> = None;

        if let Some(mapping) = kwargs {
            for (key, value) in mapping.iter() {
                let key_string = key.str()?.to_str()?.to_owned();
                match key_string.as_str() {
                    "dtype" => {
                        if !value.is_none() {
                            dtype_spec = Some(parse_dtype_like(&value)?);
                        }
                    }
                    "device" => {
                        if !value.is_none() {
                            device_spec = Some(parse_device_like(&value)?);
                        }
                    }
                    _ => {
                        return Err(PyTypeError::new_err(format!(
                            "to() got an unexpected keyword argument '{key_string}'"
                        )));
                    }
                }
            }
        }

        if args.len() > 1 {
            return Err(PyTypeError::new_err(format!(
                "to() takes at most 1 positional argument but {} were given",
                args.len()
            )));
        }

        if args.len() == 1 {
            let arg0 = args.get_item(0)?;
            if arg0.is_none() {
                // Explicit None does nothing
            } else if let Ok(py_device) = arg0.extract::<PyDevice>() {
                if device_spec.is_some() {
                    return Err(PyTypeError::new_err(
                        "to() received multiple device specifications",
                    ));
                }
                device_spec = Some(py_device.device());
            } else if let Ok(string_value) = arg0.extract::<String>() {
                match dtype::parse_dtype(&string_value) {
                    Ok(dtype) => {
                        if let Some(existing) = dtype_spec
                            && existing != dtype
                        {
                            return Err(PyTypeError::new_err(
                                "dtype specified both positionally and via keyword",
                            ));
                        }
                        dtype_spec = Some(dtype);
                    }
                    Err(_) => {
                        let device = Device::from_str(&string_value).map_err(|err| {
                            PyValueError::new_err(format!(
                                "Unsupported device specification '{string_value}': {err}"
                            ))
                        })?;
                        if device_spec.is_some() {
                            return Err(PyTypeError::new_err(
                                "to() received multiple device specifications",
                            ));
                        }
                        device_spec = Some(device);
                    }
                }
            } else {
                return Err(PyTypeError::new_err(
                    "to() expects dtype strings, device strings, or Device objects",
                ));
            }
        }

        let mut result = self.inner.clone();
        let mut mutated = false;

        if let Some(dtype) = dtype_spec
            && result.dtype() != dtype
        {
            result = result.astype(dtype).map_err(_convert_error)?;
            mutated = true;
        }

        if let Some(device) = device_spec
            && result.device() != device
        {
            result = result.to(device).map_err(_convert_error)?;
            mutated = true;
        }

        if mutated {
            Ok(Self::from_tensor(result))
        } else {
            Ok(Self {
                inner: self.inner.clone(),
            })
        }
    }

    #[pyo3(signature = (min=None, max=None))]
    pub fn clip(&self, min: Option<&Bound<PyAny>>, max: Option<&Bound<PyAny>>) -> PyResult<Self> {
        let min_val = parse_clip_bound(min, "min")?;
        let max_val = parse_clip_bound(max, "max")?;
        let result = self.inner.clip(min_val, max_val).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (min=None, max=None))]
    pub fn clamp(&self, min: Option<&Bound<PyAny>>, max: Option<&Bound<PyAny>>) -> PyResult<Self> {
        let min_val = parse_clip_bound(min, "min")?;
        let max_val = parse_clip_bound(max, "max")?;
        let result = self.inner.clamp(min_val, max_val).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn clamp_min(&self, min: f64) -> PyResult<Self> {
        let result = self.inner.clamp_min(min).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn clamp_max(&self, max: f64) -> PyResult<Self> {
        let result = self.inner.clamp_max(max).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (decimals=0))]
    pub fn round(&self, decimals: i32) -> PyResult<Self> {
        let result = self.inner.round(decimals).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn floor(&self) -> PyResult<Self> {
        let result = self.inner.floor().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn ceil(&self) -> PyResult<Self> {
        let result = self.inner.ceil().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn sign(&self) -> PyResult<Self> {
        let result = self.inner.sign().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn reciprocal(&self) -> PyResult<Self> {
        let result = self.inner.reciprocal().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn cpu(&self) -> PyResult<Self> {
        let result = self.inner.to(Device::cpu()).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn astype(&self, dtype: &str) -> PyResult<Self> {
        let dtype = dtype::parse_dtype(dtype)?;
        let result = self.inner.astype(dtype).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

}
