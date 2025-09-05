// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    device::Device,
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

const CHUNK: usize = 1024;

#[inline(always)]
fn unary_apply_f32<F>(input: &[f32], output: &mut [f32], f: F)
where
    F: Fn(f32) -> f32 + Sync,
{
    output
        .par_chunks_mut(CHUNK)
        .zip(input.par_chunks(CHUNK))
        .for_each(|(out, inp)| unsafe {
            let in_ptr = inp.as_ptr();
            let out_ptr = out.as_mut_ptr();
            for i in 0..out.len() {
                *out_ptr.add(i) = f(*in_ptr.add(i));
            }
        });
}

#[inline(always)]
fn binary_apply_f32<F>(lhs: &[f32], rhs: &[f32], output: &mut [f32], f: F)
where
    F: Fn(f32, f32) -> f32 + Sync,
{
    output
        .par_chunks_mut(CHUNK)
        .zip(lhs.par_chunks(CHUNK).zip(rhs.par_chunks(CHUNK)))
        .for_each(|(out, (a, b))| unsafe {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let out_ptr = out.as_mut_ptr();
            for i in 0..out.len() {
                *out_ptr.add(i) = f(*a_ptr.add(i), *b_ptr.add(i));
            }
        });
}

#[inline(always)]
fn unary_apply_f64<F>(input: &[f64], output: &mut [f64], f: F)
where
    F: Fn(f64) -> f64 + Sync,
{
    output
        .par_chunks_mut(CHUNK)
        .zip(input.par_chunks(CHUNK))
        .for_each(|(out, inp)| unsafe {
            let in_ptr = inp.as_ptr();
            let out_ptr = out.as_mut_ptr();
            for i in 0..out.len() {
                *out_ptr.add(i) = f(*in_ptr.add(i));
            }
        });
}

#[inline(always)]
fn binary_apply_f64<F>(lhs: &[f64], rhs: &[f64], output: &mut [f64], f: F)
where
    F: Fn(f64, f64) -> f64 + Sync,
{
    output
        .par_chunks_mut(CHUNK)
        .zip(lhs.par_chunks(CHUNK).zip(rhs.par_chunks(CHUNK)))
        .for_each(|(out, (a, b))| unsafe {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let out_ptr = out.as_mut_ptr();
            for i in 0..out.len() {
                *out_ptr.add(i) = f(*a_ptr.add(i), *b_ptr.add(i));
            }
        });
}

/// Represents a fused operation that can combine multiple tensor operations
#[derive(Debug, Clone)]
pub enum FusedOp {
    /// Element-wise addition
    Add,
    /// Element-wise subtraction  
    Sub,
    /// Element-wise multiplication
    Mul,
    /// Element-wise division
    Div,
    /// ReLU activation
    ReLU,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Exponential
    Exp,
    /// Natural logarithm
    Log,
}

/// A sequence of operations that can be fused together
#[derive(Debug, Clone)]
pub struct FusionSequence {
    pub operations: Vec<FusedOp>,
    pub input_shapes: Vec<Shape>,
    pub output_shape: Shape,
    pub dtype: DataType,
}

impl FusionSequence {
    /// Create a new fusion sequence
    pub fn new(dtype: DataType) -> Self {
        Self {
            operations: Vec::new(),
            input_shapes: Vec::new(),
            output_shape: Shape::new(vec![]),
            dtype,
        }
    }

    /// Add an operation to the fusion sequence
    pub fn add_operation(&mut self, op: FusedOp, input_shape: Shape) -> Result<()> {
        self.operations.push(op);
        self.input_shapes.push(input_shape.clone());

        // Update output shape (for now, assume same as input for element-wise ops)
        self.output_shape = input_shape;

        Ok(())
    }

    /// Check if this sequence can be fused with another operation
    pub fn can_fuse_with(&self, op: &FusedOp, shape: &Shape) -> bool {
        // For now, only fuse element-wise operations with compatible shapes
        match op {
            FusedOp::Add
            | FusedOp::Sub
            | FusedOp::Mul
            | FusedOp::Div
            | FusedOp::ReLU
            | FusedOp::Sigmoid
            | FusedOp::Tanh
            | FusedOp::Exp
            | FusedOp::Log => {
                // Check if shapes are compatible
                self.output_shape.dims() == shape.dims() && self.operations.len() < 8
                // Limit fusion depth
            }
        }
    }

    /// Execute the fused operation sequence
    pub fn execute(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        if inputs.is_empty() {
            return Err(MinitensorError::invalid_operation(
                "No input tensors provided",
            ));
        }

        let device = inputs[0].device();
        let mut current_data = inputs[0].data().clone();

        // Execute operations in sequence
        for (i, op) in self.operations.iter().enumerate() {
            let second_input = inputs.get(i + 1).map(|t| *t);
            current_data = Arc::new(self.execute_single_op(op, &current_data, second_input)?);
        }

        Ok(Tensor::new(
            current_data,
            self.output_shape.clone(),
            self.dtype,
            device,
            false, // For now, don't track gradients in fused ops
        ))
    }

    /// Execute a single operation in the fusion sequence
    fn execute_single_op(
        &self,
        op: &FusedOp,
        input: &TensorData,
        second_input: Option<&Tensor>,
    ) -> Result<TensorData> {
        match self.dtype {
            DataType::Float32 => self.execute_f32_op(op, input, second_input),
            DataType::Float64 => self.execute_f64_op(op, input, second_input),
            _ => Err(MinitensorError::invalid_operation(
                "Unsupported data type for fusion",
            )),
        }
    }

    /// Execute f32 operations
    fn execute_f32_op(
        &self,
        op: &FusedOp,
        input: &TensorData,
        second_input: Option<&Tensor>,
    ) -> Result<TensorData> {
        let input_slice = input
            .as_f32_slice()
            .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice from input"))?;

        let mut output_data = TensorData::zeros(input_slice.len(), DataType::Float32);
        let output_slice = output_data.as_f32_slice_mut().ok_or_else(|| {
            MinitensorError::internal_error("Failed to get mutable f32 slice from output")
        })?;

        match op {
            FusedOp::Add => {
                if let Some(second) = second_input {
                    let second_slice = second.data().as_f32_slice().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get f32 slice from second input")
                    })?;
                    binary_apply_f32(input_slice, second_slice, output_slice, |a, b| a + b);
                } else {
                    return Err(MinitensorError::invalid_operation(
                        "Add operation requires two inputs",
                    ));
                }
            }
            FusedOp::Sub => {
                if let Some(second) = second_input {
                    let second_slice = second.data().as_f32_slice().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get f32 slice from second input")
                    })?;
                    binary_apply_f32(input_slice, second_slice, output_slice, |a, b| a - b);
                } else {
                    return Err(MinitensorError::invalid_operation(
                        "Sub operation requires two inputs",
                    ));
                }
            }
            FusedOp::Mul => {
                if let Some(second) = second_input {
                    let second_slice = second.data().as_f32_slice().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get f32 slice from second input")
                    })?;
                    binary_apply_f32(input_slice, second_slice, output_slice, |a, b| a * b);
                } else {
                    return Err(MinitensorError::invalid_operation(
                        "Mul operation requires two inputs",
                    ));
                }
            }
            FusedOp::Div => {
                if let Some(second) = second_input {
                    let second_slice = second.data().as_f32_slice().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get f32 slice from second input")
                    })?;
                    binary_apply_f32(input_slice, second_slice, output_slice, |a, b| {
                        if b == 0.0 {
                            f32::INFINITY
                        } else {
                            a / b
                        }
                    });
                } else {
                    return Err(MinitensorError::invalid_operation(
                        "Div operation requires two inputs",
                    ));
                }
            }
            FusedOp::ReLU => {
                unary_apply_f32(input_slice, output_slice, |x| x.max(0.0));
            }
            FusedOp::Sigmoid => {
                unary_apply_f32(input_slice, output_slice, |x| 1.0 / (1.0 + (-x).exp()));
            }
            FusedOp::Tanh => {
                unary_apply_f32(input_slice, output_slice, |x| x.tanh());
            }
            FusedOp::Exp => {
                unary_apply_f32(input_slice, output_slice, |x| x.exp());
            }
            FusedOp::Log => {
                unary_apply_f32(input_slice, output_slice, |x| x.ln());
            }
        }

        Ok(output_data)
    }

    /// Execute f64 operations
    fn execute_f64_op(
        &self,
        op: &FusedOp,
        input: &TensorData,
        second_input: Option<&Tensor>,
    ) -> Result<TensorData> {
        let input_slice = input
            .as_f64_slice()
            .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice from input"))?;

        let mut output_data = TensorData::zeros(input_slice.len(), DataType::Float64);
        let output_slice = output_data.as_f64_slice_mut().ok_or_else(|| {
            MinitensorError::internal_error("Failed to get mutable f64 slice from output")
        })?;

        match op {
            FusedOp::Add => {
                if let Some(second) = second_input {
                    let second_slice = second.data().as_f64_slice().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get f64 slice from second input")
                    })?;
                    binary_apply_f64(input_slice, second_slice, output_slice, |a, b| a + b);
                } else {
                    return Err(MinitensorError::invalid_operation(
                        "Add operation requires two inputs",
                    ));
                }
            }
            FusedOp::Sub => {
                if let Some(second) = second_input {
                    let second_slice = second.data().as_f64_slice().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get f64 slice from second input")
                    })?;
                    binary_apply_f64(input_slice, second_slice, output_slice, |a, b| a - b);
                } else {
                    return Err(MinitensorError::invalid_operation(
                        "Sub operation requires two inputs",
                    ));
                }
            }
            FusedOp::Mul => {
                if let Some(second) = second_input {
                    let second_slice = second.data().as_f64_slice().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get f64 slice from second input")
                    })?;
                    binary_apply_f64(input_slice, second_slice, output_slice, |a, b| a * b);
                } else {
                    return Err(MinitensorError::invalid_operation(
                        "Mul operation requires two inputs",
                    ));
                }
            }
            FusedOp::Div => {
                if let Some(second) = second_input {
                    let second_slice = second.data().as_f64_slice().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get f64 slice from second input")
                    })?;
                    binary_apply_f64(input_slice, second_slice, output_slice, |a, b| {
                        if b == 0.0 {
                            f64::INFINITY
                        } else {
                            a / b
                        }
                    });
                } else {
                    return Err(MinitensorError::invalid_operation(
                        "Div operation requires two inputs",
                    ));
                }
            }
            FusedOp::ReLU => {
                unary_apply_f64(input_slice, output_slice, |x| x.max(0.0));
            }
            FusedOp::Sigmoid => {
                unary_apply_f64(input_slice, output_slice, |x| 1.0 / (1.0 + (-x).exp()));
            }
            FusedOp::Tanh => {
                unary_apply_f64(input_slice, output_slice, |x| x.tanh());
            }
            FusedOp::Exp => {
                unary_apply_f64(input_slice, output_slice, |x| x.exp());
            }
            FusedOp::Log => {
                unary_apply_f64(input_slice, output_slice, |x| x.ln());
            }
        }

        Ok(output_data)
    }
}

/// Memory pool for reusing tensor allocations
pub struct MemoryPool {
    pools: HashMap<(DataType, Device), VecDeque<TensorData>>,
    max_pool_size: usize,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pools: HashMap::new(),
            max_pool_size,
        }
    }

    /// Get a tensor data from the pool or allocate a new one
    pub fn get_or_allocate(&mut self, size: usize, dtype: DataType, device: Device) -> TensorData {
        let key = (dtype, device);

        if let Some(pool) = self.pools.get_mut(&key) {
            // Try to find a suitable tensor in the pool
            for i in 0..pool.len() {
                if pool[i].len() >= size {
                    let mut data = pool.remove(i).unwrap();
                    // Resize if necessary (this is a no-op if size matches)
                    if data.len() != size {
                        data = TensorData::zeros(size, dtype);
                    }
                    return data;
                }
            }
        }

        // No suitable tensor found, allocate a new one
        TensorData::zeros_on_device(size, dtype, device)
    }

    /// Return a tensor data to the pool for reuse
    pub fn return_to_pool(&mut self, data: TensorData, dtype: DataType, device: Device) {
        let key = (dtype, device);

        let pool = self.pools.entry(key).or_insert_with(VecDeque::new);

        // Only keep the tensor if the pool isn't full
        if pool.len() < self.max_pool_size {
            pool.push_back(data);
        }
        // Otherwise, let the tensor be dropped and deallocated
    }

    /// Clear all pools
    pub fn clear(&mut self) {
        self.pools.clear();
    }

    /// Get statistics about the memory pool
    pub fn stats(&self) -> MemoryPoolStats {
        let mut total_tensors = 0;
        let mut total_memory = 0;

        for pool in self.pools.values() {
            total_tensors += pool.len();
            for data in pool {
                total_memory += data.len() * data.dtype().size_in_bytes();
            }
        }

        MemoryPoolStats {
            total_tensors,
            total_memory_bytes: total_memory,
            pool_count: self.pools.len(),
        }
    }
}

/// Statistics about memory pool usage
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub total_tensors: usize,
    pub total_memory_bytes: usize,
    pub pool_count: usize,
}

/// Global memory pool instance
static GLOBAL_MEMORY_POOL: Mutex<Option<MemoryPool>> = Mutex::new(None);

/// Initialize the global memory pool
pub fn init_memory_pool(max_pool_size: usize) {
    let mut pool = GLOBAL_MEMORY_POOL.lock().unwrap();
    *pool = Some(MemoryPool::new(max_pool_size));
}

/// Get a tensor from the global memory pool
pub fn get_pooled_tensor(size: usize, dtype: DataType, device: Device) -> TensorData {
    let mut pool_guard = GLOBAL_MEMORY_POOL.lock().unwrap();
    if let Some(ref mut pool) = *pool_guard {
        pool.get_or_allocate(size, dtype, device)
    } else {
        // Pool not initialized, allocate directly
        TensorData::zeros_on_device(size, dtype, device)
    }
}

/// Return a tensor to the global memory pool
pub fn return_pooled_tensor(data: TensorData, dtype: DataType, device: Device) {
    let mut pool_guard = GLOBAL_MEMORY_POOL.lock().unwrap();
    if let Some(ref mut pool) = *pool_guard {
        pool.return_to_pool(data, dtype, device);
    }
    // If pool not initialized, just let the tensor be dropped
}

/// Get memory pool statistics
pub fn memory_pool_stats() -> Option<MemoryPoolStats> {
    let pool_guard = GLOBAL_MEMORY_POOL.lock().unwrap();
    pool_guard.as_ref().map(|pool| pool.stats())
}

/// Lazy evaluation system for deferred computation
pub struct LazyTensor {
    pub operation: FusedOp,
    pub inputs: Vec<Arc<LazyTensor>>,
    pub shape: Shape,
    pub dtype: DataType,
    pub device: Device,
    pub computed_value: Option<Arc<TensorData>>,
}

impl LazyTensor {
    /// Create a new lazy tensor from a concrete value
    pub fn from_tensor(tensor: &Tensor) -> Self {
        Self {
            operation: FusedOp::Add, // Placeholder for leaf nodes
            inputs: Vec::new(),
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            device: tensor.device(),
            computed_value: Some(tensor.data().clone()),
        }
    }

    /// Create a new lazy tensor from an operation
    pub fn from_operation(
        operation: FusedOp,
        inputs: Vec<Arc<LazyTensor>>,
        shape: Shape,
        dtype: DataType,
        device: Device,
    ) -> Self {
        Self {
            operation,
            inputs,
            shape,
            dtype,
            device,
            computed_value: None,
        }
    }

    /// Compute the value of this lazy tensor
    pub fn compute(&mut self) -> Result<Arc<TensorData>> {
        if let Some(ref value) = self.computed_value {
            return Ok(value.clone());
        }

        // Compute input values first
        let mut input_values = Vec::new();
        for input in &mut self.inputs {
            // This is a simplified approach - in practice, you'd need interior mutability
            // or a different design to handle mutable references properly
            input_values.push(
                input
                    .computed_value
                    .clone()
                    .ok_or_else(|| MinitensorError::internal_error("Input value not computed"))?,
            );
        }

        // Create a fusion sequence and execute it
        let mut sequence = FusionSequence::new(self.dtype);
        sequence.add_operation(self.operation.clone(), self.shape.clone())?;

        // For now, this is a simplified implementation
        // In practice, you'd need to handle the conversion from TensorData to Tensor properly
        let result_data = get_pooled_tensor(self.shape.numel(), self.dtype, self.device);
        let result = Arc::new(result_data);

        self.computed_value = Some(result.clone());
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    #[test]
    fn test_fusion_sequence_creation() {
        let mut sequence = FusionSequence::new(DataType::Float32);
        let shape = Shape::new(vec![4]);

        sequence.add_operation(FusedOp::Add, shape.clone()).unwrap();
        sequence.add_operation(FusedOp::ReLU, shape).unwrap();

        assert_eq!(sequence.operations.len(), 2);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(5);

        // Allocate some tensors
        let data1 = pool.get_or_allocate(100, DataType::Float32, Device::cpu());
        let data2 = pool.get_or_allocate(200, DataType::Float32, Device::cpu());

        // Return them to pool
        pool.return_to_pool(data1, DataType::Float32, Device::cpu());
        pool.return_to_pool(data2, DataType::Float32, Device::cpu());

        let stats = pool.stats();
        assert_eq!(stats.total_tensors, 2);
    }

    #[test]
    fn test_can_fuse_operations() {
        let mut sequence = FusionSequence::new(DataType::Float32);
        let shape = Shape::new(vec![4]);

        sequence.add_operation(FusedOp::Add, shape.clone()).unwrap();

        assert!(sequence.can_fuse_with(&FusedOp::ReLU, &shape));
        assert!(sequence.can_fuse_with(&FusedOp::Sigmoid, &shape));
    }

    #[test]
    fn test_global_memory_pool() {
        init_memory_pool(10);

        let data = get_pooled_tensor(50, DataType::Float32, Device::cpu());
        return_pooled_tensor(data, DataType::Float32, Device::cpu());

        if let Some(stats) = memory_pool_stats() {
            assert!(stats.total_tensors > 0);
        }
    }
}
