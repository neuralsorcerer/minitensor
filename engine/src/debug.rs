// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    tensor::{Tensor, Shape},
    device::Device,
};
use std::collections::HashMap;

/// Tensor inspection information
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shape: Shape,
    pub dtype: String,
    pub device: Device,
    pub numel: usize,
    pub requires_grad: bool,
    pub is_leaf: bool,
    pub memory_usage_bytes: usize,
    pub stride: Vec<usize>,
}

impl TensorInfo {
    /// Create tensor info from a tensor
    pub fn from_tensor(tensor: &Tensor) -> Self {
        Self {
            shape: tensor.shape().clone(),
            dtype: format!("{:?}", tensor.dtype()),
            device: tensor.device(),
            numel: tensor.numel(),
            requires_grad: tensor.requires_grad(),
            is_leaf: tensor.is_leaf(),
            memory_usage_bytes: tensor.memory_usage_bytes(),
            stride: tensor.stride().as_slice().to_vec(),
        }
    }

    /// Get a formatted summary of the tensor
    pub fn summary(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype={}, device={:?}, numel={}, requires_grad={}, memory={}MB)",
            self.shape.dims(),
            self.dtype,
            self.device,
            self.numel,
            self.requires_grad,
            self.memory_usage_bytes / 1024 / 1024
        )
    }

    /// Get detailed information as a formatted string
    pub fn detailed(&self) -> String {
        format!(
            "Tensor Information:\n\
             ├─ Shape: {:?}\n\
             ├─ Data Type: {}\n\
             ├─ Device: {:?}\n\
             ├─ Elements: {}\n\
             ├─ Memory: {} bytes ({:.2} MB)\n\
             ├─ Stride: {:?}\n\
             ├─ Requires Grad: {}\n\
             └─ Is Leaf: {}",
            self.shape.dims(),
            self.dtype,
            self.device,
            self.numel,
            self.memory_usage_bytes,
            self.memory_usage_bytes as f64 / 1024.0 / 1024.0,
            self.stride,
            self.requires_grad,
            self.is_leaf
        )
    }
}

/// Memory usage tracking for debugging
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    allocations: HashMap<String, usize>,
    peak_usage: usize,
    current_usage: usize,
}

impl MemoryTracker {
    /// Create a new memory tracker
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            peak_usage: 0,
            current_usage: 0,
        }
    }

    /// Record a memory allocation
    pub fn allocate(&mut self, name: String, size: usize) {
        self.allocations.insert(name, size);
        self.current_usage += size;
        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }
    }

    /// Record a memory deallocation
    pub fn deallocate(&mut self, name: &str) {
        if let Some(size) = self.allocations.remove(name) {
            self.current_usage = self.current_usage.saturating_sub(size);
        }
    }

    /// Get current memory usage in bytes
    pub fn current_usage(&self) -> usize {
        self.current_usage
    }

    /// Get peak memory usage in bytes
    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    /// Get current memory usage in MB
    pub fn current_usage_mb(&self) -> f64 {
        self.current_usage as f64 / 1024.0 / 1024.0
    }

    /// Get peak memory usage in MB
    pub fn peak_usage_mb(&self) -> f64 {
        self.peak_usage as f64 / 1024.0 / 1024.0
    }

    /// Get a summary of memory usage
    pub fn summary(&self) -> String {
        format!(
            "Memory Usage:\n\
             ├─ Current: {:.2} MB ({} bytes)\n\
             ├─ Peak: {:.2} MB ({} bytes)\n\
             └─ Active Allocations: {}",
            self.current_usage_mb(),
            self.current_usage,
            self.peak_usage_mb(),
            self.peak_usage,
            self.allocations.len()
        )
    }

    /// Get detailed allocation information
    pub fn detailed_allocations(&self) -> String {
        let mut result = String::from("Active Allocations:\n");
        for (name, size) in &self.allocations {
            result.push_str(&format!(
                "├─ {}: {:.2} MB ({} bytes)\n",
                name,
                *size as f64 / 1024.0 / 1024.0,
                size
            ));
        }
        result
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Computation graph visualization utilities
#[derive(Debug)]
pub struct GraphVisualizer {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: String,
    pub operation: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub requires_grad: bool,
}

#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub label: Option<String>,
}

impl GraphVisualizer {
    /// Create a new graph visualizer
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: GraphNode) {
        self.nodes.push(node);
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: GraphEdge) {
        self.edges.push(edge);
    }

    /// Generate a text-based visualization of the computation graph
    pub fn to_text(&self) -> String {
        let mut result = String::from("Computation Graph:\n");
        
        for node in &self.nodes {
            result.push_str(&format!(
                "Node {}: {} (shape: {:?}, dtype: {}, grad: {})\n",
                node.id, node.operation, node.shape, node.dtype, node.requires_grad
            ));
        }
        
        result.push('\n');
        
        for edge in &self.edges {
            let label = edge.label.as_deref().unwrap_or("");
            result.push_str(&format!("{} -> {} {}\n", edge.from, edge.to, label));
        }
        
        result
    }

    /// Generate DOT format for graphviz visualization
    pub fn to_dot(&self) -> String {
        let mut result = String::from("digraph ComputationGraph {\n");
        result.push_str("  rankdir=TB;\n");
        result.push_str("  node [shape=box];\n\n");
        
        for node in &self.nodes {
            let color = if node.requires_grad { "lightblue" } else { "lightgray" };
            result.push_str(&format!(
                "  \"{}\" [label=\"{}\\nshape: {:?}\\ndtype: {}\" fillcolor={} style=filled];\n",
                node.id, node.operation, node.shape, node.dtype, color
            ));
        }
        
        result.push('\n');
        
        for edge in &self.edges {
            let label = edge.label.as_deref().unwrap_or("");
            result.push_str(&format!("  \"{}\" -> \"{}\" [label=\"{}\"];\n", edge.from, edge.to, label));
        }
        
        result.push_str("}\n");
        result
    }
}

impl Default for GraphVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance profiler for operations
#[derive(Debug)]
pub struct OperationProfiler {
    timings: HashMap<String, Vec<f64>>,
    memory_usage: HashMap<String, Vec<usize>>,
}

impl OperationProfiler {
    /// Create a new operation profiler
    pub fn new() -> Self {
        Self {
            timings: HashMap::new(),
            memory_usage: HashMap::new(),
        }
    }

    /// Record timing for an operation
    pub fn record_timing(&mut self, operation: String, duration_ms: f64) {
        self.timings.entry(operation).or_default().push(duration_ms);
    }

    /// Record memory usage for an operation
    pub fn record_memory(&mut self, operation: String, memory_bytes: usize) {
        self.memory_usage.entry(operation).or_default().push(memory_bytes);
    }

    /// Get average timing for an operation
    pub fn average_timing(&self, operation: &str) -> Option<f64> {
        self.timings.get(operation).map(|times| {
            times.iter().sum::<f64>() / times.len() as f64
        })
    }

    /// Get average memory usage for an operation
    pub fn average_memory(&self, operation: &str) -> Option<f64> {
        self.memory_usage.get(operation).map(|memories| {
            memories.iter().sum::<usize>() as f64 / memories.len() as f64
        })
    }

    /// Generate a performance report
    pub fn report(&self) -> String {
        let mut result = String::from("Performance Report:\n");
        
        result.push_str("\nTiming Statistics:\n");
        for (op, times) in &self.timings {
            let avg = times.iter().sum::<f64>() / times.len() as f64;
            let min = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            result.push_str(&format!(
                "├─ {}: avg={:.2}ms, min={:.2}ms, max={:.2}ms, count={}\n",
                op, avg, min, max, times.len()
            ));
        }
        
        result.push_str("\nMemory Statistics:\n");
        for (op, memories) in &self.memory_usage {
            let avg = memories.iter().sum::<usize>() as f64 / memories.len() as f64;
            let min = *memories.iter().min().unwrap_or(&0);
            let max = *memories.iter().max().unwrap_or(&0);
            
            result.push_str(&format!(
                "├─ {}: avg={:.2}MB, min={:.2}MB, max={:.2}MB, count={}\n",
                op,
                avg / 1024.0 / 1024.0,
                min as f64 / 1024.0 / 1024.0,
                max as f64 / 1024.0 / 1024.0,
                memories.len()
            ));
        }
        
        result
    }
}

impl Default for OperationProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Debug utilities for tensor inspection
pub struct TensorDebugger;

impl TensorDebugger {
    /// Print detailed information about a tensor
    pub fn inspect(tensor: &Tensor) -> String {
        let info = TensorInfo::from_tensor(tensor);
        info.detailed()
    }

    /// Compare two tensors and highlight differences
    pub fn compare(tensor1: &Tensor, tensor2: &Tensor) -> String {
        let info1 = TensorInfo::from_tensor(tensor1);
        let info2 = TensorInfo::from_tensor(tensor2);
        
        let mut result = String::from("Tensor Comparison:\n");
        
        // Compare shapes
        if info1.shape.dims() != info2.shape.dims() {
            result.push_str(&format!(
                "❌ Shape: {:?} vs {:?}\n",
                info1.shape.dims(),
                info2.shape.dims()
            ));
        } else {
            result.push_str(&format!("✅ Shape: {:?}\n", info1.shape.dims()));
        }
        
        // Compare dtypes
        if info1.dtype != info2.dtype {
            result.push_str(&format!("❌ DType: {} vs {}\n", info1.dtype, info2.dtype));
        } else {
            result.push_str(&format!("✅ DType: {}\n", info1.dtype));
        }
        
        // Compare devices
        if info1.device != info2.device {
            result.push_str(&format!("❌ Device: {:?} vs {:?}\n", info1.device, info2.device));
        } else {
            result.push_str(&format!("✅ Device: {:?}\n", info1.device));
        }
        
        // Compare gradient requirements
        if info1.requires_grad != info2.requires_grad {
            result.push_str(&format!(
                "❌ Requires Grad: {} vs {}\n",
                info1.requires_grad, info2.requires_grad
            ));
        } else {
            result.push_str(&format!("✅ Requires Grad: {}\n", info1.requires_grad));
        }
        
        result
    }

    /// Check if tensor has common issues
    pub fn health_check(tensor: &Tensor) -> Vec<String> {
        let mut issues = Vec::new();
        
        // Check for NaN values
        if tensor.has_nan() {
            issues.push("⚠️  Tensor contains NaN values".to_string());
        }
        
        // Check for infinite values
        if tensor.has_inf() {
            issues.push("⚠️  Tensor contains infinite values".to_string());
        }
        
        // Check for very large values
        if let Some(max_val) = tensor.max_value() {
            if max_val > 1e6 {
                issues.push(format!("⚠️  Tensor has very large values (max: {:.2e})", max_val));
            }
        }
        
        // Check for very small gradients
        if tensor.requires_grad() {
            if let Some(grad) = tensor.grad() {
                if let Some(grad_max) = grad.max_value() {
                    if grad_max < 1e-8 {
                        issues.push("⚠️  Gradients are very small, may indicate vanishing gradient problem".to_string());
                    }
                }
            }
        }
        
        // Check memory usage
        let memory_mb = tensor.memory_usage_bytes() as f64 / 1024.0 / 1024.0;
        if memory_mb > 1000.0 {
            issues.push(format!("⚠️  Large memory usage: {:.2} MB", memory_mb));
        }
        
        if issues.is_empty() {
            issues.push("✅ No issues detected".to_string());
        }
        
        issues
    }
}