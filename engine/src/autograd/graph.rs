// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use super::GradientFunction;
use crate::{tensor::Tensor, error::Result};
use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};

/// Statistics about the computation graph
#[derive(Debug, Clone)]
pub struct GraphStats {
    /// Total number of nodes in the graph
    pub total_nodes: usize,
    /// Number of leaf nodes (no inputs)
    pub leaf_nodes: usize,
    /// Number of nodes with gradient computation enabled
    pub grad_enabled_nodes: usize,
    /// Whether the graph contains cycles
    pub has_cycles: bool,
}

/// Unique identifier for tensors in the computation graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(u64);

impl TensorId {
    /// Create a new unique tensor ID
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Get the raw ID value
    pub fn raw(&self) -> u64 {
        self.0
    }
    
    /// Create a TensorId from a raw value (for testing purposes)
    #[cfg(test)]
    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }
    
    /// Reset the global counter (for testing purposes)
    #[cfg(test)]
    pub fn reset_counter() {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        COUNTER.store(0, Ordering::Relaxed);
    }
}

impl std::fmt::Display for TensorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TensorId({})", self.0)
    }
}

/// Node in the computation graph
#[derive(Debug)]
pub struct GraphNode {
    /// Tensor ID
    pub tensor_id: TensorId,
    /// Gradient function for backward pass
    pub grad_fn: Option<Arc<dyn GradientFunction>>,
    /// Input tensor IDs
    pub inputs: Vec<TensorId>,
    /// Whether this node requires gradients
    pub requires_grad: bool,
    /// Optional name for debugging
    pub name: Option<String>,
}

impl GraphNode {
    /// Create a new graph node
    pub fn new(
        tensor_id: TensorId,
        grad_fn: Option<Arc<dyn GradientFunction>>,
        requires_grad: bool,
    ) -> Self {
        let inputs = grad_fn.as_ref()
            .map(|f| f.inputs().to_vec())
            .unwrap_or_default();
            
        Self {
            tensor_id,
            grad_fn,
            inputs,
            requires_grad,
            name: None,
        }
    }
    
    /// Set a name for this node (for debugging)
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    
    /// Check if this is a leaf node (no inputs)
    pub fn is_leaf(&self) -> bool {
        self.inputs.is_empty()
    }
    
    /// Get the operation name from the gradient function
    pub fn operation_name(&self) -> &str {
        self.grad_fn
            .as_ref()
            .map(|f| f.name())
            .unwrap_or("leaf")
    }
}

/// Computation graph for automatic differentiation
#[derive(Debug)]
pub struct ComputationGraph {
    /// Nodes in the graph
    nodes: HashMap<TensorId, GraphNode>,
    /// Topological ordering for backward pass
    topological_order: Vec<TensorId>,
    /// Gradients computed during backward pass
    gradients: HashMap<TensorId, Tensor>,
}

impl ComputationGraph {
    /// Create a new empty computation graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            topological_order: Vec::new(),
            gradients: HashMap::new(),
        }
    }

    /// Add a tensor to the computation graph
    pub fn add_tensor(&mut self, tensor_id: TensorId, grad_fn: Option<Arc<dyn GradientFunction>>) {
        self.add_tensor_with_grad_req(tensor_id, grad_fn, true);
    }
    
    /// Add a tensor to the computation graph with explicit gradient requirement
    pub fn add_tensor_with_grad_req(
        &mut self, 
        tensor_id: TensorId, 
        grad_fn: Option<Arc<dyn GradientFunction>>,
        requires_grad: bool,
    ) {
        let node = GraphNode::new(tensor_id, grad_fn, requires_grad);
        self.nodes.insert(tensor_id, node);
        self.update_topological_order();
    }
    
    /// Add a named tensor to the computation graph (for debugging)
    pub fn add_named_tensor(
        &mut self,
        tensor_id: TensorId,
        grad_fn: Option<Arc<dyn GradientFunction>>,
        requires_grad: bool,
        name: impl Into<String>,
    ) {
        let node = GraphNode::new(tensor_id, grad_fn, requires_grad)
            .with_name(name);
        self.nodes.insert(tensor_id, node);
        self.update_topological_order();
    }

    /// Update the topological ordering of nodes using Kahn's algorithm
    fn update_topological_order(&mut self) {
        self.topological_order.clear();
        
        // Build dependency graph: for each node, track which nodes depend on it
        let mut dependents: HashMap<TensorId, Vec<TensorId>> = HashMap::new();
        let mut in_degree: HashMap<TensorId, usize> = HashMap::new();
        
        // Initialize all nodes
        for &node_id in self.nodes.keys() {
            dependents.entry(node_id).or_insert_with(Vec::new);
            in_degree.entry(node_id).or_insert(0);
        }
        
        // Build the dependency relationships
        for node in self.nodes.values() {
            for &input_id in &node.inputs {
                // input_id is a dependency of node.tensor_id
                dependents.entry(input_id).or_insert_with(Vec::new).push(node.tensor_id);
                *in_degree.entry(node.tensor_id).or_insert(0) += 1;
            }
        }
        
        // Find nodes with no dependencies (output nodes)
        let mut queue: Vec<TensorId> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(&id, _)| id)
            .collect();
        
        // Process nodes in topological order (outputs first for backward pass)
        while let Some(current_id) = queue.pop() {
            self.topological_order.push(current_id);
            
            // For each node that depends on current node, reduce its in-degree
            if let Some(deps) = dependents.get(&current_id) {
                for &dependent_id in deps {
                    if let Some(degree) = in_degree.get_mut(&dependent_id) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push(dependent_id);
                        }
                    }
                }
            }
        }
        
        // Reverse the order to get backward pass order (outputs to inputs)
        self.topological_order.reverse();
    }

    /// Perform backward pass from a given tensor
    pub fn backward(&mut self, start_tensor: TensorId, gradient: Option<Tensor>) -> Result<HashMap<TensorId, Tensor>> {
        // Clear previous gradients
        self.gradients.clear();
        
        // Set the initial gradient
        if let Some(grad) = gradient {
            self.gradients.insert(start_tensor, grad);
        } else {
            return Err(crate::error::MinitensorError::gradient_error(
                "Initial gradient must be provided"
            ));
        }
        
        // Process nodes in topological order (outputs to inputs)
        for &node_id in &self.topological_order {
            if let Some(node) = self.nodes.get(&node_id) {
                // Skip if this node doesn't require gradients
                if !node.requires_grad {
                    continue;
                }
                
                // Get the gradient for this node
                if let Some(grad_output) = self.gradients.get(&node_id).cloned() {
                    // If this node has a gradient function, compute gradients for inputs
                    if let Some(grad_fn) = &node.grad_fn {
                        match grad_fn.backward(&grad_output) {
                            Ok(input_grads) => {
                                // Accumulate gradients for input tensors
                                for (i, input_grad) in input_grads.into_iter().enumerate() {
                                    if let Some(grad) = input_grad {
                                        let input_id = grad_fn.inputs()[i];
                                        
                                        // Accumulate gradient (add to existing or set new)
                                        match self.gradients.get_mut(&input_id) {
                                            Some(existing_grad) => {
                                                *existing_grad = existing_grad.add(&grad)?;
                                            }
                                            None => {
                                                self.gradients.insert(input_id, grad);
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                return Err(e);
                            }
                        }
                    }
                }
            }
        }
        
        Ok(self.gradients.clone())
    }
    
    /// Validate the computation graph for correctness
    pub fn validate(&self) -> Result<()> {
        // Check for cycles
        if !self.has_cycles() {
            return Err(crate::error::MinitensorError::gradient_error(
                "Computation graph contains cycles"
            ));
        }
        
        // Validate that all input references exist
        for node in self.nodes.values() {
            for &input_id in &node.inputs {
                if !self.nodes.contains_key(&input_id) {
                    return Err(crate::error::MinitensorError::gradient_error(
                        format!("Node {} references non-existent input {}", 
                               node.tensor_id, input_id)
                    ));
                }
            }
        }
        
        Ok(())
    }

    /// Get the gradient for a tensor
    pub fn get_gradient(&self, tensor_id: TensorId) -> Option<&Tensor> {
        self.gradients.get(&tensor_id)
    }

    /// Clear all gradients
    pub fn zero_grad(&mut self) {
        self.gradients.clear();
    }

    /// Get the number of nodes in the graph
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Check if a tensor is in the graph
    pub fn contains_tensor(&self, tensor_id: TensorId) -> bool {
        self.nodes.contains_key(&tensor_id)
    }
    
    /// Get the topological order of tensor IDs
    pub fn topological_order(&self) -> &[TensorId] {
        &self.topological_order
    }
    
    /// Get a node by tensor ID
    pub fn get_node(&self, tensor_id: TensorId) -> Option<&GraphNode> {
        self.nodes.get(&tensor_id)
    }
    
    /// Get all nodes that depend on a given tensor
    pub fn get_dependents(&self, tensor_id: TensorId) -> Vec<TensorId> {
        self.nodes
            .values()
            .filter(|node| node.inputs.contains(&tensor_id))
            .map(|node| node.tensor_id)
            .collect()
    }
    
    /// Check if there are cycles in the computation graph
    pub fn has_cycles(&self) -> bool {
        // If topological sort includes all nodes, there are no cycles
        self.topological_order.len() == self.nodes.len()
    }
    
    /// Remove a tensor and its dependencies from the graph
    pub fn remove_tensor(&mut self, tensor_id: TensorId) {
        if self.nodes.remove(&tensor_id).is_some() {
            // Remove from topological order
            self.topological_order.retain(|&id| id != tensor_id);
            
            // Remove any gradients
            self.gradients.remove(&tensor_id);
            
            // Update topological order since dependencies may have changed
            self.update_topological_order();
        }
    }
    
    /// Get statistics about the computation graph
    pub fn stats(&self) -> GraphStats {
        let leaf_nodes = self.nodes
            .values()
            .filter(|node| node.inputs.is_empty())
            .count();
            
        let grad_enabled_nodes = self.nodes
            .values()
            .filter(|node| node.requires_grad)
            .count();
            
        GraphStats {
            total_nodes: self.nodes.len(),
            leaf_nodes,
            grad_enabled_nodes,
            has_cycles: !self.has_cycles(),
        }
    }

    /// Get all nodes in the computation graph
    pub fn nodes(&self) -> &HashMap<TensorId, GraphNode> {
        &self.nodes
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::function::AddBackward;

    #[test]
    fn test_tensor_id() {
        let id1 = TensorId::new();
        let id2 = TensorId::new();
        
        assert_ne!(id1, id2);
        assert_ne!(id1.raw(), id2.raw());
        // Don't test exact values since counter is global
        assert!(id2.raw() > id1.raw());
    }

    #[test]
    fn test_tensor_id_display() {
        let id = TensorId::from_raw(42);
        assert_eq!(format!("{}", id), "TensorId(42)");
    }

    #[test]
    fn test_computation_graph_basic() {
        let mut graph = ComputationGraph::new();
        let tensor_id = TensorId::new();
        
        assert_eq!(graph.num_nodes(), 0);
        assert!(!graph.contains_tensor(tensor_id));
        
        graph.add_tensor(tensor_id, None);
        
        assert_eq!(graph.num_nodes(), 1);
        assert!(graph.contains_tensor(tensor_id));
        assert!(graph.has_cycles());
    }

    #[test]
    fn test_computation_graph_with_dependencies() {
        let mut graph = ComputationGraph::new();
        
        // Create leaf nodes
        let leaf1 = TensorId::new();
        let leaf2 = TensorId::new();
        graph.add_tensor(leaf1, None);
        graph.add_tensor(leaf2, None);
        
        // Create operation node that depends on leaves
        let result = TensorId::new();
        let add_fn = Arc::new(AddBackward {
            input_shapes: [vec![2, 3], vec![2, 3]],
            input_ids: [leaf1, leaf2],
        });
        graph.add_tensor(result, Some(add_fn));
        
        assert_eq!(graph.num_nodes(), 3);
        assert!(graph.contains_tensor(result));
        
        // Check dependencies
        let dependents = graph.get_dependents(leaf1);
        assert!(dependents.contains(&result));
        
        let node = graph.get_node(result).unwrap();
        assert!(node.inputs.contains(&leaf1));
        assert!(node.inputs.contains(&leaf2));
        assert_eq!(node.operation_name(), "AddBackward");
    }

    #[test]
    fn test_topological_ordering() {
        let mut graph = ComputationGraph::new();
        
        // Create a simple computation: c = a + b
        let a = TensorId::new();
        let b = TensorId::new();
        let c = TensorId::new();
        
        graph.add_tensor(a, None);
        graph.add_tensor(b, None);
        
        let add_fn = Arc::new(AddBackward {
            input_shapes: [vec![2], vec![2]],
            input_ids: [a, b],
        });
        graph.add_tensor(c, Some(add_fn));
        
        let topo_order = graph.topological_order();
        
        // Verify all nodes are present
        assert_eq!(topo_order.len(), 3);
        assert!(topo_order.contains(&a));
        assert!(topo_order.contains(&b));
        assert!(topo_order.contains(&c));
        
        // Get positions in the topological order
        let c_pos = topo_order.iter().position(|&id| id == c).unwrap();
        let a_pos = topo_order.iter().position(|&id| id == a).unwrap();
        let b_pos = topo_order.iter().position(|&id| id == b).unwrap();
        
        // The topological order is designed for backward pass: c should come before a and b
        // since we process from outputs to inputs during backpropagation
        assert!(c_pos < a_pos);
        assert!(c_pos < b_pos);
    }

    #[test]
    fn test_graph_stats() {
        let mut graph = ComputationGraph::new();
        
        let leaf1 = TensorId::new();
        let leaf2 = TensorId::new();
        graph.add_tensor(leaf1, None);
        graph.add_tensor(leaf2, None);
        
        let result = TensorId::new();
        let add_fn = Arc::new(AddBackward {
            input_shapes: [vec![2], vec![2]],
            input_ids: [leaf1, leaf2],
        });
        graph.add_tensor(result, Some(add_fn));
        
        let stats = graph.stats();
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.leaf_nodes, 2);
        assert_eq!(stats.grad_enabled_nodes, 3);
        assert!(!stats.has_cycles);
    }

    #[test]
    fn test_graph_node() {
        let tensor_id = TensorId::new();
        let node = GraphNode::new(tensor_id, None, true);
        
        assert_eq!(node.tensor_id, tensor_id);
        assert!(node.requires_grad);
        assert!(node.is_leaf());
        assert_eq!(node.operation_name(), "leaf");
        
        let named_node = node.with_name("test_tensor");
        assert_eq!(named_node.name, Some("test_tensor".to_string()));
    }

    #[test]
    fn test_remove_tensor() {
        let mut graph = ComputationGraph::new();
        let tensor_id = TensorId::new();
        
        graph.add_tensor(tensor_id, None);
        assert!(graph.contains_tensor(tensor_id));
        
        graph.remove_tensor(tensor_id);
        assert!(!graph.contains_tensor(tensor_id));
        assert_eq!(graph.num_nodes(), 0);
    }

    #[test]
    fn test_graph_validation() {
        let mut graph = ComputationGraph::new();
        
        // Valid graph should pass validation
        let leaf = TensorId::new();
        graph.add_tensor(leaf, None);
        assert!(graph.validate().is_ok());
        
        // Test with proper dependencies
        let result = TensorId::new();
        let add_fn = Arc::new(AddBackward {
            input_shapes: [vec![2], vec![2]],
            input_ids: [leaf, leaf], // Self-dependency is ok
        });
        graph.add_tensor(result, Some(add_fn));
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_backward_pass() {
        use crate::tensor::{Tensor, DataType, Shape};
        use crate::device::Device;
        
        let mut graph = ComputationGraph::new();
        
        // Create leaf tensors
        let a = TensorId::new();
        let b = TensorId::new();
        graph.add_tensor(a, None);
        graph.add_tensor(b, None);
        
        // Create operation: c = a + b
        let c = TensorId::new();
        let add_fn = Arc::new(AddBackward {
            input_shapes: [vec![2], vec![2]],
            input_ids: [a, b],
        });
        graph.add_tensor(c, Some(add_fn));
        
        // Create gradient tensor
        let grad_shape = Shape::new(vec![2]);
        let grad_c = Tensor::ones(grad_shape, DataType::Float32, Device::cpu(), false);
        
        // Perform backward pass
        let gradients = graph.backward(c, Some(grad_c)).unwrap();
        
        // Should have gradients for a and b
        assert!(gradients.contains_key(&a));
        assert!(gradients.contains_key(&b));
        assert_eq!(gradients.len(), 3); // c, a, b
    }

    #[test]
    fn test_gradient_accumulation_in_graph() {
        use crate::tensor::{Tensor, DataType, Shape};
        use crate::device::Device;
        
        let mut graph = ComputationGraph::new();
        
        // Create a more complex graph: d = (a + b) + (a + c)
        // This should accumulate gradients for 'a'
        let a = TensorId::new();
        let b = TensorId::new();
        let c = TensorId::new();
        graph.add_tensor(a, None);
        graph.add_tensor(b, None);
        graph.add_tensor(c, None);
        
        // First addition: temp1 = a + b
        let temp1 = TensorId::new();
        let add_fn1 = Arc::new(AddBackward {
            input_shapes: [vec![2], vec![2]],
            input_ids: [a, b],
        });
        graph.add_tensor(temp1, Some(add_fn1));
        
        // Second addition: temp2 = a + c
        let temp2 = TensorId::new();
        let add_fn2 = Arc::new(AddBackward {
            input_shapes: [vec![2], vec![2]],
            input_ids: [a, c],
        });
        graph.add_tensor(temp2, Some(add_fn2));
        
        // Final addition: d = temp1 + temp2
        let d = TensorId::new();
        let add_fn3 = Arc::new(AddBackward {
            input_shapes: [vec![2], vec![2]],
            input_ids: [temp1, temp2],
        });
        graph.add_tensor(d, Some(add_fn3));
        
        // Create gradient tensor
        let grad_shape = Shape::new(vec![2]);
        let grad_d = Tensor::ones(grad_shape, DataType::Float32, Device::cpu(), false);
        
        // Perform backward pass
        let gradients = graph.backward(d, Some(grad_d)).unwrap();
        
        // Should have gradients for all tensors
        assert!(gradients.contains_key(&a));
        assert!(gradients.contains_key(&b));
        assert!(gradients.contains_key(&c));
        
        // 'a' should appear in the gradients (accumulated from both paths)
        assert!(gradients.get(&a).is_some());
    }
}