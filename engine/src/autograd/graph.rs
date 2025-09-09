// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::{GradientFunction, TensorId};
use crate::{error::Result, operations::arithmetic, tensor::Tensor};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::collections::hash_map::Entry;
use std::sync::Arc;

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

/// Node in the computation graph
pub struct GraphNode {
    /// Tensor ID
    pub tensor_id: TensorId,
    /// Gradient function for backward pass
    pub grad_fn: Option<Arc<dyn GradientFunction>>,
    /// Input tensor IDs
    pub inputs: SmallVec<[TensorId; 4]>,
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
        let mut inputs: SmallVec<[TensorId; 4]> = SmallVec::new();
        if let Some(f) = grad_fn.as_ref() {
            inputs.extend_from_slice(f.input_ids());
        }

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
        self.grad_fn.as_ref().map(|f| f.name()).unwrap_or("leaf")
    }
}

/// Computation graph for automatic differentiation
pub struct ComputationGraph {
    /// Nodes in the graph
    nodes: FxHashMap<TensorId, GraphNode>,
    /// Topological ordering for backward pass
    topological_order: Vec<TensorId>,
    /// Gradients computed during backward pass
    gradients: FxHashMap<TensorId, Tensor>,
    /// Whether the cached topological order is stale
    needs_order_update: bool,
}

impl ComputationGraph {
    /// Create a new empty computation graph
    pub fn new() -> Self {
        Self {
            nodes: FxHashMap::default(),
            topological_order: Vec::new(),
            gradients: FxHashMap::default(),
            needs_order_update: false,
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
        if let Some(ref f) = grad_fn {
            for &input_id in f.input_ids() {
                self.nodes
                    .entry(input_id)
                    .or_insert_with(|| GraphNode::new(input_id, None, true));
            }
        }

        let node = GraphNode::new(tensor_id, grad_fn, requires_grad);
        self.nodes.insert(tensor_id, node);
        self.needs_order_update = true;
    }

    /// Add a named tensor to the computation graph (for debugging)
    pub fn add_named_tensor(
        &mut self,
        tensor_id: TensorId,
        grad_fn: Option<Arc<dyn GradientFunction>>,
        requires_grad: bool,
        name: impl Into<String>,
    ) {
        let node = GraphNode::new(tensor_id, grad_fn, requires_grad).with_name(name);
        self.nodes.insert(tensor_id, node);
        self.needs_order_update = true;
    }

    /// Update the topological ordering of nodes using a non-recursive DFS.
    /// This avoids building explicit dependent lists and minimizes allocations
    /// compared to Kahn's algorithm while still detecting cycles.
    fn update_topological_order(&mut self) {
        self.topological_order.clear();
        self.topological_order.reserve(self.nodes.len());

        // 0 = unvisited, 1 = visiting, 2 = visited
        let mut state: FxHashMap<TensorId, u8> = FxHashMap::default();
        state.reserve(self.nodes.len());
        let mut stack: Vec<(TensorId, usize)> = Vec::with_capacity(self.nodes.len());
        let mut has_cycle = false;

        for &node_id in self.nodes.keys() {
            if state.get(&node_id).copied().unwrap_or(0) != 0 {
                continue;
            }
            stack.push((node_id, 0));
            while let Some((current_id, idx)) = stack.last_mut() {
                match state.get(current_id).copied().unwrap_or(0) {
                    0 => {
                        state.insert(*current_id, 1);
                    }
                    1 => {}
                    2 => {
                        stack.pop();
                        continue;
                    }
                    _ => unreachable!(),
                }

                let inputs = if let Some(node) = self.nodes.get(current_id) {
                    &node.inputs
                } else {
                    state.insert(*current_id, 2);
                    stack.pop();
                    continue;
                };

                if *idx < inputs.len() {
                    let next = inputs[*idx];
                    *idx += 1;
                    if !self.nodes.contains_key(&next) {
                        continue;
                    }
                    match state.get(&next).copied().unwrap_or(0) {
                        0 => stack.push((next, 0)),
                        1 => {
                            has_cycle = true;
                            break;
                        }
                        2 => {}
                        _ => unreachable!(),
                    }
                } else {
                    state.insert(*current_id, 2);
                    self.topological_order.push(*current_id);
                    stack.pop();
                }
            }
            if has_cycle {
                break;
            }
        }

        if has_cycle {
            self.topological_order.clear();
        } else {
            // We collected nodes from inputs to outputs; reverse for backward pass
            self.topological_order.reverse();
        }

        self.needs_order_update = false;
    }

    fn ensure_topological_order(&mut self) {
        if self.needs_order_update {
            self.update_topological_order();
        }
    }

    /// Perform backward pass from a given tensor
    pub fn backward(
        &mut self,
        start_tensor: TensorId,
        gradient: Option<Tensor>,
    ) -> Result<FxHashMap<TensorId, Tensor>> {
        self.ensure_topological_order();
        // Clear previous gradients and ensure sufficient capacity for accumulation
        self.gradients.clear();
        self.gradients.reserve(self.nodes.len());

        // Set the initial gradient
        if let Some(grad) = gradient {
            self.gradients.insert(start_tensor, grad);
        } else {
            return Err(crate::error::MinitensorError::gradient_error(
                "Initial gradient must be provided",
            ));
        }

        // Process nodes in topological order (outputs to inputs)
        for &node_id in &self.topological_order {
            if let Some(node) = self.nodes.get(&node_id) {
                // Skip if this node doesn't require gradients
                if !node.requires_grad {
                    continue;
                }

                // Take the gradient for this node out of the map to avoid cloning
                if let Some(grad_output) = self.gradients.remove(&node_id) {
                    // If this node has a gradient function, compute gradients for inputs
                    if let Some(grad_fn) = &node.grad_fn {
                        let input_grads = grad_fn.backward(&grad_output)?;
                        for (input_id, grad) in input_grads {
                            match self.gradients.entry(input_id) {
                                Entry::Occupied(mut e) => {
                                    arithmetic::add_inplace(e.get_mut(), &grad)?;
                                }
                                Entry::Vacant(e) => {
                                    e.insert(grad);
                                }
                            }
                        }
                    }
                    // Re-insert the gradient for this node so it is available in the
                    // final gradient map returned to callers
                    self.gradients.insert(node_id, grad_output);
                }
            }
        }

        Ok(self.gradients.clone())
    }

    /// Validate the computation graph for correctness
    pub fn validate(&mut self) -> Result<()> {
        // Check for cycles
        if self.has_cycles() {
            return Err(crate::error::MinitensorError::gradient_error(
                "Computation graph contains cycles",
            ));
        }

        // Validate that all input references exist
        for node in self.nodes.values() {
            for &input_id in &node.inputs {
                if !self.nodes.contains_key(&input_id) {
                    return Err(crate::error::MinitensorError::gradient_error(format!(
                        "Node {} references non-existent input {}",
                        node.tensor_id, input_id
                    )));
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
    pub fn topological_order(&mut self) -> &[TensorId] {
        self.ensure_topological_order();
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
    pub fn has_cycles(&mut self) -> bool {
        self.ensure_topological_order();
        self.topological_order.len() != self.nodes.len()
    }

    /// Remove a tensor and its dependencies from the graph
    pub fn remove_tensor(&mut self, tensor_id: TensorId) {
        if self.nodes.remove(&tensor_id).is_some() {
            // Remove from topological order
            self.topological_order.retain(|&id| id != tensor_id);

            // Remove any gradients
            self.gradients.remove(&tensor_id);

            // Mark order as stale since dependencies changed
            self.needs_order_update = true;
        }
    }

    /// Get statistics about the computation graph
    pub fn stats(&mut self) -> GraphStats {
        let leaf_nodes = self
            .nodes
            .values()
            .filter(|node| node.inputs.is_empty())
            .count();

        let grad_enabled_nodes = self
            .nodes
            .values()
            .filter(|node| node.requires_grad)
            .count();

        GraphStats {
            total_nodes: self.nodes.len(),
            leaf_nodes,
            grad_enabled_nodes,
            has_cycles: self.has_cycles(),
        }
    }

    /// Get all nodes in the computation graph
    pub fn nodes(&self) -> &FxHashMap<TensorId, GraphNode> {
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
    use crate::autograd::AddBackward;

    #[test]
    fn test_tensor_id() {
        let id1 = TensorId::new();
        let id2 = TensorId::new();

        assert_ne!(id1, id2);
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
        assert!(!graph.has_cycles());
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
    fn test_cycle_detection() {
        let mut graph = ComputationGraph::new();
        let a = TensorId::new();
        let b = TensorId::new();

        let add_a = Arc::new(AddBackward {
            input_shapes: [vec![1], vec![1]],
            input_ids: [b, b],
        });
        let add_b = Arc::new(AddBackward {
            input_shapes: [vec![1], vec![1]],
            input_ids: [a, a],
        });

        graph.add_tensor(a, Some(add_a));
        graph.add_tensor(b, Some(add_b));

        assert!(graph.has_cycles());
        assert!(graph.validate().is_err());
        assert!(graph.stats().has_cycles);
        assert!(graph.topological_order().is_empty());
    }

    #[test]
    fn test_backward_pass() {
        use crate::device::Device;
        use crate::tensor::{DataType, Shape, Tensor};

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
        use crate::device::Device;
        use crate::tensor::{DataType, Shape, Tensor};

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

    #[test]
    fn test_stats_zero_grad_and_removal() {
        use crate::device::Device;
        use crate::tensor::{DataType, Shape, Tensor};

        let mut graph = ComputationGraph::new();
        let a = TensorId::new();
        let b = TensorId::new();
        graph.add_tensor(a, None);
        graph.add_tensor(b, None);

        let c = TensorId::new();
        let add_fn = Arc::new(AddBackward {
            input_shapes: [vec![2], vec![2]],
            input_ids: [a, b],
        });
        graph.add_tensor(c, Some(add_fn));

        let stats = graph.stats();
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.leaf_nodes, 2);
        assert_eq!(stats.grad_enabled_nodes, 3);

        let grad = Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), false);
        graph.backward(c, Some(grad)).unwrap();
        assert!(graph.get_gradient(c).is_some());
        graph.zero_grad();
        assert!(graph.get_gradient(c).is_none());

        let order_before = graph.topological_order().to_vec();
        graph.remove_tensor(b);
        let order_after = graph.topological_order().to_vec();
        assert!(!graph.contains_tensor(b));
        assert!(!order_after.contains(&b));
        assert!(order_after.len() < order_before.len());
    }
}
