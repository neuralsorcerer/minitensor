// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::{GradientFunction, TensorId};
use crate::{error::Result, ops::arithmetic, tensor::Tensor};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::hash_map::Entry;
use std::sync::{Arc, Mutex, Weak};

/// A gradient function together with the ones that produced its inputs.
///
/// The graph finds a recorded node by its output's id but does not keep it;
/// this is what does. An output holds its node, each node holds the nodes
/// before it, and so a tensor that can still be backpropagated keeps exactly
/// its own history alive. Once the last such tensor is gone, so is the history
/// and every activation it saved -- which is how a forward pass that is never
/// backpropagated costs nothing once its result is dropped, as in PyTorch,
/// rather than holding its activations until someone clears the graph.
///
/// Held links cannot form a cycle: a node only links to nodes that existed
/// before it.
struct Linked {
    function: Arc<dyn GradientFunction>,
    upstream: Mutex<SmallVec<[Arc<dyn GradientFunction>; 4]>>,
}

impl GradientFunction for Linked {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        self.function.backward(grad_output)
    }

    fn input_ids(&self) -> &[TensorId] {
        self.function.input_ids()
    }

    fn name(&self) -> &'static str {
        self.function.name()
    }

    fn release_upstream(&self) {
        let links = std::mem::take(&mut *self.upstream.lock().unwrap_or_else(|p| p.into_inner()));
        crate::tensor::release_grad_fns(links);
    }
}

impl Drop for Linked {
    fn drop(&mut self) {
        // The links go through the release queue rather than being dropped
        // here: a chain of them is as long as the forward pass was, and
        // dropping it by recursion overflows the stack (see `Tensor`'s drop).
        // The function itself can go inline -- the tensors it saved already
        // release through that queue as they drop.
        let links = std::mem::take(self.upstream.get_mut().unwrap_or_else(|p| p.into_inner()));
        crate::tensor::release_grad_fns(links);
    }
}

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
    /// Gradient function for backward pass, and who keeps it alive.
    function: NodeFunction,
    /// Input tensor IDs
    pub inputs: SmallVec<[TensorId; 4]>,
    /// Whether this node requires gradients
    pub requires_grad: bool,
    /// Optional name for debugging
    pub name: Option<String>,
}

/// Who keeps a node's gradient function alive.
enum NodeFunction {
    /// A leaf: nothing to run.
    Leaf,
    /// The graph does. For nodes added to it directly, as its own tests do.
    Owned(Arc<dyn GradientFunction>),
    /// The node's output tensor does, and so does every node downstream of it,
    /// through the links `autograd::with_grad_fn` gives each function to the
    /// ones before it. The graph only finds the node by id: once no tensor can
    /// backpropagate through it, the function and everything it saved are
    /// gone, and the entry is swept.
    Tracked(Weak<dyn GradientFunction>),
}

impl GraphNode {
    /// Create a new graph node
    pub fn new(
        tensor_id: TensorId,
        grad_fn: Option<Arc<dyn GradientFunction>>,
        requires_grad: bool,
    ) -> Self {
        let inputs = Self::inputs_of(grad_fn.as_ref());
        Self {
            tensor_id,
            function: grad_fn.map_or(NodeFunction::Leaf, NodeFunction::Owned),
            inputs,
            requires_grad,
            name: None,
        }
    }

    /// A node the graph finds but does not keep; see [`NodeFunction::Tracked`].
    fn tracked(
        tensor_id: TensorId,
        grad_fn: &Arc<dyn GradientFunction>,
        requires_grad: bool,
    ) -> Self {
        Self {
            tensor_id,
            function: NodeFunction::Tracked(Arc::downgrade(grad_fn)),
            inputs: Self::inputs_of(Some(grad_fn)),
            requires_grad,
            name: None,
        }
    }

    fn inputs_of(grad_fn: Option<&Arc<dyn GradientFunction>>) -> SmallVec<[TensorId; 4]> {
        grad_fn.map_or_else(SmallVec::new, |f| SmallVec::from_slice(f.input_ids()))
    }

    /// The node's gradient function, if it has one and it is still alive.
    pub fn grad_fn(&self) -> Option<Arc<dyn GradientFunction>> {
        match &self.function {
            NodeFunction::Leaf => None,
            NodeFunction::Owned(function) => Some(Arc::clone(function)),
            NodeFunction::Tracked(function) => function.upgrade(),
        }
    }

    /// Whether the node was recorded with a gradient function, alive or not.
    fn is_interior(&self) -> bool {
        !matches!(self.function, NodeFunction::Leaf)
    }

    /// False once a tracked node's function is gone: nothing can reach it any
    /// more, and it is waiting to be swept.
    fn is_live(&self) -> bool {
        match &self.function {
            NodeFunction::Tracked(function) => function.strong_count() > 0,
            _ => true,
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
        match &self.function {
            NodeFunction::Leaf => "leaf",
            NodeFunction::Owned(function) => function.name(),
            NodeFunction::Tracked(function) => function.upgrade().map_or("released", |f| f.name()),
        }
    }
}

/// A single step of a backward pass: the tensor to propagate from and the
/// gradient function (if any) that produces gradients for its inputs.
///
/// Plans hold `Arc` clones of the gradient functions so they can be executed
/// without keeping the graph borrowed. This is what allows the thread-local
/// graph to stay accessible while user-visible gradient kernels run.
pub struct BackwardStep {
    pub tensor_id: TensorId,
    pub grad_fn: Option<Arc<dyn GradientFunction>>,
    pub requires_grad: bool,
}

/// Execute a previously planned backward pass.
///
/// `plan` must be in reverse-topological order (outputs before inputs), as
/// produced by [`ComputationGraph::plan_backward`]. The function is free of
/// any borrow of the graph itself, so it is safe to run while the global
/// graph is unlocked.
pub fn execute_backward_plan(
    plan: &[BackwardStep],
    start_tensor: TensorId,
    gradient: Tensor,
) -> Result<FxHashMap<TensorId, Tensor>> {
    let mut gradients: FxHashMap<TensorId, Tensor> = FxHashMap::default();
    gradients.reserve(plan.len().max(1));
    gradients.insert(start_tensor, gradient);

    for step in plan {
        // Skip nodes that never received a gradient (dead branches) and nodes
        // that do not participate in differentiation.
        if !step.requires_grad {
            continue;
        }
        // Take the gradient for this node out of the map to avoid cloning.
        if let Some(grad_output) = gradients.remove(&step.tensor_id) {
            if let Some(grad_fn) = &step.grad_fn {
                let input_grads = grad_fn.backward(&grad_output)?;
                for (input_id, grad) in input_grads {
                    match gradients.entry(input_id) {
                        Entry::Occupied(mut e) => {
                            arithmetic::add_inplace(e.get_mut(), &grad)?;
                        }
                        Entry::Vacant(e) => {
                            e.insert(grad);
                        }
                    }
                }
            }
            // Re-insert the gradient for this node so it remains available to
            // callers via `get_gradient` after the pass completes.
            gradients.insert(step.tensor_id, grad_output);
        }
    }

    Ok(gradients)
}

/// Computation graph for automatic differentiation
pub struct ComputationGraph {
    /// Nodes in the graph
    nodes: FxHashMap<TensorId, GraphNode>,
    /// Gradients computed during backward pass
    gradients: FxHashMap<TensorId, Tensor>,
    /// Interior tensors whose gradients the most recent
    /// [`Self::release_saved_subgraph`] kept, so that a non-leaf `.grad` can
    /// still be read after the pass that produced it. Dropped one pass later.
    retained_interior: FxHashSet<TensorId>,
    /// The node count at which [`Self::track`] next sweeps out released
    /// nodes: twice what the last sweep left, so sweeping costs a constant
    /// amount per node recorded.
    sweep_at: usize,
}

/// The smallest graph worth sweeping.
const MIN_SWEEP: usize = 4096;

impl ComputationGraph {
    /// Create a new empty computation graph
    pub fn new() -> Self {
        Self {
            nodes: FxHashMap::default(),
            gradients: FxHashMap::default(),
            retained_interior: FxHashSet::default(),
            sweep_at: MIN_SWEEP,
        }
    }

    /// Record `tensor_id` as produced by `grad_fn`, and return the function the
    /// tensor should own: `grad_fn` linked to the nodes that produced its
    /// inputs (see [`Linked`]). The graph finds the node but does not keep it.
    pub(crate) fn record(
        &mut self,
        tensor_id: TensorId,
        grad_fn: Arc<dyn GradientFunction>,
        requires_grad: bool,
    ) -> Arc<dyn GradientFunction> {
        // One lookup per input does both jobs: the placeholder a leaf needs,
        // and the link a recorded input needs.
        let mut upstream = SmallVec::new();
        for &input_id in grad_fn.input_ids() {
            let node = self
                .nodes
                .entry(input_id)
                .or_insert_with(|| GraphNode::new(input_id, None, true));
            upstream.extend(node.grad_fn());
        }
        let linked: Arc<dyn GradientFunction> = Arc::new(Linked {
            function: grad_fn,
            upstream: Mutex::new(upstream),
        });
        self.nodes.insert(
            tensor_id,
            GraphNode::tracked(tensor_id, &linked, requires_grad),
        );
        if self.nodes.len() >= self.sweep_at {
            self.sweep();
        }
        linked
    }

    /// Drop the entries of nodes nothing can backpropagate through any more,
    /// and the leaf placeholders only they referred to.
    ///
    /// A leaf is kept while a live node consumes it or a gradient is stored for
    /// it, the same rule [`Self::release_saved_subgraph`] applies to the leaves
    /// of a pass.
    pub fn sweep(&mut self) {
        self.nodes.retain(|_, node| node.is_live());
        let consumed: FxHashSet<TensorId> = self
            .nodes
            .values()
            .flat_map(|node| node.inputs.iter().copied())
            .collect();
        let gradients = &self.gradients;
        self.nodes.retain(|id, node| {
            node.is_interior() || consumed.contains(id) || gradients.contains_key(id)
        });
        self.sweep_at = (self.nodes.len() * 2).max(MIN_SWEEP);
    }

    fn add_input_placeholders(&mut self, input_ids: &[TensorId]) {
        for &input_id in input_ids {
            self.nodes
                .entry(input_id)
                .or_insert_with(|| GraphNode::new(input_id, None, true));
        }
    }

    /// The node for `id`, unless it has been released.
    fn live_node(&self, id: &TensorId) -> Option<&GraphNode> {
        self.nodes.get(id).filter(|node| node.is_live())
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
            self.add_input_placeholders(f.input_ids());
        }

        let node = GraphNode::new(tensor_id, grad_fn, requires_grad);
        self.nodes.insert(tensor_id, node);
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
    }

    /// Visit the subgraph reachable from `start` through input edges, in
    /// reverse-topological order (outputs before inputs), calling `visit` for
    /// each reachable node id. Detects cycles and reports them as errors.
    fn visit_reachable_reverse_topo(
        &self,
        start: TensorId,
        mut visit: impl FnMut(&GraphNode),
    ) -> Result<()> {
        if self.live_node(&start).is_none() {
            return Ok(());
        }

        // 0 = unvisited, 1 = visiting, 2 = visited
        let mut state: FxHashMap<TensorId, u8> = FxHashMap::default();
        let mut stack: Vec<(TensorId, usize)> = Vec::new();
        // Post-order collection (inputs before outputs); reversed at the end.
        let mut post_order: Vec<TensorId> = Vec::new();

        stack.push((start, 0));
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

            let inputs = match self.live_node(current_id) {
                Some(node) => &node.inputs,
                None => {
                    state.insert(*current_id, 2);
                    stack.pop();
                    continue;
                }
            };

            if *idx < inputs.len() {
                let next = inputs[*idx];
                *idx += 1;
                if self.live_node(&next).is_none() {
                    continue;
                }
                match state.get(&next).copied().unwrap_or(0) {
                    0 => stack.push((next, 0)),
                    1 => {
                        return Err(crate::error::MinitensorError::gradient_error(
                            "Computation graph contains cycles",
                        ));
                    }
                    2 => {}
                    _ => unreachable!(),
                }
            } else {
                state.insert(*current_id, 2);
                post_order.push(*current_id);
                stack.pop();
            }
        }

        for id in post_order.into_iter().rev() {
            if let Some(node) = self.live_node(&id) {
                visit(node);
            }
        }
        Ok(())
    }

    /// Build an executable backward plan for the subgraph reachable from
    /// `start_tensor`. Only reachable nodes are visited, so the cost of a
    /// backward pass is proportional to the size of the traced subgraph, not
    /// to every tensor ever recorded on this thread.
    pub fn plan_backward(&self, start_tensor: TensorId) -> Result<Vec<BackwardStep>> {
        let mut plan = Vec::new();
        self.visit_reachable_reverse_topo(start_tensor, |node| {
            plan.push(BackwardStep {
                tensor_id: node.tensor_id,
                grad_fn: node.grad_fn(),
                requires_grad: node.requires_grad,
            });
        })?;
        Ok(plan)
    }

    /// Replace the stored gradient map with the results of a backward pass.
    pub fn set_gradients(&mut self, gradients: FxHashMap<TensorId, Tensor>) {
        self.gradients = gradients;
    }

    /// Accumulate a freshly computed gradient map into the stored gradients,
    /// summing on collision. That is the `backward()` contract: repeated
    /// backward passes add into `.grad` (of the persistent leaf tensors) until
    /// `zero_grad` clears it, which is what enables gradient accumulation over
    /// micro-batches and multi-loss training. Interior (non-leaf) gradients are
    /// transient and dropped by [`Self::release_saved_subgraph`] after each
    /// non-retaining pass, so this map does not grow across training steps.
    ///
    /// The caller runs this with grad recording disabled; `add_inplace` copies
    /// on write when the stored gradient is shared (e.g. a `.grad` clone the
    /// user still holds), so accumulation never mutates a returned gradient.
    pub fn accumulate_gradients(&mut self, gradients: FxHashMap<TensorId, Tensor>) -> Result<()> {
        for (id, grad) in gradients {
            match self.gradients.entry(id) {
                Entry::Occupied(mut existing) => {
                    arithmetic::add_inplace(existing.get_mut(), &grad)?;
                }
                Entry::Vacant(slot) => {
                    slot.insert(grad);
                }
            }
        }
        Ok(())
    }

    /// Clone the stored gradient map. Intended for tests and diagnostics; the
    /// training path reads individual gradients via [`Self::get_gradient`].
    pub fn gradients_snapshot(&self) -> FxHashMap<TensorId, Tensor> {
        self.gradients.clone()
    }

    /// Drop every node reachable from `start` that carries a gradient
    /// function. This releases the tensors captured for backward (activations,
    /// saved operands) as soon as the pass is finished, instead of holding
    /// them until the next optimizer step. Leaf nodes and stored gradients are
    /// preserved so `get_gradient` keeps working.
    ///
    /// Returns the released gradient functions. A tracked one is not the
    /// graph's to free -- its output and the nodes downstream own it -- so the
    /// caller cuts their links to the nodes before them
    /// ([`GradientFunction::release_upstream`]), after letting go of the
    /// graph: that is what frees the pass's history while the loss that
    /// started it is still held.
    #[must_use = "the released functions still hold their history until unlinked"]
    pub fn release_saved_subgraph(&mut self, start: TensorId) -> Vec<Arc<dyn GradientFunction>> {
        let mut interior = FxHashSet::default();
        let mut released = Vec::new();
        let mut spent_leaves = Vec::new();
        // Ignore cycle errors here: releasing is best-effort cleanup.
        let _ = self.visit_reachable_reverse_topo(start, |node| {
            if node.is_interior() {
                interior.insert(node.tensor_id);
                released.extend(node.grad_fn());
            } else {
                spent_leaves.push(node.tensor_id);
            }
        });

        // Interior gradient *entries* are kept, not dropped with their node:
        // minitensor exposes non-leaf `.grad` after backward.
        // They are only meaningful for the pass that produced them, though, so
        // this drops the ones the previous pass left behind.
        //
        // Keeping them indefinitely was a leak. Interior tensors get a fresh
        // id on every forward pass, so nothing ever overwrote an old entry, and
        // the only thing that emptied the map was `optimizer.step()` resetting
        // the whole graph. A `backward()` not followed by a step -- which is
        // exactly what gradient accumulation does -- therefore retained every
        // activation gradient it had ever computed.
        let stale = std::mem::replace(&mut self.retained_interior, interior);
        for id in stale {
            // A tensor that this pass wrote a gradient for keeps it.
            if !self.retained_interior.contains(&id) {
                self.gradients.remove(&id);
            }
        }

        for id in &self.retained_interior {
            self.nodes.remove(id);
        }

        // A leaf that the pass produced no gradient for is not a parameter --
        // it is a constant, or an input that does not require one. Every
        // scalar operand is one of these: `x * 2.0` wraps the `2.0` in a fresh
        // tensor with a fresh id, which enters the graph only as a placeholder
        // for that grad_fn's input list, and nothing removed it. One node per
        // call, forever, for every op with a scalar operand.
        //
        // Parameters keep theirs, because a backward that reached them stored
        // a gradient. One that did not reach them loses the placeholder and
        // regains it on the next forward, which is where placeholders come
        // from in the first place.
        for id in spent_leaves {
            if !self.gradients.contains_key(&id) {
                self.nodes.remove(&id);
            }
        }
        released
    }

    /// Perform backward pass from a given tensor.
    ///
    /// Gradients are stored in the graph and can be queried afterwards with
    /// [`Self::get_gradient`]; nothing is cloned on the hot path.
    pub fn backward(&mut self, start_tensor: TensorId, gradient: Option<Tensor>) -> Result<()> {
        if !self.nodes.contains_key(&start_tensor) && crate::autograd::is_graph_consumed() {
            return Err(
                crate::error::MinitensorError::gradient_error_with_suggestion(
                    "Computation graph for this tensor has already been freed",
                    "Re-run the forward pass or call backward(retain_graph=True)",
                    None,
                ),
            );
        }

        let gradient = gradient.ok_or_else(|| {
            crate::error::MinitensorError::gradient_error("Initial gradient must be provided")
        })?;

        let plan = self.plan_backward(start_tensor)?;
        // Gradient kernels must not record new autograd nodes; make that
        // explicit instead of relying on the graph being borrowed.
        let _guard = crate::autograd::NoGradGuard::new();
        self.gradients = execute_backward_plan(&plan, start_tensor, gradient)?;
        Ok(())
    }

    /// Validate the computation graph for correctness
    pub fn validate(&self) -> Result<()> {
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

    /// How many gradients are currently stored.
    pub fn num_gradients(&self) -> usize {
        self.gradients.len()
    }

    /// Clear all gradients
    pub fn zero_grad(&mut self) {
        self.gradients.clear();
        // Nothing left for the next release to drop.
        self.retained_interior.clear();
    }

    /// Remove the stored gradient for a single tensor, if any.
    pub fn remove_gradient(&mut self, tensor_id: TensorId) -> Option<Tensor> {
        self.gradients.remove(&tensor_id)
    }

    /// Overwrite the stored gradient for a single tensor. Unlike
    /// [`Self::accumulate_gradients`] this replaces rather than sums, which is
    /// what in-place gradient transforms (clipping, scaling) need.
    pub fn set_gradient(&mut self, tensor_id: TensorId, grad: Tensor) {
        self.gradients.insert(tensor_id, grad);
    }

    /// Get the number of nodes in the graph
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Check if a tensor is in the graph
    pub fn contains_tensor(&self, tensor_id: TensorId) -> bool {
        self.nodes.contains_key(&tensor_id)
    }

    /// Whether `tensor_id` has a node here that has not been released.
    pub fn contains_live(&self, tensor_id: TensorId) -> bool {
        self.live_node(&tensor_id).is_some()
    }

    /// Compute a reverse-topological order (outputs before inputs) of the
    /// whole graph. Diagnostic API: the backward pass itself only orders the
    /// reachable subgraph via [`Self::plan_backward`]. Returns an empty vector
    /// if the graph contains cycles.
    pub fn topological_order(&self) -> Vec<TensorId> {
        // 0 = unvisited, 1 = visiting, 2 = visited
        let mut state: FxHashMap<TensorId, u8> = FxHashMap::default();
        state.reserve(self.nodes.len());
        let mut stack: Vec<(TensorId, usize)> = Vec::with_capacity(self.nodes.len());
        let mut post_order: Vec<TensorId> = Vec::with_capacity(self.nodes.len());

        for &root in self.nodes.keys() {
            if state.get(&root).copied().unwrap_or(0) != 0 {
                continue;
            }
            stack.push((root, 0));
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

                let inputs = match self.nodes.get(current_id) {
                    Some(node) => &node.inputs,
                    None => {
                        state.insert(*current_id, 2);
                        stack.pop();
                        continue;
                    }
                };

                if *idx < inputs.len() {
                    let next = inputs[*idx];
                    *idx += 1;
                    if !self.nodes.contains_key(&next) {
                        continue;
                    }
                    match state.get(&next).copied().unwrap_or(0) {
                        0 => stack.push((next, 0)),
                        1 => return Vec::new(), // cycle
                        2 => {}
                        _ => unreachable!(),
                    }
                } else {
                    state.insert(*current_id, 2);
                    post_order.push(*current_id);
                    stack.pop();
                }
            }
        }

        post_order.reverse();
        post_order
    }

    /// Get a node by tensor ID
    pub fn get_node(&self, tensor_id: TensorId) -> Option<&GraphNode> {
        self.nodes.get(&tensor_id)
    }

    /// Get all nodes that depend on a given tensor
    /// Whether any live node consumes `tensor_id` as an input.
    ///
    /// Cheaper than `get_dependents` for the "is this safe to mutate" question:
    /// it short-circuits and allocates nothing.
    pub fn has_dependents(&self, tensor_id: TensorId) -> bool {
        self.nodes
            .values()
            .any(|node| node.inputs.contains(&tensor_id) && node.is_live())
    }

    pub fn get_dependents(&self, tensor_id: TensorId) -> Vec<TensorId> {
        self.nodes
            .values()
            .filter(|node| node.inputs.contains(&tensor_id) && node.is_live())
            .map(|node| node.tensor_id)
            .collect()
    }

    /// Check if there are cycles in the computation graph
    pub fn has_cycles(&self) -> bool {
        self.topological_order().len() != self.nodes.len()
    }

    /// Remove a tensor and its dependencies from the graph
    pub fn remove_tensor(&mut self, tensor_id: TensorId) {
        if self.nodes.remove(&tensor_id).is_some() {
            // Remove any gradients
            self.gradients.remove(&tensor_id);
        }
    }

    /// Get statistics about the computation graph
    pub fn stats(&self) -> GraphStats {
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
            input_requires_grad: [true, true],
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
            input_requires_grad: [true, true],
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
            input_requires_grad: [true, true],
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
            input_requires_grad: [true, true],
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
            input_requires_grad: [true, true],
        });
        let add_b = Arc::new(AddBackward {
            input_shapes: [vec![1], vec![1]],
            input_ids: [a, a],
            input_requires_grad: [true, true],
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
            input_requires_grad: [true, true],
        });
        graph.add_tensor(c, Some(add_fn));

        // Create gradient tensor
        let grad_shape = Shape::new(vec![2]);
        let grad_c = Tensor::ones(grad_shape, DataType::Float32, Device::cpu(), false);

        // Perform backward pass
        graph.backward(c, Some(grad_c)).unwrap();
        let gradients = graph.gradients_snapshot();

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
            input_requires_grad: [true, true],
        });
        graph.add_tensor(temp1, Some(add_fn1));

        // Second addition: temp2 = a + c
        let temp2 = TensorId::new();
        let add_fn2 = Arc::new(AddBackward {
            input_shapes: [vec![2], vec![2]],
            input_ids: [a, c],
            input_requires_grad: [true, true],
        });
        graph.add_tensor(temp2, Some(add_fn2));

        // Final addition: d = temp1 + temp2
        let d = TensorId::new();
        let add_fn3 = Arc::new(AddBackward {
            input_shapes: [vec![2], vec![2]],
            input_ids: [temp1, temp2],
            input_requires_grad: [true, true],
        });
        graph.add_tensor(d, Some(add_fn3));

        // Create gradient tensor
        let grad_shape = Shape::new(vec![2]);
        let grad_d = Tensor::ones(grad_shape, DataType::Float32, Device::cpu(), false);

        // Perform backward pass
        graph.backward(d, Some(grad_d)).unwrap();
        let gradients = graph.gradients_snapshot();

        // Should have gradients for all tensors
        assert!(gradients.contains_key(&a));
        assert!(gradients.contains_key(&b));
        assert!(gradients.contains_key(&c));

        // 'a' should appear in the gradients (accumulated from both paths)
        assert!(gradients.contains_key(&a));
    }

    #[test]
    fn test_plan_backward_only_visits_reachable_subgraph() {
        let mut graph = ComputationGraph::new();

        // Graph 1: c = a + b
        let a = TensorId::new();
        let b = TensorId::new();
        let c = TensorId::new();
        graph.add_tensor(a, None);
        graph.add_tensor(b, None);
        graph.add_tensor(
            c,
            Some(Arc::new(AddBackward {
                input_shapes: [vec![2], vec![2]],
                input_ids: [a, b],
                input_requires_grad: [true, true],
            })),
        );

        // Unrelated graph 2: z = x + y
        let x = TensorId::new();
        let y = TensorId::new();
        let z = TensorId::new();
        graph.add_tensor(x, None);
        graph.add_tensor(y, None);
        graph.add_tensor(
            z,
            Some(Arc::new(AddBackward {
                input_shapes: [vec![2], vec![2]],
                input_ids: [x, y],
                input_requires_grad: [true, true],
            })),
        );

        let plan = graph.plan_backward(c).unwrap();
        let planned: Vec<TensorId> = plan.iter().map(|s| s.tensor_id).collect();
        assert_eq!(planned.len(), 3);
        assert!(planned.contains(&a) && planned.contains(&b) && planned.contains(&c));
        assert!(!planned.contains(&x) && !planned.contains(&y) && !planned.contains(&z));
        // Reverse-topological: the output comes first.
        assert_eq!(planned[0], c);
    }

    #[test]
    fn test_release_saved_subgraph_drops_interior_nodes_keeps_gradients() {
        use crate::device::Device;
        use crate::tensor::{DataType, Shape, Tensor};

        let mut graph = ComputationGraph::new();
        let a = TensorId::new();
        let b = TensorId::new();
        let c = TensorId::new();
        graph.add_tensor(a, None);
        graph.add_tensor(b, None);
        graph.add_tensor(
            c,
            Some(Arc::new(AddBackward {
                input_shapes: [vec![2], vec![2]],
                input_ids: [a, b],
                input_requires_grad: [true, true],
            })),
        );

        let grad = Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), false);
        graph.backward(c, Some(grad)).unwrap();
        assert!(graph.get_gradient(a).is_some());

        let _ = graph.release_saved_subgraph(c);
        // Interior node (with grad_fn) removed, leaves preserved.
        assert!(!graph.contains_tensor(c));
        assert!(graph.contains_tensor(a));
        assert!(graph.contains_tensor(b));
        // Gradients stay readable after release.
        assert!(graph.get_gradient(a).is_some());
        assert!(graph.get_gradient(b).is_some());
        // Including the interior one, for a non-leaf `.grad` read.
        assert!(graph.get_gradient(c).is_some());
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
            input_requires_grad: [true, true],
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
