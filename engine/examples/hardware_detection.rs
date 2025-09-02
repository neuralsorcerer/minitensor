// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::autograd::ComputationGraph;
use engine::{HardwareProfiler, ResourceOptimizer};

fn main() {
    println!("=== Minitensor Hardware Detection Example ===\n");

    // Create hardware profiler
    let profiler = HardwareProfiler::new();

    // Profile the system
    println!("Profiling system hardware...");
    let hardware_profile = profiler.profile_system();

    // Generate and display report
    let report = profiler.generate_report(&hardware_profile);
    println!("{}", report);

    // Create resource optimizer
    println!("Creating resource optimizer...");
    let optimizer = ResourceOptimizer::new(hardware_profile);

    // Create a simple computation graph for testing
    let computation_graph = ComputationGraph::new();

    // Optimize execution plan
    println!("Optimizing execution plan...");
    let execution_plan = optimizer.optimize_execution(&computation_graph);

    println!("Execution plan created successfully!");
    println!(
        "Estimated execution time: {:?}",
        execution_plan.estimated_total_time
    );
    println!(
        "Parallelization strategy: {:?}",
        execution_plan.parallelization_strategy
    );
    println!(
        "Memory allocation strategy: {:?}",
        execution_plan.memory_plan.allocation_strategy
    );
}
