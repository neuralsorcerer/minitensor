# Metal Backend

The Metal backend provides high-performance GPU acceleration for tensor operations on Apple devices using Apple's Metal framework. This backend is specifically optimized for Apple Silicon and Intel-based Macs, taking advantage of the unified memory architecture and Metal Performance Shaders.

## Features

- **Apple Silicon Optimization**: Leverages unified memory architecture for zero-copy operations
- **High Performance**: Optimized compute shaders for tensor operations
- **Automatic Thread Group Sizing**: Dynamic optimization based on device capabilities
- **Memory Efficiency**: Shared memory mode for optimal performance
- **Comprehensive Operations**: Full suite of tensor and activation functions

## Supported Operations

### Element-wise Operations

- Addition (`add_kernel`)
- Multiplication (`mul_kernel`)

### Linear Algebra

- Matrix multiplication (`matmul_kernel`) with 2D thread groups

### Activation Functions

- ReLU (`relu_kernel`)
- Sigmoid (`sigmoid_kernel`)

## Architecture

### Core Components

1. **MetalBackend**: Main backend implementation

   - Device initialization and management
   - Memory allocation with unified memory support
   - Compute pipeline management and caching
   - Optimal thread group size calculation

2. **MetalOps**: High-level operation interface

   - Shader compilation and pipeline creation
   - Optimized kernel execution
   - Performance-aware thread group sizing

3. **Memory Management**:
   - Buffer tracking with unique IDs
   - Shared memory mode for zero-copy operations
   - Automatic cleanup and ARC integration

### Memory Model

The Metal backend uses Apple's unified memory architecture for optimal performance:

```rust
// Allocation uses shared memory mode for zero-copy access
let ptr = backend.allocate(size_bytes)?;

// Operations work directly with shared memory
ops.add_ptr(a_ptr, b_ptr, result_ptr, n)?;

// Automatic cleanup with ARC
backend.deallocate(ptr, size_bytes)?;
```

## Usage

### Basic Setup

```rust
use engine::backends::metal::{MetalBackend, MetalOps};
use engine::backends::Backend;
use std::sync::Arc;

// Check availability (macOS only)
if !MetalBackend::is_available() {
    return Err("Metal not available");
}

// Initialize backend
let backend = Arc::new(MetalBackend::initialize()?);
let ops = MetalOps::new(backend.clone())?;
```

### Element-wise Operation

```rust
// Create input data
let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
let b_data = vec![5.0f32, 6.0, 7.0, 8.0];

// Create Metal buffers with shared memory
let a_buffer = backend.create_buffer_with_data(&a_data,
    metal::MTLResourceOptions::StorageModeShared)?;
let b_buffer = backend.create_buffer_with_data(&b_data,
    metal::MTLResourceOptions::StorageModeShared)?;
let c_buffer = backend.create_buffer(16,
    metal::MTLResourceOptions::StorageModeShared)?;

// Execute addition
ops.add(&a_buffer, &b_buffer, &c_buffer, 4)?;

// Read result (zero-copy with shared memory)
let mut result = vec![0.0f32; 4];
backend.copy_buffer_to_host(&c_buffer, &mut result)?;
```

### Matrix Operations

```rust
// 2x2 matrix multiplication
let a = vec![1.0f32, 2.0, 3.0, 4.0];  // Matrix A
let b = vec![5.0f32, 6.0, 7.0, 8.0];  // Matrix B

let a_buffer = backend.create_buffer_with_data(&a,
    metal::MTLResourceOptions::StorageModeShared)?;
let b_buffer = backend.create_buffer_with_data(&b,
    metal::MTLResourceOptions::StorageModeShared)?;
let c_buffer = backend.create_buffer(16,
    metal::MTLResourceOptions::StorageModeShared)?;

// Execute: C = A * B (2x2 * 2x2 = 2x2)
ops.matmul(&a_buffer, &b_buffer, &c_buffer, 2, 2, 2)?;
```

## Performance Optimizations

### Apple Silicon Advantages

- **Unified Memory**: Zero-copy operations between CPU and GPU
- **High Memory Bandwidth**: Up to 400GB/s on M1 Ultra
- **Efficient Scheduling**: Hardware-accelerated command submission
- **Power Efficiency**: Optimized for mobile and desktop use

### Thread Group Optimization

```rust
// Automatic optimal sizing based on device capabilities
let thread_group_size = backend.optimal_thread_group_size(&pipeline);
let thread_group_size_2d = backend.optimal_thread_group_size_2d(&pipeline);

// Considers:
// - Max threads per threadgroup
// - Thread execution width (SIMD width)
// - Memory access patterns
// - Occupancy optimization
```

### Memory Strategies

- **Shared Memory Mode**: Default for optimal CPU-GPU sharing
- **Private Memory**: For GPU-only computations
- **Managed Memory**: Automatic synchronization when needed

## Requirements

### System Requirements

- macOS 10.13+ (Metal 2.0)
- Apple Silicon (M1/M2/M3) or Intel Mac with discrete GPU
- Xcode Command Line Tools

### Rust Dependencies

- `metal`: Apple's Metal framework bindings
- Standard library threading primitives

## Testing

The Metal backend includes comprehensive tests:

```bash
# Run basic tests (macOS only)
cargo test --features metal

# Run with specific test
cargo test --features metal test_metal_operations

# Run example (macOS only)
cargo run --example metal_demo --features metal
```

## Apple Silicon Specific Features

### Unified Memory Architecture

- Zero-copy data sharing between CPU and GPU
- Reduced memory bandwidth requirements
- Simplified memory management

### Performance Characteristics

- **M1**: 8-core GPU, 68.25 GB/s memory bandwidth
- **M1 Pro**: 14/16-core GPU, 200 GB/s memory bandwidth
- **M1 Max**: 24/32-core GPU, 400 GB/s memory bandwidth
- **M2**: 8/10-core GPU, 100 GB/s memory bandwidth
- **M3**: Up to 40-core GPU, enhanced ray tracing

### Optimization Tips

1. **Use Shared Memory**: Always prefer `StorageModeShared` for CPU-GPU data
2. **Batch Operations**: Combine multiple small operations into larger kernels
3. **Thread Group Sizing**: Use automatic sizing for optimal occupancy
4. **Memory Coalescing**: Ensure contiguous memory access patterns

## Troubleshooting

### Common Issues

1. **Metal not available**

   - Ensure running on macOS
   - Check Metal support: `system_profiler SPDisplaysDataType`
   - Update to latest macOS version

2. **Shader compilation errors**

   - Verify Metal Shading Language syntax
   - Check compute capability requirements
   - Review shader source for typos

3. **Performance issues**
   - Use shared memory mode for frequent CPU-GPU transfers
   - Optimize thread group sizes
   - Profile with Instruments.app

### Debug Information

Enable debug logging and profiling:

```rust
// Check device capabilities
let metal_device = backend.metal_device();
println!("Device: {}", metal_device.name());
println!("Max threads per threadgroup: {}",
         metal_device.max_threads_per_threadgroup().width);

// Check pipeline performance
if let Some(pipeline) = backend.get_compute_pipeline("add_kernel") {
    println!("Thread execution width: {}", pipeline.thread_execution_width());
    println!("Max total threads: {}", pipeline.max_total_threads_per_threadgroup());
}
```

### Instruments Integration

Use Xcode Instruments for detailed profiling:

1. **GPU Timeline**: Analyze command buffer execution
2. **Metal System Trace**: Track memory usage and transfers
3. **Compute Performance**: Optimize thread group utilization

## Future Enhancements

- [ ] Metal Performance Shaders (MPS) integration
- [ ] Support for more data types (fp16, int8)
- [ ] Advanced memory management with memory pools
- [ ] Multi-GPU support for Mac Pro
- [ ] Ray tracing acceleration (M3+)
- [ ] Neural Engine integration for ML workloads

## Contributing

When contributing to the Metal backend:

1. Ensure compatibility with all supported macOS versions
2. Test on both Apple Silicon and Intel Macs
3. Follow Metal best practices for shader development
4. Use Instruments for performance validation
5. Maintain compatibility with unified memory architecture

## Comparison with Other Backends

| Feature     | Metal                      | CUDA                | OpenCL          |
| ----------- | -------------------------- | ------------------- | --------------- |
| Platform    | macOS only                 | NVIDIA GPUs         | Cross-platform  |
| Memory      | Unified (zero-copy)        | Discrete            | Varies          |
| Performance | Excellent on Apple Silicon | Excellent on NVIDIA | Good            |
| Development | Apple tools                | NVIDIA tools        | Vendor-neutral  |
| Ecosystem   | Apple-focused              | Broad ML/HPC        | General compute |

The Metal backend is the optimal choice for Apple devices, providing the best performance and integration with the Apple ecosystem.
