# OpenCL Backend

The OpenCL backend provides cross-platform GPU acceleration for tensor operations using the OpenCL framework. This backend enables minitensor to run on a wide variety of GPU hardware from different vendors including NVIDIA, AMD, Intel, and others.

## Features

- **Cross-platform GPU support**: Works with GPUs from multiple vendors
- **Comprehensive kernel library**: Optimized kernels for common tensor operations
- **Memory management**: Efficient buffer allocation and data transfer
- **Async execution**: Non-blocking kernel execution with proper synchronization
- **Error handling**: Robust error reporting and recovery

## Supported Operations

### Element-wise Operations

- Addition (`add_kernel`)
- Multiplication (`mul_kernel`)

### Linear Algebra

- Matrix multiplication (`matmul_kernel`)

### Activation Functions

- ReLU (`relu_kernel`)
- Sigmoid (`sigmoid_kernel`)

## Architecture

### Core Components

1. **OpenCLBackend**: Main backend implementation

   - Device initialization and management
   - Memory allocation and deallocation
   - Data transfer between host and device
   - Program compilation and kernel management

2. **OpenCLOps**: High-level operation interface

   - Kernel execution with proper parameter binding
   - Error handling and validation
   - Performance optimization

3. **Memory Management**:
   - Buffer tracking with unique IDs
   - Automatic cleanup on deallocation
   - Zero-copy operations where possible

### Memory Model

The OpenCL backend uses a pointer-based memory model that maps to OpenCL buffer objects:

```rust
// Allocation returns a pointer that maps to a tracked OpenCL buffer
let ptr = backend.allocate(size_bytes)?;

// Operations work with these pointers
ops.add_ptr(a_ptr, b_ptr, result_ptr, n)?;

// Automatic cleanup when deallocated
backend.deallocate(ptr, size_bytes)?;
```

## Usage

### Basic Setup

```rust
use engine::backends::opencl::{OpenCLBackend, OpenCLOps};
use engine::backends::Backend;
use std::sync::Arc;

// Check availability
if !OpenCLBackend::is_available() {
    return Err("OpenCL not available");
}

// Initialize backend
let backend = Arc::new(OpenCLBackend::initialize()?);
let ops = OpenCLOps::new(backend.clone())?;
```

### Element-wise Operations

```rust
// Create input data
let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
let b_data = vec![5.0f32, 6.0, 7.0, 8.0];

// Create OpenCL buffers
let a_buffer = backend.create_buffer_with_data(&a_data, CL_MEM_READ_ONLY)?;
let b_buffer = backend.create_buffer_with_data(&b_data, CL_MEM_READ_ONLY)?;
let c_buffer = backend.create_buffer(4, CL_MEM_WRITE_ONLY)?;

// Execute addition
ops.add(&a_buffer, &b_buffer, &c_buffer, 4)?;

// Read result
let mut result = vec![0.0f32; 4];
backend.read_buffer(&c_buffer, &mut result)?;
```

### Matrix Operations

```rust
// 2x2 matrix multiplication
let a = vec![1.0f32, 2.0, 3.0, 4.0];  // Matrix A
let b = vec![5.0f32, 6.0, 7.0, 8.0];  // Matrix B

let a_buffer = backend.create_buffer_with_data(&a, CL_MEM_READ_ONLY)?;
let b_buffer = backend.create_buffer_with_data(&b, CL_MEM_READ_ONLY)?;
let c_buffer = backend.create_buffer(4, CL_MEM_WRITE_ONLY)?;

// Execute: C = A * B (2x2 * 2x2 = 2x2)
ops.matmul(&a_buffer, &b_buffer, &c_buffer, 2, 2, 2)?;
```

## Performance Considerations

### Kernel Optimization

- Kernels are optimized for coalesced memory access
- Work group sizes are automatically determined
- Local memory is used where beneficial

### Memory Transfer

- Minimize host-device transfers
- Use asynchronous transfers when possible
- Prefer in-place operations

### Error Handling

- All operations return `Result<T>` for proper error handling
- OpenCL errors are mapped to minitensor error types
- Detailed error messages for debugging

## Requirements

### System Requirements

- OpenCL 1.2 or later
- Compatible GPU drivers
- OpenCL runtime libraries

### Rust Dependencies

- `opencl3`: OpenCL bindings for Rust
- Standard library threading primitives

## Testing

The OpenCL backend includes comprehensive tests:

```bash
# Run basic tests
cargo test --features opencl

# Run with specific test
cargo test --features opencl test_opencl_operations

# Run example
cargo run --example opencl_demo --features opencl
```

## Troubleshooting

### Common Issues

1. **OpenCL not available**

   - Install GPU drivers with OpenCL support
   - Verify OpenCL runtime is installed
   - Check `clinfo` output for available devices

2. **Kernel compilation errors**

   - Check OpenCL version compatibility
   - Verify device supports required features
   - Review kernel source for syntax errors

3. **Memory allocation failures**
   - Reduce buffer sizes
   - Check available GPU memory
   - Implement memory pooling for frequent allocations

### Debug Information

Enable debug logging to get detailed information:

```rust
// Check device capabilities
let devices = GpuDevice::detect_all();
for device in devices {
    if device.device_type == DeviceType::OpenCL {
        println!("OpenCL Device: {}", device.name);
        println!("Memory: {} MB", device.memory_size / 1024 / 1024);
        println!("Compute Units: {}", device.max_compute_units);
    }
}
```

## Future Enhancements

- [ ] Kernel fusion for improved performance
- [ ] Support for more data types (fp16, int8)
- [ ] Advanced memory management with pooling
- [ ] Multi-device execution
- [ ] Profiling and performance analysis tools
- [ ] Custom kernel compilation from source

## Contributing

When contributing to the OpenCL backend:

1. Ensure all tests pass on multiple OpenCL devices
2. Add tests for new functionality
3. Update documentation for API changes
4. Follow Rust best practices for unsafe code
5. Verify cross-platform compatibility
