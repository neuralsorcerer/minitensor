# Custom Operations System

MiniTensor's custom-operations system is currently a Rust-engine extension
point with a small Python execution API. The compiled examples demonstrate how
to register Rust `CustomOp` implementations in the global registry and invoke
them from Python. Python can register and manage plugin metadata, but the
current public Python package does **not** expose a Python `CustomOpBuilder` for
creating new tensor kernels entirely in Python.

## Current public Python API

The top-level `minitensor` package exposes these helpers when the Rust extension
is available:

| Function | Behavior |
| --- | --- |
| `register_example_custom_ops()` | Registers the bundled Rust example operations: `swish`, `gelu`, `mish`, `power`, and `layer_norm`. |
| `list_custom_ops_py()` | Returns registered operation names. |
| `is_custom_op_registered_py(name)` | Checks whether a name is present in the global registry. |
| `execute_custom_op_py(name, inputs)` | Executes a registered operation with a Python list of tensors or tensor wrappers and returns a `Tensor`. |
| `unregister_custom_op_py(name)` | Removes an operation from the global registry. |

Example:

```python
import minitensor as mt

mt.register_example_custom_ops()
assert mt.is_custom_op_registered_py("swish")

x = mt.Tensor([[1.0, 2.0, -1.0]], requires_grad=True)
y = mt.execute_custom_op_py("swish", [x])
print(y.shape)
```

`execute_custom_op_py` accepts either core `Tensor` objects or wrapper objects
with a `_tensor` attribute. The binding returns a tensor object directly; older
examples that manually allocated a wrapper around the returned core tensor are
not required for the current binding.

## Rust engine model

Custom operations implement the `CustomOp` trait. The trait is `Send + Sync` so
registered operations can be shared safely by the global registry.

```rust
pub trait CustomOp: Send + Sync {
    fn name(&self) -> &str;
    fn validate_inputs(&self, inputs: &[&Tensor]) -> Result<()>;
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor>;
    fn create_gradient_function(
        &self,
        inputs: &[&Tensor],
        output: &Tensor,
    ) -> Option<Arc<dyn GradientFunction>>;
    fn num_inputs(&self) -> usize;
    fn output_shape(&self, input_shapes: &[&Shape]) -> Result<Shape>;
    fn output_dtype(&self, input_dtypes: &[DataType]) -> Result<DataType>;
    fn output_device(&self, input_devices: &[&Device]) -> Result<Device>;
}
```

The engine also provides `CustomOpBuilder::new(name, num_inputs)` for Rust code.
A builder can attach forward logic, optional backward logic, validation, and
output metadata inference before calling `build()`.

```rust
let op = CustomOpBuilder::new("my_operation", 2)
    .forward(|inputs| {
        let lhs = inputs[0];
        let rhs = inputs[1];
        arithmetic::add(lhs, rhs)
    })
    .validate(|inputs| {
        if inputs[0].shape() != inputs[1].shape() {
            return Err(MinitensorError::shape_mismatch(
                inputs[0].shape().dims().to_vec(),
                inputs[1].shape().dims().to_vec(),
            ));
        }
        Ok(())
    })
    .build()?;
```

## Bundled example operations

The registered examples are intentionally simple demonstration operations. They
are useful for testing the registry and binding path; they are not promised to
match the fully optimized mathematical layers in `minitensor.nn`.

### `swish`

Forward pass:

$$
\operatorname{swish}(x) = x\,\sigma(x), \qquad
\sigma(x)=\frac{1}{1+e^{-x}}.
$$

For a scalar component, the exact derivative is

$$
\frac{d}{dx}\left[x\sigma(x)\right]
= \sigma(x) + x\sigma(x)(1-\sigma(x)).
$$

The example Rust backward implementation is deliberately simplified and returns
a tensor of ones with the input shape and dtype. Use it as a registry example,
not as a numerically exact training primitive.

### `gelu`

The comments in the Rust example mention the common tanh approximation

$$
\operatorname{GELU}(x) \approx \tfrac{1}{2}x\left(1 + \tanh\left(\sqrt{2/\pi}
(x + 0.044715x^3)\right)\right),
$$

but the demonstration code actually computes

$$
x(1 + \tanh(x)).
$$

This differs by a missing factor of `1/2` and omits the cubic approximation
term. Prefer the built-in `nn.GELU`/tensor GELU operation when you need standard
GELU semantics.

### `mish`

The standard Mish activation is

$$
\operatorname{mish}(x)=x\tanh(\log(1+e^x)).
$$

The bundled example simplifies it to `x * tanh(x)` for demonstration.

### `power`

The operation is named `power`, validates that both inputs have identical shape,
and advertises output dtype promotion between `float32` and `float64`. Its
forward pass is currently simplified to elementwise multiplication of base and
exponent tensors, not exponentiation:

$$
powerExample(a,b)=a\,b.
$$

For true scalar exponentiation $a^b$, the derivatives would be
$\partial a^b/\partial a = b a^{b-1}$ and
$\partial a^b/\partial b = a^b \log a$ where defined. The example backward
path does not implement those formulas.

### `layer_norm`

The example validates that `weight` and `bias` are one-dimensional and match the
last input dimension. Its forward pass currently returns a clone of the input.
A mathematical layer normalization over the final dimension would compute

$$
\mu = \frac{1}{H}\sum_{j=1}^{H} x_j, \qquad
\sigma^2 = \frac{1}{H}\sum_{j=1}^{H}(x_j-\mu)^2,
$$

$$
y_j = \gamma_j\frac{x_j-\mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta_j.
$$

Use `Tensor.layer_norm(...)` or neural-network normalization layers for actual
normalization behavior.

## Registration lifecycle

Rust operations are registered globally by name. Registering the bundled
examples more than once may report duplicate-registration errors depending on
registry state. If a test or script needs a clean state, unregister the names it
registered:

```python
for name in ["swish", "gelu", "mish", "power", "layer_norm"]:
    if mt.is_custom_op_registered_py(name):
        mt.unregister_custom_op_py(name)
```

## Testing guidance

When adding a real Rust custom operation:

- Validate input count, shapes, dtypes, devices, and edge cases explicitly.
- Write Rust unit tests for `validate_inputs`, `output_shape`, and forward
  values.
- Compare gradients against finite differences for differentiable operations.
- Test Python bindings with tensors and wrapper objects.
- Document whether the operation is a pedagogical example or production-ready
  mathematical primitive.
