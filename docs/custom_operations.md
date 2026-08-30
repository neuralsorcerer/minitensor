# Custom Operations System

An operation MiniTensor does not have can be added from Python, with no Rust
toolchain and no rebuild, and it then participates in autograd on the same terms
as a built-in one. The same registry also holds operations written in Rust
against the `CustomOp` trait, which is what the bundled examples are.

## Registering one from Python

`register_custom_op(name, forward, backward=None, num_inputs=1)` is the
extension point.

`forward` is called with the input tensors as positional arguments and returns a
tensor. `backward`, when given, is called with `(grad_output, inputs, output)`
-- the incoming gradient, a tuple of the saved inputs, and the saved output --
and returns one gradient per input: a bare tensor when there is one input,
otherwise a sequence, in which `None` means no gradient flows to that input.

Whether you write a `backward` decides which of two things you get, and they are
the only two a caller can sensibly mean.

**Without one**, the forward is recorded like any other Python function, and the
operation differentiates by composition:

```python
import minitensor as mt

mt.register_custom_op("scaled_square", lambda x: (x * x) * 0.5)

x = mt.Tensor([1.0, 2.0], dtype="float64", requires_grad=True)
mt.execute_custom_op_py("scaled_square", [x]).sum().backward()
print(x.grad.tolist())  # [1.0, 2.0]

mt.clear_autograd_graph()
mt.unregister_custom_op_py("scaled_square")
```

**With one**, the forward runs with gradient recording *off* and the gradient is
whatever `backward` says. That is what makes a straight-through estimator
possible -- the true derivative of a step function is zero everywhere, and zero
is not the gradient anyone wants:

```python
import minitensor as mt


def step(x):
    return (x > 0.0).astype("float64")


mt.register_custom_op("hard_step", step, lambda grad, inputs, output: grad)

x = mt.Tensor([-1.0, 0.5, 2.0], dtype="float64", requires_grad=True)
mt.execute_custom_op_py("hard_step", [x]).sum().backward()
print(x.grad.tolist())  # [1.0, 1.0, 1.0]

mt.clear_autograd_graph()
mt.unregister_custom_op_py("hard_step")
```

Recording is off in that second case for a reason: were it on, the graph would
hold two paths to the same gradient -- the forward's own and the backward's --
and they would add.

An operation of several inputs declares `num_inputs` and returns a gradient for
each:

```python
import minitensor as mt

mt.register_custom_op(
    "weighted",
    lambda a, b: a * b,
    lambda grad, inputs, output: (grad * inputs[1], grad * inputs[0]),
    num_inputs=2,
)

a = mt.Tensor([1.0, 2.0], dtype="float64", requires_grad=True)
b = mt.Tensor([3.0, 4.0], dtype="float64", requires_grad=True)
mt.execute_custom_op_py("weighted", [a, b]).sum().backward()
print(a.grad.tolist(), b.grad.tolist())  # [3.0, 4.0] [1.0, 2.0]

mt.clear_autograd_graph()
mt.unregister_custom_op_py("weighted")
```

A gradient whose shape or dtype does not match its input is refused rather than
accumulated into a buffer it does not fit, and an exception raised inside either
callable is reported with its own message -- so what you see is a traceback from
your own code.

## The rest of the registry API

| Function | Behavior |
| --- | --- |
| `register_custom_op(name, forward, backward=None, num_inputs=1)` | Registers an operation whose forward and backward are Python callables. A name already taken is refused. |
| `register_example_custom_ops()` | Registers the bundled Rust example operations: `swish`, `gelu`, `mish`, `power`, and `layer_norm`. |
| `list_custom_ops_py()` | Returns registered operation names. |
| `is_custom_op_registered_py(name)` | Checks whether a name is present in the global registry. |
| `execute_custom_op_py(name, inputs)` | Executes a registered operation with a Python list of tensors or tensor wrappers and returns a `Tensor`. |
| `unregister_custom_op_py(name)` | Removes an operation from the global registry. |

```python
import minitensor as mt

mt.register_example_custom_ops()
assert mt.is_custom_op_registered_py("swish")

x = mt.Tensor([[1.0, 2.0, -1.0]], requires_grad=True)
y = mt.execute_custom_op_py("swish", [x])
print(y.shape)
```

The registry is process-wide, so a name is taken until it is unregistered.

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

### Writing a backward pass

`backward` takes a closure receiving a `BackwardContext`, which carries the
incoming gradient together with the inputs and output saved from the forward
pass. Return one gradient per input, keyed by `ctx.input_ids`; omitting an
input's entry means no gradient flows to it.

```rust
pub struct BackwardContext<'a> {
    pub grad_output: &'a Tensor,
    pub inputs: &'a [Tensor],
    pub output: &'a Tensor,
    pub input_ids: &'a [TensorId],
}
```

The saved values are what make a non-linear derivative expressible — it has to
be evaluated at the point the forward pass was computed. For $y = x^3$:

```rust
.backward(|ctx| {
    let mut gradients = FxHashMap::default();
    let (Some(&id), Some(x)) = (ctx.input_ids.first(), ctx.input(0)) else {
        return Ok(gradients);
    };
    // dy/dx = 3x^2, times the incoming gradient (chain rule).
    let sq = arithmetic::mul(x, x)?;
    let two_sq = arithmetic::add(&sq, &sq)?;
    let dydx = arithmetic::add(&two_sq, &sq)?;
    gradients.insert(id, arithmetic::mul(ctx.grad_output, &dydx)?);
    Ok(gradients)
})
```

`ctx.output` is available for derivatives that are cheaper in terms of the
result, such as $\sigma'(x) = y(1-y)$. Convenience accessors `ctx.input(i)`,
`ctx.input_shape(i)`, `ctx.input_dtype(i)`, and `ctx.input_device(i)` return
`None` when the operation received fewer inputs.

Saving the inputs and output copies no data — tensors share their storage — but
it does keep those buffers alive until the graph is released.

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
not as a numerically exact training primitive. The exact derivative above *is*
expressible with the builder — see [Writing a backward pass](#writing-a-backward-pass)
— the bundled examples simply do not implement it.

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
import minitensor as mt

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
