# @tangent.to/grad

Reverse-mode automatic differentiation for JavaScript (ESM). An array-valued
tape over plain nested arrays, with adjoints for the linear algebra that
statistical models are made of.

Write a log-likelihood once; get its exact gradient. No hand-derived
derivatives per model, no finite differences.

```js
import { valueAndGrad, add, square } from '@tangent.to/grad';

const f = (p) => add(square(p.mu), square(p.sigma));
valueAndGrad(f)({ mu: 3, sigma: 4 });
// { value: 25, gradient: { mu: 6, sigma: 8 } }
```

## Why

Gradients are the bottleneck in the tangent suite, in two distinct ways.

**Correctness.** Hamiltonian Monte Carlo and NUTS need ∇log p. Approximating it
by central differences costs 2·(#params) full likelihood evaluations *per
leapfrog step*, and — the more serious problem — the leapfrog integrator is no
longer symplectic, so the sampler's acceptance rate degrades or it biases
silently. Neither Stan nor PyMC will use finite differences for this.

**Coverage.** Hand-derived gradients get written for the cases someone had time
for. In `@tangent.to/ds`, the Gaussian process has ~90 lines of kernel-specific
closed forms covering only the Matérn family; every other kernel drops to a
derivative-free search that stops at a visibly worse optimum. The quality of
your fit depends on which kernel you happened to pick.

Both dissolve into the same missing layer.

## Design

**Array-valued, not scalar-valued.** Nodes are matrices and vectors, never
individual numbers. In JavaScript the per-node bookkeeping costs a few hundred
nanoseconds — noise beside an O(n³) Cholesky, ruinous beside a scalar multiply.
A micrograd-style scalar tape cannot carry a statistical model at useful sizes.

**Two hand-written linalg adjoints, and no more.** Only `cholesky` and
`triangularSolve` carry a derived adjoint. Log-determinants, SPD solves,
quadratic forms and Gaussian log-likelihoods all *compose* from those, and so
inherit correct derivatives for free. Every hand-derived formula is a place to
be subtly wrong, so there are as few as the problem allows.

**Rank capped at 2.** Scalars, vectors, matrices. Every model in the suite is
expressed in those; rank-N would cost broadcasting complexity in every adjoint
for no consumer.

Forward factorizations come from [`@tangent.to/lina`](https://github.com/tangent-to/lina),
which is scipy-validated.

## The motivating example

A Gaussian process log marginal likelihood, gradients included:

```js
import { addDiag, dot, exp, div, mul, neg, square, sub, solvePSD, logdetPSD, valueAndGrad } from '@tangent.to/grad';

const logML = (p) => {
  const K = addDiag(mul(p.v, exp(neg(div(D2, mul(2, square(p.l)))))), 0.05);
  return sub(mul(-0.5, dot(y, solvePSD(K, y))), mul(0.5, logdetPSD(K)));
};

const { value, gradient } = valueAndGrad(logML)({ l: 1.3, v: 0.9 });
// gradient.l and gradient.v, exact
```

Swap the kernel expression and the gradients follow. No derivation.

## Validation

Every adjoint is checked two ways, because finite differences alone would not
catch an adjoint that is systematically wrong but smooth:

- central finite differences on the composed objective;
- closed forms known exactly — `d log|A| / dA = A⁻¹`, and for the Gaussian
  process the trace identity `∂L/∂θ = ½ tr((ααᵀ − K⁻¹) ∂K/∂θ)`, computed
  independently with lina.

The GP gradients agree with the trace identity to 9 decimal places, for both a
squared-exponential and a rational-quadratic kernel.

## Wiring into a probabilistic model

`valueAndGradFns` returns the `(fn, gradFn)` pair that
[`@tangent.to/mc`](https://github.com/tangent-to/mc)'s `potential` takes, so a
likelihood written in these ops is differentiated exactly instead of by finite
differences:

```js
const { value, gradient } = valueAndGradFns((p) => {
  const mu = add(mul(p.slope, X), p.intercept);
  const z = div(sub(Y, mu), p.sigma);
  return sub(mul(-0.5, sum(square(z))), mul(N, log(p.sigma)));
});
model.potential('y', value, gradient);
```

The two functions share one evaluation, so the sampler's value-and-gradient
path sweeps the data once rather than twice.

Measured on a multiple regression with 300 observations, against mc's
finite-difference fallback:

| free params | FD evaluations per gradient | autograd | wall-clock speedup |
|---|---|---|---|
| 3 | 6 | 1 | 1.4× |
| 6 | 12 | 1 | 1.6× |
| 11 | 22 | 1 | 4.6× |
| 21 | 42 | 1 | 6.9× |

Autograd's cost is flat in the parameter count; finite differences grow as 2P.
The speedup is smaller than 2P because one taped forward-and-backward costs
several bare forward passes.

Accuracy matters more than either. Against a hand-derived closed form, the
autograd gradient agrees to ~1e-13; central differences are off by ~2e-7. That
error is what breaks the symplectic property leapfrog integration relies on.

## Status

Early. The op set covers what the suite's models need; it is not a deep
learning framework and does not try to be. No GPU backend: WebGPU has no `f64`,
and the numerics here depend on double precision.

## API

| | |
|---|---|
| `variable(x)`, `Var` | tape leaves and nodes |
| `valueAndGrad(f)`, `grad(f)` | differentiate a scalar objective |
| `add` `sub` `mul` `div` `neg` | elementwise arithmetic, scalar broadcasting |
| `exp` `log` `sqrt` `square` `pow` `tanh` `sigmoid` | elementwise functions |
| `sum` `mean` | reductions to a scalar |
| `matmul` `dot` `transpose` `reshape` `diagPart` `trace` `addDiag` | array manipulation |
| `cholesky` `triangularSolve` `logdetPSD` `solvePSD` | differentiable linear algebra |

`valueAndGrad` accepts either a plain array (what an optimizer passes) or a
`{name: value}` map (what a probabilistic model passes), and returns the
gradient in the same shape.

## License

MIT
