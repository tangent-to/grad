# @tangent.to/grad

Reverse-mode automatic differentiation for JavaScript (ESM). An array-valued
tape over plain nested arrays, with adjoints for the linear algebra that
statistical models are made of.

Write a log-likelihood once; get its exact gradient. No hand-derived
derivatives per model, no finite differences.

The automatic-differentiation leaf of the
[tangent suite](https://github.com/tangent-to) — MIT-licensed infrastructure
consumed by [ds](https://github.com/tangent-to/ds) (Gaussian-process
hyperparameters), [mc](https://github.com/tangent-to/mc) (HMC/NUTS gradients)
and [sem](https://github.com/tangent-to/sem) (the ML discrepancy).

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

**The kernel is a loop, not an element.** Every elementwise op is written as a
whole loop over its operands, called once per pass. The obvious alternative,
passing a per-element function to a shared loop body, sends that call site
megamorphic once a dozen ops share it: the arithmetic stops inlining and an
addition costs upwards of 100 ns instead of well under one. On the regression
above, hoisting the dispatch out of the inner loop was worth more than every
other optimization in this package put together.

**No operator overloading, so `add` takes every term at once.** JavaScript
cannot define `+` on an object, which is how PyMC writes a model mean as
`mu0 + tau * z + gamma`. Nothing here can recover that. What a binary op does
add on top of the language's limit is nesting, and that part is avoidable:
`add` and `mul` take any number of operands and fold left, so a five-term mean
is one call rather than four wrapped ones. The graph is identical either way.
`sub` and `div` stay binary, since `sub(a, b, c)` reads ambiguously.

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

## Cost, and `compile`

`valueAndGrad` rebuilds the graph on every call: a `Var` and a closure per
operation, a topological sort, a fresh gradient buffer per node. On a
340-observation regression with 15 parameters, that bookkeeping was 92% of the
runtime and the arithmetic was the small part. None of it changes between
calls, because the shapes are fixed and the sequence of operations is the same.
Only the values move.

`compile` keeps the graph, writes the new values into its leaves, and replays
it:

```js
import { compile } from '@tangent.to/grad';

const vg = compile((p) => negLogLik(p));
for (const p of chain) vg(p);          // one graph, many evaluations
```

Per gradient on that model, and over a 4-chain 800-iteration NUTS run:

| | ms | full run |
|---|---|---|
| `valueAndGrad` | 0.258 | 54 s |
| `compile` | 0.096 | 20 s |
| gradient derived by hand | 0.047 | 9 s |

The three agree bit for bit.

The constraint is that the graph must be the same on every call. There are two
ways to break that, and both take deliberate effort to write: branching on a
parameter's numeric value by reaching into `.data`, or closing over data that
is mutated between calls. A branch *inside* an op is fine, and is the reason
`relu` and `maximum` exist: the kernel picks a side per element while the graph
stays put. A change in a parameter's shape is detected and rebuilds the plan,
so varying dimensions cost a rebuild rather than a wrong answer.

`valueAndGradFns(f, { compile: true })` opts the mc pair in. It is off by
default because a static graph is an assumption about your objective, and one
this package cannot check for you.

## Where this pays, and where it does not

Measured, not assumed.

**Probabilistic models — yes.** Finite-difference gradients cost 2P likelihood
evaluations per gradient and break the symplectic property leapfrog relies on.
See the table above: 7.7x end to end on a 21-parameter NUTS run, same
posterior, validated against PyMC.

**Structural equation models — yes.** The ML discrepancy
`F = log|Σ| + tr(SΣ⁻¹)` over `Σ(θ) = F(I−A)⁻¹Ψ(I−A)⁻ᵀFᵀ` is a scalar objective
in few parameters: reverse mode's best case. `(I−A)` is not symmetric, which is
why `solveGeneral`/`inv` exist, and only the observed block of Σ is compared
with the data, which is why `slice` does.

**Stiff ODE Jacobians — no.** Tried and measured against `@tangent.to/ode`'s
finite-difference Jacobian on a stiff reaction-diffusion system: identical step
counts, and up to 26x slower at n = 30. A Newton iteration converges to the
same answer with an approximate Jacobian, so the accuracy buys nothing, and a
square Jacobian is reverse mode's worst shape. `jacobian` is still exported and
correct — it pays where outputs are few — but it is not the tool for that job.

## Status

Early. The op set covers what the suite's models need; it is not a deep
learning framework and does not try to be. No GPU backend: WebGPU has no `f64`,
and the numerics here depend on double precision. No forward mode, which is
what a wide Jacobian would want.

## API

| | |
|---|---|
| `variable(x)`, `Var` | tape leaves and nodes |
| `valueAndGrad(f)`, `grad(f)` | differentiate a scalar objective |
| `compile(f)` | the same, reusing the tape across calls; see [Cost](#cost-and-compile) |
| `valueAndGradFns(f, opts)` | the `(fn, gradFn)` pair mc's `potential` takes |
| `add` `mul` | elementwise, scalar broadcasting, and variadic: `add(a, b, c, d)` |
| `sub` `div` `neg` | the same, strictly binary |
| `exp` `log` `sqrt` `square` `pow` `tanh` `sigmoid` | elementwise functions |
| `maximum` `minimum` `relu` | elementwise clamps. At a tie the adjoint goes to the left operand; `relu'(0) = 0` |
| `sum` `mean` | reductions to a scalar |
| `matmul` `dot` `transpose` `reshape` `slice` `concat` `diagPart` `trace` `addDiag` | array manipulation |
| `cholesky` `triangularSolve` `logdetPSD` `solvePSD` | differentiable linear algebra, symmetric positive definite |
| `solveGeneral` `inv` | the same for a general square matrix (LU) |
| `jacobian(f)` | ∂f/∂x for a vector-valued f |

`valueAndGrad` accepts either a plain array (what an optimizer passes) or a
`{name: value}` map (what a probabilistic model passes), and returns the
gradient in the same shape.

## License

MIT
