// ---
// title: Exact gradients without deriving them
// id: grad-gradients
// ---

// %% [markdown]
/*
`@tangent.to/grad` differentiates a function you wrote, exactly, by recording
the operations it performs and walking them backwards. Write a log-likelihood
once and its gradient comes for free — no hand-derived formulas per model, and
no finite differences.

Nodes on the tape are matrices and vectors, never individual numbers. That is
the design constraint that makes it viable in JavaScript: the per-node
bookkeeping is noise beside an O(n³) Cholesky, and ruinous beside a scalar
multiply.
*/

// %% [javascript]

import * as __lib from 'https://esm.sh/@tangent.to/grad';
const valueAndGrad = __lib.valueAndGrad;
const jacobian = __lib.jacobian;
const add = __lib.add;
const sub = __lib.sub;
const mul = __lib.mul;
const div = __lib.div;
const exp = __lib.exp;
const log = __lib.log;
const square = __lib.square;
const sum = __lib.sum;
const dot = __lib.dot;
const concat = __lib.concat;
const slice = __lib.slice;
const matmul = __lib.matmul;
const addDiag = __lib.addDiag;
const solvePSD = __lib.solvePSD;
const logdetPSD = __lib.logdetPSD;
const cholesky = __lib.cholesky;

// A scalar objective of two named parameters. `valueAndGrad` returns both the
// value and the gradient, and the gradient arrives in the shape the parameters
// went in — here a {name: value} map.
const simple = valueAndGrad((p) => add(square(p.mu), square(p.sigma)));
simple({ mu: 3, sigma: 4 });

// %% [markdown]
/*
## A least-squares objective

The gradient of a sum of squared residuals, against the closed form
`-2 Xᵀ(y - Xβ)` that anyone would derive by hand. They agree to machine
precision — which is the point: the tape is not an approximation.
*/

// %% [javascript]

const X = [[1, 0.2], [1, 0.8], [1, 1.6], [1, 2.4], [1, 3.1]];
const y = [1.1, 2.0, 3.2, 4.1, 5.4];

const rss = valueAndGrad((beta) => sum(square(sub(y, matmul(X, beta)))));
const at = [0.5, 1.3];
const { value, gradient } = rss(at);

// the same gradient, derived by hand
const resid = X.map((row, i) => y[i] - (row[0] * at[0] + row[1] * at[1]));
const byHand = [0, 1].map((j) => -2 * X.reduce((s, row, i) => s + row[j] * resid[i], 0));

({ value, gradient, byHand });

// %% [markdown]
/*
## Differentiable linear algebra

This is what separates a statistics autograd from a toy one. A Gaussian
process log marginal likelihood needs the derivative of `log|K|` and of
`yᵀK⁻¹y` with respect to the kernel's hyperparameters:

$$\log p(y \mid \theta) = -\tfrac{1}{2} y^\top K^{-1} y - \tfrac{1}{2}\log|K| - \tfrac{n}{2}\log 2\pi$$

Only `cholesky` and `triangularSolve` carry a hand-written adjoint. The
log-determinant and the solve *compose* from them, so they inherit correct
derivatives with no second formula to get wrong.
*/

// %% [javascript]

// Squared distances between five one-dimensional inputs: a constant of the
// problem, not a parameter.
const xs = [0, 0.6, 1.4, 2.3, 3.0];
const D2 = xs.map((a) => xs.map((b) => (a - b) ** 2));
const yObs = xs.map((x) => Math.sin(x));

// A squared-exponential kernel, its noise, and the likelihood, written once.
const logML = (p) => {
  const K = addDiag(mul(p.v, exp(mul(-0.5, div(D2, square(p.l))))), 0.05);
  return sub(mul(-0.5, dot(yObs, solvePSD(K, yObs))), mul(0.5, logdetPSD(K)));
};

const gp = valueAndGrad(logML)({ l: 1.2, v: 0.9 });
gp;

// %% [markdown]
/*
Swap the kernel expression for a rational quadratic, a Matérn, a sum of
several — the gradients follow. Nothing above needs rederiving, which is the
whole argument for having this layer at all.

## Checking against finite differences

The crude method this package exists to replace, kept here as a check. Agreement
to ~1e-8 on a smooth objective is what a correct adjoint looks like; the tape is
exact and the difference you see is the finite differencing's own error.
*/

// %% [javascript]

const fd = (f, x, h = 1e-6) =>
  Object.fromEntries(Object.keys(x).map((k) => [
    k,
    (f({ ...x, [k]: x[k] + h }) - f({ ...x, [k]: x[k] - h })) / (2 * h),
  ]));

const point = { l: 1.2, v: 0.9 };
const numeric = fd((p) => valueAndGrad(logML)(p).value, point);

({ tape: gp.gradient, finiteDifferences: numeric });

// %% [markdown]
/*
## Jacobians

`jacobian` differentiates a vector-valued function, one reverse sweep per
output over a single forward pass. Here the Van der Pol oscillator's right-hand
side, against its hand-derived Jacobian.
*/

// %% [javascript]

const MU = 4;
const vdp = (v) => {
  const y0 = slice(v, [0], [1]);
  const y1 = slice(v, [1], [1]);
  return concat([y1, mul(MU, sub(mul(sub(1, square(y0)), y1), y0))]);
};

const point2 = [2, 0.7];
const J = jacobian(vdp)(point2);
const exact = [
  [0, 1],
  [MU * (-2 * point2[0] * point2[1] - 1), MU * (1 - point2[0] ** 2)],
];

({ jacobian: J, exact });

// %% [markdown]
/*
Reverse mode wants few outputs and many inputs. A square Jacobian like this one
is its worst shape — for a stiff ODE solver, finite differences are genuinely
the better tool, and the package's docs say so rather than leaving you to find
out. Use this where the outputs are few: a delta-method standard error, the
sensitivity of a handful of summaries to many inputs.
*/
