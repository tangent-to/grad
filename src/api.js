/**
 * High-level differentiation API.
 *
 * `valueAndGrad(f)` turns a scalar objective written in this package's ops into
 * a function returning both its value and its exact gradient. Two argument
 * shapes are accepted, because the suite has two callers:
 *
 *   - a plain array or number — what an optimizer (`@tangent.to/opt`) passes;
 *   - a `{name: number|number[]}` map — what a probabilistic model
 *     (`@tangent.to/mc`) passes, and what its samplers expect back.
 *
 * The gradient comes back in the same shape it went in, so it drops straight
 * into a leapfrog step or an L-BFGS iteration with no marshalling.
 */

import { Var, variable } from './tape.js';
import { shapeStr, toNested } from './tensor.js';

/** Is this a `{name: value}` parameter map rather than a single value? @private */
function isParamMap(x) {
  return x !== null && typeof x === 'object' && !Array.isArray(x) &&
    !(x instanceof Float64Array) && !(x instanceof Var);
}

/**
 * Read a leaf's accumulated gradient back into the shape its input arrived in.
 * @private
 */
function gradOf(v, original) {
  if (v.grad === null) {
    // The objective never touched this parameter. A zero gradient is the honest
    // answer, and silently omitting the key would break a sampler that indexes
    // every parameter by name.
    return typeof original === 'number' ? 0 : new Array(v.value.data.length).fill(0);
  }
  const t = { data: v.grad, shape: v.shape };
  return toNested(t);
}

/**
 * Differentiate a scalar objective, returning both value and gradient.
 *
 * @param {(x: any) => Var} f - objective, built from this package's ops. It
 *   receives `Var`s in the same structure as the input and must return a
 *   scalar `Var`.
 * @returns {(x: any) => { value: number, gradient: any }}
 *
 * @example
 * const f = (p) => add(square(p.mu), square(p.sigma));
 * valueAndGrad(f)({ mu: 3, sigma: 4 });
 * // { value: 25, gradient: { mu: 6, sigma: 8 } }
 */
export function valueAndGrad(f) {
  if (typeof f !== 'function') throw new Error('valueAndGrad: expected a function');

  return (x) => {
    let wrapped;
    let leaves;
    if (isParamMap(x)) {
      wrapped = {};
      leaves = {};
      for (const [k, v] of Object.entries(x)) {
        leaves[k] = variable(v, `parameter "${k}"`);
        wrapped[k] = leaves[k];
      }
    } else {
      leaves = variable(x, 'parameter');
      wrapped = leaves;
    }

    const out = f(wrapped);
    if (!(out instanceof Var)) {
      throw new Error(
        'valueAndGrad: the objective must return a Var built from this package\'s ops; ' +
          `got ${out === null ? 'null' : typeof out}. A plain number means the ops were ` +
          'bypassed somewhere, which breaks the chain.',
      );
    }
    if (!out.isScalar) {
      throw new Error(
        `valueAndGrad: the objective must return a scalar, got ${shapeStr(out.shape)}. ` +
          'Reduce it with sum() or mean() first.',
      );
    }

    out.backward();

    let gradient;
    if (isParamMap(x)) {
      gradient = {};
      for (const k of Object.keys(x)) gradient[k] = gradOf(leaves[k], x[k]);
    } else {
      gradient = gradOf(leaves, x);
    }
    return { value: out.data[0], gradient };
  };
}

/**
 * Gradient only, discarding the objective's value.
 *
 * @param {(x: any) => Var} f
 * @returns {(x: any) => any} gradient, shaped like the input
 */
export function grad(f) {
  const vg = valueAndGrad(f);
  return (x) => vg(x).gradient;
}

/** Structural equality over a `{name: number|number[]}` parameter map. @private */
function sameParams(a, b) {
  if (a === undefined || b === undefined) return false;
  if (typeof a === 'number') return a === b;
  if (Array.isArray(a) || a instanceof Float64Array) {
    if (!(Array.isArray(b) || b instanceof Float64Array) || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }
  if (a === null || typeof a !== 'object') return false;
  const ka = Object.keys(a);
  if (ka.length !== Object.keys(b).length) return false;
  for (const k of ka) if (!sameParams(a[k], b[k])) return false;
  return true;
}

/** Defensive copy of a parameter map, so a caller mutating in place cannot
 * poison the cache below. @private */
function copyParams(x) {
  if (typeof x === 'number') return x;
  if (Array.isArray(x) || x instanceof Float64Array) return Array.from(x);
  const out = {};
  for (const [k, v] of Object.entries(x)) out[k] = copyParams(v);
  return out;
}

/**
 * Split an objective into the SEPARATE value and gradient functions that an
 * API taking a `(fn, gradFn)` pair expects — `@tangent.to/mc`'s
 * `model.potential(name, fn, gradFn)` is the case this exists for.
 *
 * The two share one evaluation: calling `.value(p)` then `.gradient(p)` on the
 * same parameters runs the tape once, not twice. That matters because a
 * sampler's value-and-gradient path calls both in turn, and the forward pass
 * is a full sweep over the data.
 *
 * The cache holds exactly one entry and compares parameters structurally
 * against a defensive copy, so mutating a parameter array in place invalidates
 * it correctly rather than returning a stale gradient.
 *
 * @param {(x: any) => Var} f - objective built from this package's ops
 * @returns {{ value: (x:any) => number, gradient: (x:any) => any }}
 *
 * @example
 * const { value, gradient } = valueAndGradFns((p) => logLik(p));
 * model.potential('y', value, gradient);
 */
export function valueAndGradFns(f) {
  const vg = valueAndGrad(f);
  let lastInput;
  let lastResult;

  const evaluate = (x) => {
    if (lastResult !== undefined && sameParams(lastInput, x)) return lastResult;
    lastResult = vg(x);
    lastInput = copyParams(x);
    return lastResult;
  };

  return {
    value: (x) => evaluate(x).value,
    gradient: (x) => evaluate(x).gradient,
  };
}

/**
 * Jacobian of a VECTOR-valued function: `J[i][j] = ∂f(x)ᵢ / ∂xⱼ`.
 *
 * Cost is one forward pass plus one reverse pass per OUTPUT — the tape is
 * built once and seeded m times.
 *
 * DO NOT reach for this to supply a stiff ODE solver's ∂f/∂y. It was written
 * for that and measured against `@tangent.to/ode`'s finite-difference
 * Jacobian on a stiff reaction-diffusion system; it lost, and lost worse as
 * the system grew:
 *
 *     n     FD      exact     steps (FD / exact)
 *     2    15 ms    27 ms          175 / 175
 *    10    23 ms   218 ms          171 / 171
 *    30    60 ms  1559 ms          171 / 171
 *
 * The step counts are identical, which is the whole story: a Newton iteration
 * converges to the same answer with an approximate Jacobian — the residual is
 * still evaluated exactly — so finite-difference error costs nothing there.
 * Meanwhile a square Jacobian is the worst case for reverse mode: n sweeps
 * over an n-node graph, against n+1 evaluations of cheap scalar arithmetic.
 * Forward mode, or finite differences with sparsity colouring, is the right
 * tool for that shape.
 *
 * Reverse mode pays when outputs are FEW and the map to them is expensive —
 * a delta-method standard error, the sensitivity of a handful of summaries to
 * many inputs.
 *
 * @param {(x: Var) => Var} f - vector-valued function built from these ops
 * @returns {(x: number[]) => number[][]} m × n Jacobian
 *
 * @example
 * const J = jacobian((y) => stack([mul(-2, y0(y)), sub(y0(y), y1(y))]));
 */
export function jacobian(f) {
  if (typeof f !== 'function') throw new Error('jacobian: expected a function');

  return (x) => {
    const leaf = variable(x, 'input');
    const out = f(leaf);
    if (!(out instanceof Var)) {
      throw new Error('jacobian: the function must return a Var built from this package\'s ops');
    }
    if (out.shape.length > 1) {
      throw new Error(
        `jacobian: the function must return a scalar or a vector, got ${shapeStr(out.shape)}`,
      );
    }
    const m = out.value.data.length;
    const n = leaf.value.data.length;
    const J = new Array(m);
    const seed = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      // One reverse sweep per output row, reusing the forward pass. backward()
      // reallocates every node's gradient, so the sweeps do not contaminate
      // each other.
      seed.fill(0);
      seed[i] = 1;
      out.backward(seed);
      J[i] = leaf.grad === null ? new Array(n).fill(0) : Array.from(leaf.grad);
    }
    return J;
  };
}
