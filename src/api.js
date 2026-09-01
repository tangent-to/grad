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
