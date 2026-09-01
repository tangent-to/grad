/**
 * Reverse-mode tape.
 *
 * Define-by-run: every op builds a `Var` holding its result, its parents, and a
 * closure returning the gradient contribution to each parent. `backward()`
 * walks the graph in reverse topological order and accumulates.
 *
 * Nodes are ARRAY-valued, not scalar-valued. That is the design constraint
 * behind the whole package: in JavaScript the per-node bookkeeping costs a few
 * hundred nanoseconds, which is noise beside an O(n³) Cholesky or an O(n²)
 * matmul, and ruinous beside a scalar multiply. A scalar-valued tape
 * (micrograd-style) cannot carry a statistical model at useful sizes.
 */

import { asTensor, sameShape, shapeStr, sizeOf } from './tensor.js';

/** A node in the computation graph. */
export class Var {
  /**
   * @param {import('./tensor.js').Tensor} value - forward value
   * @param {Var[]} [parents] - inputs this node was computed from
   * @param {(g: Float64Array) => Array<Float64Array|null>} [backward] - given the
   *   gradient of the root w.r.t. this node's value, return the contribution to
   *   each parent, in `parents` order. `null` skips a parent (e.g. an integer
   *   index argument), which is cheaper than allocating a zero buffer.
   */
  constructor(value, parents = [], backward = null) {
    this.value = value;
    this.parents = parents;
    this._backward = backward;
    /** @type {Float64Array|null} Accumulated gradient, filled by backward(). */
    this.grad = null;
  }

  get shape() {
    return this.value.shape;
  }

  get data() {
    return this.value.data;
  }

  /** True for a rank-0 node, the only kind `backward()` can seed on its own. */
  get isScalar() {
    return this.value.shape.length === 0;
  }

  /**
   * Accumulate gradients through the graph, from this node backwards.
   *
   * @param {Float64Array|number[]} [seed] - gradient of the objective w.r.t.
   *   this node. Required unless the node is a scalar, where it defaults to 1.
   * @returns {this}
   */
  backward(seed = null) {
    if (seed === null) {
      if (!this.isScalar) {
        throw new Error(
          `backward() needs an explicit seed for a ${shapeStr(this.shape)} node; ` +
            'only a scalar objective can seed itself with 1.',
        );
      }
      seed = Float64Array.of(1);
    } else {
      seed = seed instanceof Float64Array ? seed : Float64Array.from(seed);
      if (seed.length !== this.value.data.length) {
        throw new Error(
          `seed has ${seed.length} elements but this node holds ${this.value.data.length}`,
        );
      }
    }

    // Reverse topological order, iteratively: a deep graph (an unrolled ODE
    // solve, a long IRLS chain) would blow the call stack with recursion.
    const order = [];
    const seen = new Set();
    const stack = [[this, false]];
    while (stack.length) {
      const [node, expanded] = stack.pop();
      if (expanded) {
        order.push(node);
        continue;
      }
      if (seen.has(node)) continue;
      seen.add(node);
      stack.push([node, true]);
      for (const p of node.parents) if (!seen.has(p)) stack.push([p, false]);
    }

    for (const node of order) node.grad = new Float64Array(node.value.data.length);
    this.grad = seed.slice();

    for (let i = order.length - 1; i >= 0; i--) {
      const node = order[i];
      if (!node._backward) continue; // leaf
      const contribs = node._backward(node.grad);
      for (let k = 0; k < node.parents.length; k++) {
        const c = contribs[k];
        if (!c) continue;
        const g = node.parents[k].grad;
        for (let j = 0; j < g.length; j++) g[j] += c[j];
      }
    }
    return this;
  }
}

/**
 * Wrap a value as a leaf of the tape.
 *
 * @param {number|number[]|number[][]|Float64Array|import('./tensor.js').Tensor} x
 * @param {string} [name='value'] - argument name for error messages
 * @returns {Var}
 */
export function variable(x, name = 'value') {
  return new Var(asTensor(x, name));
}

/**
 * Build a Var from a forward value plus a backward closure. Every op goes
 * through here, so the shape bookkeeping lives in one place.
 *
 * @param {import('./tensor.js').Tensor} value
 * @param {Var[]} parents
 * @param {(g: Float64Array) => Array<Float64Array|null>} backward
 * @returns {Var}
 */
export function node(value, parents, backward) {
  if (value.data.length !== sizeOf(value.shape)) {
    throw new Error(`op produced ${value.data.length} elements for shape ${shapeStr(value.shape)}`);
  }
  return new Var(value, parents, backward);
}

/**
 * Coerce an op argument to a Var, wrapping raw numbers and arrays as constants.
 * Lets ops be written as `add(x, 1)` without the caller lifting the 1.
 *
 * @param {Var|number|number[]|number[][]|Float64Array} x
 * @param {string} [name='argument']
 * @returns {Var}
 */
export function toVar(x, name = 'argument') {
  return x instanceof Var ? x : new Var(asTensor(x, name));
}

/**
 * Assert two operands have the same shape, for ops that do not broadcast.
 * @param {Var} a
 * @param {Var} b
 * @param {string} op
 */
export function requireSameShape(a, b, op) {
  if (!sameShape(a.shape, b.shape)) {
    throw new Error(`${op}: shapes ${shapeStr(a.shape)} and ${shapeStr(b.shape)} do not match`);
  }
}
