/**
 * Differentiable operations.
 *
 * Binary elementwise ops broadcast by numpy's rules: shapes are aligned from
 * the right, and an axis of length 1, or a missing leading axis, is repeated
 * to match the other operand. A bias `[d]` added to `X·W` of `[n, d]`, a
 * per-row weight `[n, 1]` against `[n, d]`, a mask `[n, 1, T]` against
 * attention scores `[n, T, T]`: each is one call, and its adjoint sums the
 * gradient back over the axes that were repeated.
 *
 * One restriction numpy does not have: the result must have one operand's
 * shape. `[n]` against `[n, 1]` would broadcast BOTH to an `[n, n]` table,
 * which is almost always a column of residuals meeting a vector of
 * observations by mistake, and in a likelihood it is a wrong number rather
 * than an error. Write the outer product with `matmul`, or reshape one side,
 * when it is meant.
 */

import { sameShape, shapeStr, sizeOf, zeros } from './tensor.js';
import { node, toVar } from './tape.js';
import { addStrided, copyStrided, simplify, stridesOf } from './strided.js';

/**
 * Fold a gradient contribution back onto a scalar parent that was broadcast:
 * the SUM over the output it was spread across.
 * @private
 */
function unbroadcastScalar(contrib) {
  if (contrib.length === 1) return contrib;
  let s = 0;
  for (let i = 0; i < contrib.length; i++) s += contrib[i];
  return Float64Array.of(s);
}

/**
 * Shape of an elementwise result under numpy's rules, restricted to results
 * that are one operand's shape (see the module comment).
 * @private
 */
function broadcastShape(a, b, op) {
  const sa = a.shape;
  const sb = b.shape;
  if (sa.length === 0) return sb;
  if (sb.length === 0) return sa;
  if (sameShape(sa, sb)) return sa;
  const rank = Math.max(sa.length, sb.length);
  const out = new Array(rank);
  for (let k = 0; k < rank; k++) {
    const ka = k - (rank - sa.length);
    const kb = k - (rank - sb.length);
    const da = ka >= 0 ? sa[ka] : 1;
    const db = kb >= 0 ? sb[kb] : 1;
    if (da === db || db === 1) out[k] = da;
    else if (da === 1) out[k] = db;
    else {
      throw new Error(
        `${op}: shapes ${shapeStr(sa)} and ${shapeStr(sb)} do not match: ` +
          `aligned from the right, axis ${k} is ${da} against ${db}, and neither is 1`,
      );
    }
  }
  if (!sameShape(out, sa) && !sameShape(out, sb)) {
    throw new Error(
      `${op}: shapes ${shapeStr(sa)} and ${shapeStr(sb)} would both have to be broadcast, ` +
        `to ${shapeStr(out)}. That is usually a column meeting a vector by mistake; ` +
        'reshape one operand to the result shape if the outer combination is meant.',
    );
  }
  return out;
}

/**
 * Strides that read `v` as if it had the broadcast shape: its own strides,
 * with 0 on every axis it is repeated along. @private
 */
function broadcastStrides(vShape, shape) {
  const own = stridesOf(vShape);
  const lead = shape.length - vShape.length;
  return shape.map((d, k) => {
    const j = k - lead;
    if (j < 0) return 0;
    return vShape[j] === 1 && d !== 1 ? 0 : own[j];
  });
}

/**
 * Stride for reading an operand elementwise: 0 for a broadcast scalar, which
 * pins every read to element 0, and 1 otherwise. Resolving the broadcast to a
 * stride ONCE lets the kernels below be a single flat loop instead of
 * variants or a per-element branch.
 * @private
 */
const strideOf = (v) => (v.shape.length === 0 ? 0 : 1);

/**
 * Build a binary elementwise op from a forward and a backward KERNEL.
 *
 * A kernel is a whole loop, called once per pass — not a per-element callback.
 * That distinction is the difference between a usable tape and an unusable one.
 * With `f` passed per element, every op instance funnels a different function
 * through the same call site inside one shared loop body; the site goes
 * megamorphic, the arithmetic cannot inline, and a `+` costs upwards of 100 ns
 * instead of under one. Measured on the elementwise chain of a 340-observation
 * regression, hoisting the dispatch out of the loop is worth more than every
 * other optimization in this package put together.
 *
 * So the plumbing is shared and the arithmetic is not. Each kernel below is its
 * own function containing its own loop, and the one indirect call left happens
 * 38 times per gradient rather than 7496.
 *
 * @private
 * @param {string} op - name, for error messages
 * @param {(out:Float64Array, A:Float64Array, B:Float64Array, sa:number, sb:number, n:number) => void} fwd
 * @param {(g:Float64Array, ga:Float64Array, gb:Float64Array, A:Float64Array, B:Float64Array, out:Float64Array, sa:number, sb:number, n:number) => void} bwd
 */
function binary(op, fwd, bwd) {
  return (aIn, bIn, ...rest) => {
    // A binary op handed three operands used to compute the first two and drop
    // the rest, with no error and no clue: `sub(a, b, c)` quietly returned
    // `a - b`. That is a wrong number rather than a failure, which is the worst
    // shape a mistake can take here, so it is refused.
    if (rest.length > 0) {
      throw new Error(
        `${op}: takes exactly two operands, got ${2 + rest.length}. ` +
          'add and mul are the variadic ones.',
      );
    }
    const a = toVar(aIn, `${op} left operand`);
    const b = toVar(bIn, `${op} right operand`);
    const shape = broadcastShape(a, b, op);
    const n = sizeOf(shape);
    // An operand repeated along some axis is expanded into a buffer of the
    // result's size on every forward pass, so the kernels stay single flat
    // loops; a scalar reads at stride 0 and an operand of the result's shape
    // reads its own storage, neither paying for a copy.
    const expands = (v) => v.shape.length > 0 && !sameShape(v.shape, shape);
    const ea = expands(a);
    const eb = expands(b);
    const walkA = ea ? simplify(shape, broadcastStrides(a.shape, shape)) : null;
    const walkB = eb ? simplify(shape, broadcastStrides(b.shape, shape)) : null;
    const A = ea ? new Float64Array(n) : a.value.data;
    const B = eb ? new Float64Array(n) : b.value.data;
    const sa = strideOf(a);
    const sb = strideOf(b);
    const out = new Float64Array(n);
    const forward = () => {
      if (ea) copyStrided(A, a.value.data, walkA.shape, walkA.strides, 0);
      if (eb) copyStrided(B, b.value.data, walkB.shape, walkB.strides, 0);
      fwd(out, A, B, sa, sb, n);
    };

    // Allocated once, refilled on every reverse pass. `backward()` accumulates a
    // node's contributions into its parents before touching the next node, so
    // there is no window in which these could be read stale.
    const ga = new Float64Array(n);
    const gb = new Float64Array(n);
    const fa = ea ? new Float64Array(a.value.data.length) : null;
    const fb = eb ? new Float64Array(b.value.data.length) : null;
    const fold = (g, v, f, walk) => {
      if (f !== null) {
        f.fill(0);
        addStrided(f, g, walk.shape, walk.strides, 0);
        return f;
      }
      return v.shape.length === 0 ? unbroadcastScalar(g) : g;
    };
    return node({ data: out, shape: shape.slice() }, [a, b], (g) => {
      bwd(g, ga, gb, A, B, out, sa, sb, n);
      return [fold(ga, a, fa, walkA), fold(gb, b, fb, walkB)];
    }, forward, { op });
  };
}

/**
 * Build a unary elementwise op from a forward and a backward kernel.
 * Same reasoning as {@link binary}: the loop is the unit, not the element.
 *
 * @private
 * @param {string} op
 * @param {(out:Float64Array, A:Float64Array, n:number) => void} fwd
 * @param {(g:Float64Array, ga:Float64Array, A:Float64Array, out:Float64Array, n:number) => void} bwd
 */
function unary(op, fwd, bwd, args = undefined) {
  return (aIn, ...rest) => {
    if (rest.length > 0) {
      throw new Error(`${op}: takes exactly one operand, got ${1 + rest.length}`);
    }
    const a = toVar(aIn, `${op} operand`);
    const n = a.value.data.length;
    const A = a.value.data;
    const out = new Float64Array(n);
    const forward = () => fwd(out, A, n);
    const ga = new Float64Array(n);
    return node({ data: out, shape: a.shape.slice() }, [a], (g) => {
      bwd(g, ga, A, out, n);
      return [ga];
    }, forward, args === undefined ? { op } : { op, args });
  };
}

/**
 * The escape hatch: a unary op from a per-element function, for the cases where
 * the derivative depends on something captured at construction and there is no
 * fixed kernel to write. `pow` is the only one. Pays the megamorphic call
 * described on {@link binary}, so do not reach for it to add a common op.
 * @private
 */
function unaryFn(op, f, df, args) {
  return unary(
    op,
    (out, A, n) => { for (let i = 0; i < n; i++) out[i] = f(A[i]); },
    (g, ga, A, out, n) => { for (let i = 0; i < n; i++) ga[i] = g[i] * df(A[i], out[i]); },
    args,
  );
}

/**
 * Let an associative op take any number of operands, by folding left.
 *
 * JavaScript has no operator overloading, so a model's mean cannot be written
 * `mu0 + tau * z + gamma` the way PyMC writes it: `+` on a `Var` is not
 * something this package can define. What it CAN remove is the nesting that
 * comes from a strictly binary op, which is what actually makes a five-term
 * sum unreadable:
 *
 *     add(add(add(add(a, b), c), d), e)      // before
 *     add(a, b, c, d, e)                     // after
 *
 * The fold builds exactly the graph the nested form did, so nothing about
 * differentiation, broadcasting or cost changes. Only `add` and `mul` get
 * this. `sub` and `div` stay binary on purpose: `sub(a, b, c)` would have to
 * mean `a - b - c`, which reads like it might mean something else, and a
 * subtraction chain is rare enough not to be worth the ambiguity.
 *
 * @private
 */
function variadic(op, binaryOp) {
  return (...args) => {
    if (args.length < 2) {
      throw new Error(`${op}: needs at least two operands, got ${args.length}`);
    }
    let acc = binaryOp(args[0], args[1]);
    for (let i = 2; i < args.length; i++) acc = binaryOp(acc, args[i]);
    return acc;
  };
}

/**
 * Sum of two or more operands, elementwise, broadcasting a scalar against
 * anything and a vector against the rows of a matrix.
 *
 * @param {...(Var|number|number[]|number[][])} operands - at least two
 * @returns {Var}
 *
 * @example
 * const mu = add(intercept, mul(slope, X), seasonOffset);
 */
export const add = variadic('add', binary(
  'add',
  (out, A, B, sa, sb, n) => { for (let i = 0; i < n; i++) out[i] = A[i * sa] + B[i * sb]; },
  (g, ga, gb, _A, _B, _out, _sa, _sb, n) => {
    for (let i = 0; i < n; i++) { ga[i] = g[i]; gb[i] = g[i]; }
  },
));

export const sub = binary(
  'sub',
  (out, A, B, sa, sb, n) => { for (let i = 0; i < n; i++) out[i] = A[i * sa] - B[i * sb]; },
  (g, ga, gb, _A, _B, _out, _sa, _sb, n) => {
    for (let i = 0; i < n; i++) { ga[i] = g[i]; gb[i] = -g[i]; }
  },
);

/**
 * Product of two or more operands, elementwise, broadcasting a scalar against
 * anything and a vector against the rows of a matrix.
 *
 * @param {...(Var|number|number[]|number[][])} operands - at least two
 * @returns {Var}
 */
export const mul = variadic('mul', binary(
  'mul',
  (out, A, B, sa, sb, n) => { for (let i = 0; i < n; i++) out[i] = A[i * sa] * B[i * sb]; },
  (g, ga, gb, A, B, _out, sa, sb, n) => {
    for (let i = 0; i < n; i++) {
      ga[i] = g[i] * B[i * sb];
      gb[i] = g[i] * A[i * sa];
    }
  },
));

export const div = binary(
  'div',
  (out, A, B, sa, sb, n) => { for (let i = 0; i < n; i++) out[i] = A[i * sa] / B[i * sb]; },
  (g, ga, gb, A, B, _out, sa, sb, n) => {
    for (let i = 0; i < n; i++) {
      const y = B[i * sb];
      ga[i] = g[i] / y;
      gb[i] = (-g[i] * A[i * sa]) / (y * y);
    }
  },
);

export const neg = unary(
  'neg',
  (out, A, n) => { for (let i = 0; i < n; i++) out[i] = -A[i]; },
  (g, ga, _A, _out, n) => { for (let i = 0; i < n; i++) ga[i] = -g[i]; },
);

export const exp = unary(
  'exp',
  (out, A, n) => { for (let i = 0; i < n; i++) out[i] = Math.exp(A[i]); },
  (g, ga, _A, out, n) => { for (let i = 0; i < n; i++) ga[i] = g[i] * out[i]; },
);

export const log = unary(
  'log',
  (out, A, n) => { for (let i = 0; i < n; i++) out[i] = Math.log(A[i]); },
  (g, ga, A, _out, n) => { for (let i = 0; i < n; i++) ga[i] = g[i] / A[i]; },
);

export const sqrt = unary(
  'sqrt',
  (out, A, n) => { for (let i = 0; i < n; i++) out[i] = Math.sqrt(A[i]); },
  (g, ga, _A, out, n) => { for (let i = 0; i < n; i++) ga[i] = g[i] / (2 * out[i]); },
);

export const square = unary(
  'square',
  (out, A, n) => { for (let i = 0; i < n; i++) out[i] = A[i] * A[i]; },
  (g, ga, A, _out, n) => { for (let i = 0; i < n; i++) ga[i] = g[i] * 2 * A[i]; },
);

export const tanh = unary(
  'tanh',
  (out, A, n) => { for (let i = 0; i < n; i++) out[i] = Math.tanh(A[i]); },
  (g, ga, _A, out, n) => { for (let i = 0; i < n; i++) ga[i] = g[i] * (1 - out[i] * out[i]); },
);

export const sigmoid = unary(
  'sigmoid',
  (out, A, n) => { for (let i = 0; i < n; i++) out[i] = 1 / (1 + Math.exp(-A[i])); },
  (g, ga, _A, out, n) => { for (let i = 0; i < n; i++) ga[i] = g[i] * out[i] * (1 - out[i]); },
);

/**
 * Elementwise maximum, broadcasting a scalar against anything.
 *
 * The subgradient at a tie is a convention, not a derivation: when `a === b`
 * the whole adjoint goes to the LEFT operand, so `maximum(x, 0)` at `x = 0`
 * reports `dx = 1`. Deterministic and cheap, and the caller landing exactly on
 * the tie can predict which way it falls. Splitting the adjoint evenly would
 * be defensible too, but it is not what this does.
 *
 * `Math.max` is the forward, so a NaN operand propagates rather than being
 * quietly outranked. A sampler stepping outside a support needs the non-finite
 * value to reach it.
 *
 * @param {Var|number|number[]|number[][]} a
 * @param {Var|number|number[]|number[][]} b
 * @returns {Var}
 */
export const maximum = binary(
  'maximum',
  (out, A, B, sa, sb, n) => { for (let i = 0; i < n; i++) out[i] = Math.max(A[i * sa], B[i * sb]); },
  (g, ga, gb, A, B, _out, sa, sb, n) => {
    for (let i = 0; i < n; i++) {
      // `>=` is false when either operand is NaN, so the adjoint follows the
      // right operand there, matching Math.max having propagated the NaN.
      const left = A[i * sa] >= B[i * sb] ? 1 : 0;
      ga[i] = g[i] * left;
      gb[i] = g[i] * (1 - left);
    }
  },
);

/**
 * Elementwise minimum, broadcasting a scalar against anything.
 *
 * Mirrors {@link maximum}: at a tie the whole adjoint goes to the LEFT operand,
 * so `minimum(x, 0)` at `x = 0` reports `dx = 1`. A NaN operand propagates.
 *
 * @param {Var|number|number[]|number[][]} a
 * @param {Var|number|number[]|number[][]} b
 * @returns {Var}
 */
export const minimum = binary(
  'minimum',
  (out, A, B, sa, sb, n) => { for (let i = 0; i < n; i++) out[i] = Math.min(A[i * sa], B[i * sb]); },
  (g, ga, gb, A, B, _out, sa, sb, n) => {
    for (let i = 0; i < n; i++) {
      const left = A[i * sa] <= B[i * sb] ? 1 : 0;
      ga[i] = g[i] * left;
      gb[i] = g[i] * (1 - left);
    }
  },
);

/**
 * Rectified linear unit, `max(x, 0)`.
 *
 * The same thing as `maximum(x, 0)`, as a unary op: no second operand and no
 * broadcast machinery for the case that wants neither. It is also the
 * primitive that makes a piecewise-linear model expressible at all, a
 * quadratic-plateau dose response being the case in hand: the join sits at a
 * PARAMETER, so which observations fall below it changes as the optimizer or
 * sampler moves, and no precomputed mask can stand in for the clamp.
 *
 * `relu'(0) = 0`, matching JAX and PyTorch. A NaN propagates.
 *
 * @param {Var|number|number[]|number[][]} a
 * @returns {Var}
 */
/**
 * ln|Γ(x)| and its derivative ψ(x), the digamma function. The same Lanczos
 * (g = 7) and asymptotic-series algorithms `@tangent.to/proba` uses, so the
 * two agree to rounding; grad carries its own copy rather than a dependency,
 * because it is the low-level package.
 * @private
 */
const LANCZOS_G = 7;
const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  1.5056327351493116e-7,
];
const LN_SQRT_2PI = 0.9189385332046727;
function lgammaScalar(x) {
  if (Number.isNaN(x)) return NaN;
  if (x <= 0 && Number.isInteger(x)) return Infinity;
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgammaScalar(1 - x);
  const z = x - 1;
  let sum = LANCZOS[0];
  for (let i = 1; i < LANCZOS.length; i++) sum += LANCZOS[i] / (z + i);
  const t = z + LANCZOS_G + 0.5;
  return LN_SQRT_2PI + (z + 0.5) * Math.log(t) - t + Math.log(sum);
}
function digammaScalar(x) {
  if (Number.isNaN(x)) return NaN;
  if (x <= 0 && Number.isInteger(x)) return NaN;
  if (x < 0) return digammaScalar(1 - x) - Math.PI / Math.tan(Math.PI * x);
  let result = 0;
  while (x < 10) { result -= 1 / x; x += 1; }
  const inv = 1 / x;
  const inv2 = inv * inv;
  result += Math.log(x) - 0.5 * inv -
    inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 * (1 / 252 - inv2 * (1 / 240 - inv2 / 132))));
  return result;
}

/**
 * Elementwise log-gamma, `ln|Γ(x)|`, with the digamma function as its
 * derivative. What a Gamma or Beta log-density needs when its shape
 * parameter is itself being differentiated, as in a hierarchical prior.
 *
 * @param {Var|number|number[]|number[][]} aIn
 * @returns {Var}
 */
export const lgamma = unary(
  'lgamma',
  (out, A, n) => { for (let i = 0; i < n; i++) out[i] = lgammaScalar(A[i]); },
  (g, ga, A, _out, n) => { for (let i = 0; i < n; i++) ga[i] = g[i] * digammaScalar(A[i]); },
);

export const relu = unary(
  'relu',
  // Math.max, not `A[i] > 0 ? A[i] : 0`: the ternary answers 0 for NaN and
  // swallows it, which is exactly the signal a sampler needs to see.
  (out, A, n) => { for (let i = 0; i < n; i++) out[i] = Math.max(A[i], 0); },
  (g, ga, A, _out, n) => { for (let i = 0; i < n; i++) ga[i] = A[i] > 0 ? g[i] : 0; },
);

/**
 * Raise elementwise to a CONSTANT power. The exponent is not differentiated —
 * for a variable exponent write `exp(mul(k, log(x)))`, which carries both
 * partials and makes the domain restriction on `x` explicit.
 *
 * @param {Var|number|number[]|number[][]} aIn
 * @param {number} k
 * @returns {Var}
 */
export function pow(aIn, k) {
  if (typeof k !== 'number' || !Number.isFinite(k)) {
    throw new Error(`pow: exponent must be a finite number; got ${k}`);
  }
  return unaryFn('pow', (x) => Math.pow(x, k), (x) => k * Math.pow(x, k - 1), [k])(aIn);
}

/**
 * Where a reduction runs: the whole tensor, or one axis of it. Seen as
 * `[outer, len, inner]` around the reduced axis, every reduction below is one
 * loop shape, whatever the rank. @private
 */
function reduction(op, a, options) {
  const n = a.value.data.length;
  const whole = { outer: 1, len: n, inner: 1, shape: [], args: undefined };
  if (options === undefined) return whole;
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new Error(`${op}: options must be an object like { axis: 1 }`);
  }
  const { axis, keepDims = false } = options;
  if (axis === undefined) {
    if (keepDims) throw new Error(`${op}: keepDims needs an axis`);
    return whole;
  }
  const rank = a.shape.length;
  if (!Number.isInteger(axis) || axis < 0 || axis >= rank) {
    throw new Error(`${op}: axis ${axis} does not exist on a ${shapeStr(a.shape)} operand`);
  }
  const outer = sizeOf(a.shape.slice(0, axis));
  const inner = sizeOf(a.shape.slice(axis + 1));
  const shape = keepDims
    ? a.shape.map((d, k) => (k === axis ? 1 : d))
    : a.shape.filter((_, k) => k !== axis);
  return { outer, len: a.shape[axis], inner, shape, args: [{ axis, keepDims: Boolean(keepDims) }] };
}

/** Sum or mean along a reduction; `scale` is 1 or 1/len. @private */
function additiveReduce(op, aIn, options, mean) {
  const a = toVar(aIn, `${op} operand`);
  const { outer, len, inner, shape, args } = reduction(op, a, options);
  const A = a.value.data;
  const out = new Float64Array(outer * inner);
  const ga = new Float64Array(A.length);
  // A mean divides rather than multiplying by 1/len, as 0.3 did, so a
  // whole-tensor mean is bit-for-bit what it was.
  if (inner === 1) {
    // The whole tensor, or the last axis: each output sums one contiguous run.
    const forward = () => {
      for (let o = 0, base = 0; o < outer; o++, base += len) {
        let s = 0;
        for (let k = 0; k < len; k++) s += A[base + k];
        out[o] = mean ? s / len : s;
      }
    };
    return node({ data: out, shape }, [a], (g) => {
      for (let o = 0, base = 0; o < outer; o++, base += len) ga.fill(mean ? g[o] / len : g[o], base, base + len);
      return [ga];
    }, forward, args === undefined ? { op } : { op, args });
  }
  const forward = () => {
    out.fill(0);
    for (let o = 0; o < outer; o++) {
      const base = o * len * inner;
      const ob = o * inner;
      for (let k = 0; k < len; k++) {
        const row = base + k * inner;
        for (let i = 0; i < inner; i++) out[ob + i] += A[row + i];
      }
    }
    if (mean) for (let i = 0; i < out.length; i++) out[i] /= len;
  };
  return node({ data: out, shape }, [a], (g) => {
    for (let o = 0; o < outer; o++) {
      const base = o * len * inner;
      const ob = o * inner;
      for (let k = 0; k < len; k++) {
        const row = base + k * inner;
        for (let i = 0; i < inner; i++) ga[row + i] = mean ? g[ob + i] / len : g[ob + i];
      }
    }
    return [ga];
  }, forward, args === undefined ? { op } : { op, args });
}

/**
 * Sum every element to a scalar, or sum along one axis.
 *
 * With `{ axis }` the axis is removed: a `[n, d]` matrix summed over axis 1
 * gives one total per row, `[n]`. `keepDims: true` leaves it in place at
 * length 1 instead, `[n, 1]`, which broadcasts straight back against the
 * matrix it came from — what a softmax or a per-row normalization needs.
 *
 * @param {Var|number[]|number[][]} aIn
 * @param {{ axis?: number, keepDims?: boolean }} [options]
 * @returns {Var}
 *
 * @example
 * const rowTotals = sum(X, { axis: 1, keepDims: true }); // [n, 1]
 */
export function sum(aIn, options) {
  return additiveReduce('sum', aIn, options, false);
}

/**
 * Mean of every element as a scalar, or the mean along one axis. Options as
 * on {@link sum}.
 *
 * @param {Var|number[]|number[][]} aIn
 * @param {{ axis?: number, keepDims?: boolean }} [options]
 * @returns {Var}
 */
export function mean(aIn, options) {
  return additiveReduce('mean', aIn, options, true);
}

/**
 * Largest element as a scalar, or the largest along one axis. Options as on
 * {@link sum}.
 *
 * The adjoint goes to the position that won. At a tie that is the FIRST
 * position along the axis, the same left-wins convention as
 * {@link maximum}. A NaN wins outright and propagates, so a sampler outside
 * a support still sees it.
 *
 * @param {Var|number[]|number[][]} aIn
 * @param {{ axis?: number, keepDims?: boolean }} [options]
 * @returns {Var}
 *
 * @example
 * // A stable log-sum-exp per row of logits Z: [n, d] → [n, 1]
 * const m = max(Z, { axis: 1, keepDims: true });
 * const lse = add(m, log(sum(exp(sub(Z, m)), { axis: 1, keepDims: true })));
 */
export function max(aIn, options) {
  const a = toVar(aIn, 'max operand');
  const { outer, len, inner, shape, args } = reduction('max', a, options);
  const A = a.value.data;
  const out = new Float64Array(outer * inner);
  const winner = new Int32Array(outer * inner);
  const forward = () => {
    for (let o = 0; o < outer; o++) {
      const base = o * len * inner;
      for (let i = 0; i < inner; i++) {
        let at = base + i;
        let best = A[at];
        for (let k = 1; k < len && !Number.isNaN(best); k++) {
          const p = base + k * inner + i;
          const v = A[p];
          if (v > best || Number.isNaN(v)) {
            best = v;
            at = p;
          }
        }
        out[o * inner + i] = best;
        winner[o * inner + i] = at;
      }
    }
  };
  const ga = new Float64Array(A.length);
  return node({ data: out, shape }, [a], (g) => {
    ga.fill(0);
    for (let j = 0; j < winner.length; j++) ga[winner[j]] = g[j];
    return [ga];
  }, forward, args === undefined ? { op: 'max' } : { op: 'max', args });
}

/**
 * Matrix product over the last two axes, as in numpy.
 *
 * - `[m, k] · [k, n]` is the matrix product; a vector `[k]` on the right is
 *   a column and gives `[m]`.
 * - `[..., m, k] · [k, n]` applies one matrix to every leading index: a
 *   weight matrix against a batch of sequences `[n, T, F] · [F, H]`. Storage
 *   is row-major, so this is one `(… · m) × k` product, not a loop.
 * - `[..., m, k] · [..., k, n]` with the same leading axes multiplies block
 *   by block: attention's `Q · Kᵀ` per sequence.
 *
 * @param {Var|number[][]} aIn - (…, m, k)
 * @param {Var|number[][]|number[]} bIn - (k, n), (k) or (…, k, n)
 * @returns {Var} (…, m, n), or (…, m) for a vector on the right
 */
export function matmul(aIn, bIn) {
  const a = toVar(aIn, 'matmul left operand');
  const b = toVar(bIn, 'matmul right operand');
  const ra = a.shape.length;
  const rb = b.shape.length;
  if (ra < 2) {
    throw new Error(`matmul: left operand must have at least two axes, got ${shapeStr(a.shape)}`);
  }
  if (rb === 0) {
    throw new Error('matmul: right operand cannot be a scalar; use mul');
  }
  const lead = a.shape.slice(0, -2);
  const [m, k] = a.shape.slice(-2);
  const batched = rb >= 3;
  const vectorRhs = rb === 1;
  const k2 = vectorRhs ? b.shape[0] : b.shape[rb - 2];
  const n = vectorRhs ? 1 : b.shape[rb - 1];
  if (k !== k2) {
    throw new Error(
      `matmul: inner dimensions disagree — ${shapeStr(a.shape)} against ${shapeStr(b.shape)}`,
    );
  }
  if (batched && (rb !== ra || !sameShape(b.shape.slice(0, -2), lead))) {
    throw new Error(
      `matmul: batch axes disagree — ${shapeStr(a.shape)} against ${shapeStr(b.shape)}; ` +
        'a batched right operand needs the same leading axes as the left',
    );
  }

  // Unbatched, the leading axes fold into the rows: one product of M rows.
  const blocks = batched ? sizeOf(lead) : 1;
  const M = batched ? m : sizeOf(lead) * m;
  const A = a.value.data;
  const B = b.value.data;
  const out = new Float64Array(blocks * M * n);
  const ga = new Float64Array(A.length);
  const gb = new Float64Array(B.length);
  const Bt = new Float64Array(k * n);
  const gbt = new Float64Array(k * n);

  // The kernels are closures over this node's sizes, not shared functions
  // taking them as arguments: V8 optimizes the inner loops markedly better
  // when M, k and n are captured constants (a shared kernel cost a third of
  // the forward pass on a 300 × 300 product). Offsets select a block.
  const forwardBlock = (oo, ao, bo) => {
    out.fill(0, oo, oo + M * n); // the kernel accumulates
    for (let i = 0; i < M; i++) {
      const ai = ao + i * k;
      const oi = oo + i * n;
      for (let p = 0; p < k; p++) {
        const aip = A[ai + p];
        if (aip === 0) continue;
        const bp = bo + p * n;
        for (let j = 0; j < n; j++) out[oi + j] += aip * B[bp + j];
      }
    }
  };
  // Ā = Ḡ Bᵀ, B̄ = Aᵀ Ḡ, walked as (i, j, p) so a zero entry of Ḡ is skipped
  // whole: it contributes nothing, and a non-finite A or B meeting it does
  // not become NaN. The p loop runs along rows of A and of Ā; B is read
  // through its transpose and B̄ is accumulated transposed, so that loop is
  // contiguous in all four arrays instead of striding by n.
  const backwardBlock = (g, go, ao, bo) => {
    for (let p = 0; p < k; p++) {
      const bp = bo + p * n;
      for (let j = 0; j < n; j++) Bt[j * k + p] = B[bp + j];
    }
    ga.fill(0, ao, ao + M * k);
    gbt.fill(0);
    for (let i = 0; i < M; i++) {
      const ai = ao + i * k;
      const gi = go + i * n;
      for (let j = 0; j < n; j++) {
        const gij = g[gi + j];
        if (gij === 0) continue;
        const bj = j * k;
        for (let p = 0; p < k; p++) {
          ga[ai + p] += gij * Bt[bj + p];
          gbt[bj + p] += A[ai + p] * gij;
        }
      }
    }
    for (let p = 0; p < k; p++) {
      const bp = bo + p * n;
      for (let j = 0; j < n; j++) gb[bp + j] = gbt[j * k + p];
    }
  };
  // One block — every unbatched product, the GLM's X·β included — gets
  // kernels without offsets: on a skinny X the per-row offset arithmetic
  // showed up as 5–10%.
  const forwardOne = () => {
    out.fill(0);
    for (let i = 0; i < M; i++) {
      for (let p = 0; p < k; p++) {
        const aip = A[i * k + p];
        if (aip === 0) continue;
        for (let j = 0; j < n; j++) out[i * n + j] += aip * B[p * n + j];
      }
    }
  };
  const backwardOne = (g) => {
    for (let p = 0; p < k; p++) {
      for (let j = 0; j < n; j++) Bt[j * k + p] = B[p * n + j];
    }
    ga.fill(0);
    gbt.fill(0);
    for (let i = 0; i < M; i++) {
      const ai = i * k;
      for (let j = 0; j < n; j++) {
        const gij = g[i * n + j];
        if (gij === 0) continue;
        const bj = j * k;
        for (let p = 0; p < k; p++) {
          ga[ai + p] += gij * Bt[bj + p];
          gbt[bj + p] += A[ai + p] * gij;
        }
      }
    }
    for (let p = 0; p < k; p++) {
      for (let j = 0; j < n; j++) gb[p * n + j] = gbt[j * k + p];
    }
  };
  const forward = blocks === 1 ? forwardOne : () => {
    for (let t = 0; t < blocks; t++) forwardBlock(t * M * n, t * M * k, batched ? t * k * n : 0);
  };

  const shape = vectorRhs ? [...lead, m] : [...lead, m, n];
  return node({ data: out, shape }, [a, b], (g) => {
    if (blocks === 1) backwardOne(g);
    else for (let t = 0; t < blocks; t++) backwardBlock(g, t * M * n, t * M * k, batched ? t * k * n : 0);
    return [ga, gb];
  }, forward, { op: 'matmul' });
}

/**
 * Inner product of two vectors, as a scalar.
 * @param {Var|number[]} uIn
 * @param {Var|number[]} vIn
 * @returns {Var}
 */
export function dot(uIn, vIn) {
  const u = toVar(uIn, 'dot left operand');
  const v = toVar(vIn, 'dot right operand');
  if (u.shape.length !== 1 || v.shape.length !== 1 || u.shape[0] !== v.shape[0]) {
    throw new Error(`dot: needs two vectors of equal length, got ${shapeStr(u.shape)} and ${shapeStr(v.shape)}`);
  }
  const n = u.shape[0];
  const out = new Float64Array(1);
  const forward = () => {
    let s = 0;
    for (let i = 0; i < n; i++) s += u.value.data[i] * v.value.data[i];
    out[0] = s;
  };
  const gu = new Float64Array(n);
  const gv = new Float64Array(n);
  return node({ data: out, shape: [] }, [u, v], (g) => {
    for (let i = 0; i < n; i++) {
      gu[i] = g[0] * v.value.data[i];
      gv[i] = g[0] * u.value.data[i];
    }
    return [gu, gv];
  }, forward, { op: 'dot' });
}

/**
 * Permute axes. With no permutation the axes reverse, which for a matrix is
 * the transpose; `transpose(x, [0, 2, 1])` swaps the last two axes of a
 * batch, the `Kᵀ` in attention's `Q · Kᵀ`. The adjoint applies the inverse
 * permutation, adding into the parent in place.
 *
 * @param {Var|number[][]} aIn
 * @param {number[]} [perm] - a permutation of `0 … rank − 1`; reversed by default
 * @returns {Var}
 */
export function transpose(aIn, perm) {
  const a = toVar(aIn, 'transpose operand');
  const rank = a.shape.length;
  if (rank < 2) {
    throw new Error(`transpose: needs at least two axes, got ${shapeStr(a.shape)}`);
  }
  const p = perm === undefined ? a.shape.map((_, k) => rank - 1 - k) : perm;
  if (!Array.isArray(p) || p.length !== rank || !p.every((k) => Number.isInteger(k) && k >= 0 && k < rank) ||
    new Set(p).size !== rank) {
    throw new Error(`transpose: perm must be a permutation of 0…${rank - 1}, got ${JSON.stringify(perm)}`);
  }
  const own = stridesOf(a.shape);
  const shape = p.map((k) => a.shape[k]);
  const walk = simplify(shape, p.map((k) => own[k]));
  const src = a.value.data;
  const out = new Float64Array(src.length);
  const forward = () => copyStrided(out, src, walk.shape, walk.strides, 0);
  return node({ data: out, shape }, [a], (g) => [(into) => addStrided(into, g, walk.shape, walk.strides, 0)],
    forward, perm === undefined ? { op: 'transpose' } : { op: 'transpose', args: [p.slice()] });
}

/**
 * Diagonal of a square matrix, as a vector.
 * @param {Var|number[][]} aIn
 * @returns {Var}
 */
export function diagPart(aIn) {
  const a = toVar(aIn, 'diagPart operand');
  if (a.shape.length !== 2 || a.shape[0] !== a.shape[1]) {
    throw new Error(`diagPart: needs a square matrix, got ${shapeStr(a.shape)}`);
  }
  const n = a.shape[0];
  const out = new Float64Array(n);
  const forward = () => {
    for (let i = 0; i < n; i++) out[i] = a.value.data[i * n + i];
  };
  const ga = new Float64Array(n * n);
  return node({ data: out, shape: [n] }, [a], (g) => {
    for (let i = 0; i < n; i++) ga[i * n + i] = g[i];
    return [ga];
  }, forward, { op: 'diagPart' });
}

/**
 * Trace of a square matrix, as a scalar.
 * @param {Var|number[][]} aIn
 * @returns {Var}
 */
export function trace(aIn) {
  return sum(diagPart(aIn));
}

/**
 * Add a constant to the diagonal of a square matrix — the jitter/noise idiom
 * (`K + αI`) that every Gaussian-process likelihood opens with. Differentiable
 * in both the matrix and the scalar.
 *
 * @param {Var|number[][]} aIn - square matrix
 * @param {Var|number|number[]} alphaIn - scalar, or one variance per row
 * @returns {Var}
 */
export function addDiag(aIn, alphaIn) {
  const a = toVar(aIn, 'addDiag matrix');
  const alpha = toVar(alphaIn, 'addDiag diagonal');
  if (a.shape.length !== 2 || a.shape[0] !== a.shape[1]) {
    throw new Error(`addDiag: needs a square matrix, got ${shapeStr(a.shape)}`);
  }
  const n = a.shape[0];
  const perRow = alpha.shape.length === 1;
  if (perRow && alpha.shape[0] !== n) {
    throw new Error(`addDiag: diagonal has ${alpha.shape[0]} entries for a ${n}×${n} matrix`);
  }
  if (!perRow && alpha.shape.length !== 0) {
    throw new Error(`addDiag: diagonal must be a scalar or a vector, got ${shapeStr(alpha.shape)}`);
  }
  const out = new Float64Array(n * n);
  const forward = () => {
    out.set(a.value.data);
    for (let i = 0; i < n; i++) out[i * n + i] += perRow ? alpha.value.data[i] : alpha.value.data[0];
  };

  const gAlpha = new Float64Array(perRow ? n : 1);
  return node({ data: out, shape: [n, n] }, [a, alpha], (g) => {
    gAlpha.fill(0);
    for (let i = 0; i < n; i++) {
      if (perRow) gAlpha[i] = g[i * n + i];
      else gAlpha[0] += g[i * n + i];
    }
    // g is the node's own grad buffer, which the caller keeps accumulating
    // into; the matrix parent needs a copy, not a view of it.
    return [g.slice(), gAlpha];
  }, forward, { op: 'addDiag' });
}

/**
 * Reinterpret a tensor's shape without moving data. Row-major order is
 * preserved, so a length-6 vector becomes a 2×3 matrix reading left to right,
 * top to bottom. The adjoint is the identity — only the shape label changes.
 *
 * The everyday use is packing a flat parameter vector, which is what an
 * optimizer or a sampler hands you, into the matrix a model is written in.
 *
 * A batch of sequences moves between `[n, T, F]`, `[n, T·F]` and `[n·T, F]`
 * the same way: the second is what a dense layer reads, the third what lets
 * one matmul apply a weight matrix to every time step of every sequence.
 *
 * @param {Var|number[]|number[][]} aIn
 * @param {number[]} shape - positive integers holding the same element count
 * @returns {Var}
 */
export function reshape(aIn, shape) {
  const a = toVar(aIn, 'reshape operand');
  if (!Array.isArray(shape) || shape.some((d) => !Number.isInteger(d) || d < 1)) {
    throw new Error('reshape: shape must be an array of positive integers');
  }
  if (sizeOf(shape) !== a.value.data.length) {
    throw new Error(
      `reshape: cannot view ${a.value.data.length} elements as ${shapeStr(shape)}`,
    );
  }
  const out = new Float64Array(a.value.data.length);
  const forward = () => out.set(a.value.data);
  return node({ data: out, shape: shape.slice() }, [a], (g) => [g.slice()], forward, { op: 'reshape', args: [shape.slice()] });
}

/**
 * Extract a contiguous block along every axis — the differentiable form of
 * `rows.slice(...).map((r) => r.slice(...))`, at any rank.
 *
 * A structural equation model needs it: Σ is built over latent AND observed
 * variables, then only the observed block is compared with the data. A
 * recurrent network needs it too: one time step of `[n, T, F]` is
 * `slice(x, [0, t, 0], [n, 1, F])`.
 *
 * The adjoint adds into the parent's gradient in place rather than returning
 * a dense buffer the size of the parent: T time-step slices of one batch
 * would otherwise cost O(T²) in the reverse pass.
 *
 * @param {Var|number[]|number[][]} aIn
 * @param {number[]} start - starting index per axis
 * @param {number[]} size - extent per axis
 * @returns {Var}
 */
export function slice(aIn, start, size) {
  const a = toVar(aIn, 'slice operand');
  const rank = a.shape.length;
  if (rank === 0) throw new Error('slice: cannot slice a scalar');
  if (!Array.isArray(start) || !Array.isArray(size) || start.length !== rank || size.length !== rank) {
    throw new Error(`slice: start and size need ${rank} entr${rank === 1 ? 'y' : 'ies'} for a ${shapeStr(a.shape)} operand`);
  }
  for (let d = 0; d < rank; d++) {
    if (!Number.isInteger(start[d]) || !Number.isInteger(size[d]) || start[d] < 0 || size[d] < 1) {
      throw new Error('slice: start must be a non-negative integer and size a positive integer');
    }
    if (start[d] + size[d] > a.shape[d]) {
      throw new Error(
        `slice: [${start[d]}, ${start[d] + size[d]}) exceeds axis ${d} of ${shapeStr(a.shape)}`,
      );
    }
  }
  const strides = stridesOf(a.shape);
  let base = 0;
  for (let d = 0; d < rank; d++) base += start[d] * strides[d];
  const shape = size.slice();
  const walk = simplify(shape, strides);
  const src = a.value.data;
  const out = new Float64Array(sizeOf(shape));
  const forward = () => copyStrided(out, src, walk.shape, walk.strides, base);
  return node({ data: out, shape }, [a], (g) => [(into) => addStrided(into, g, walk.shape, walk.strides, base)],
    forward, { op: 'slice', args: [start.slice(), size.slice()] });
}

/**
 * Join parts end to end. Scalars and vectors become one vector. Tensors of
 * rank 2 or more join along `axis`, agreeing on every other axis: for
 * matrices 0 stacks rows and 1 puts them side by side, which is how a
 * network joins the outputs of two branches for every row of a batch; axis
 * 1 of `[n, 1, H]` states is how a recurrent network stacks its steps back
 * into `[n, T, H]`. The adjoint hands each part its slice.
 *
 * @param {Array<Var|number|number[]|number[][]>} parts
 * @param {{ axis?: number }} [options] - for rank 2 and above; 0 by default
 * @returns {Var}
 */
export function concat(parts, options = {}) {
  if (!Array.isArray(parts) || parts.length === 0) {
    throw new Error('concat: needs a non-empty array of parts');
  }
  const vars = parts.map((p, i) => toVar(p, `concat part ${i}`));
  if (vars[0].shape.length >= 2) return concatAlong(vars, options.axis ?? 0);
  for (const v of vars) {
    if (v.shape.length > 1) {
      throw new Error(
        `concat: parts must all be scalars and vectors, or all of one rank ≥ 2; got ${shapeStr(v.shape)}`,
      );
    }
  }
  const lengths = vars.map((v) => v.value.data.length);
  const total = lengths.reduce((a, b) => a + b, 0);
  const out = new Float64Array(total);
  const forward = () => {
    let off = 0;
    for (const v of vars) {
      out.set(v.value.data, off);
      off += v.value.data.length;
    }
  };
  return node({ data: out, shape: [total] }, vars, (g) => {
    const contribs = [];
    let o = 0;
    for (const len of lengths) {
      contribs.push(g.slice(o, o + len));
      o += len;
    }
    return contribs;
  }, forward, { op: 'concat', list: true });
}

/** How a rank is named in an error message. @private */
const rankName = (rank) => (rank === 2 ? 'matrices' : `rank-${rank} tensors`);

/**
 * Parts of one rank ≥ 2 along `axis`. Seen as `[outer, block]`, where `outer`
 * is the product of the axes before `axis` and each part's block is
 * everything from `axis` on, every part is a run of `outer` contiguous
 * blocks. @private
 */
function concatAlong(vars, axis) {
  const ref = vars[0].shape;
  const rank = ref.length;
  if (!Number.isInteger(axis) || axis < 0 || axis >= rank) {
    throw new Error(`concat: axis must be an integer from 0 to ${rank - 1} for ${rankName(rank)}, got ${axis}`);
  }
  for (const v of vars) {
    if (v.shape.length !== rank) {
      throw new Error(`concat: parts must all be ${rankName(rank)}, got ${shapeStr(v.shape)}`);
    }
    for (let k = 0; k < rank; k++) {
      if (k !== axis && v.shape[k] !== ref[k]) {
        throw new Error(`concat: parts must agree on axis ${k}, got ${shapeStr(ref)} and ${shapeStr(v.shape)}`);
      }
    }
  }
  const outer = sizeOf(ref.slice(0, axis));
  const blocks = vars.map((v) => sizeOf(v.shape.slice(axis)));
  const width = blocks.reduce((a, b) => a + b, 0);
  const shape = ref.slice();
  shape[axis] = vars.reduce((a, v) => a + v.shape[axis], 0);
  const out = new Float64Array(outer * width);
  const forward = () => {
    let off = 0;
    for (let k = 0; k < vars.length; k++) {
      const src = vars[k].value.data;
      const b = blocks[k];
      if (outer === 1) out.set(src, off);
      else for (let o = 0; o < outer; o++) out.set(src.subarray(o * b, o * b + b), o * width + off);
      off += b;
    }
  };
  const bufs = blocks.map((b) => new Float64Array(outer * b));
  return node({ data: out, shape }, vars, (g) => {
    let off = 0;
    for (let k = 0; k < vars.length; k++) {
      const b = blocks[k];
      const buf = bufs[k];
      for (let o = 0; o < outer; o++) buf.set(g.subarray(o * width + off, o * width + off + b), o * b);
      off += b;
    }
    return bufs;
  }, forward, { op: 'concat', list: true, args: [{ axis }] });
}

/**
 * Pick entries along the first axis of a table by index:
 * `out[i] = table[index[i]]`. The result has the index's shape followed by
 * the rest of the table's, so a `[V, D]` embedding table gathered at `[n, T]`
 * token ids gives `[n, T, D]`, and a `[K]` vector of group effects gathered
 * at `[n]` group labels gives `[n]` — a hierarchical model's `z[site]`.
 *
 * The index is not differentiated. It may be a constant, or an input leaf
 * that changes between calls (a mini-batch of token ids); it is read and
 * checked on every forward pass. The adjoint scatter-adds back into the
 * table, so an entry picked twice receives both contributions.
 *
 * @param {Var|number[]|number[][]} tableIn - rank 1 or more
 * @param {Var|number|number[]|number[][]} indexIn - integers in [0, rows)
 * @returns {Var}
 *
 * @example
 * const mu = add(alpha, gather(siteEffect, siteOfEachRow));
 */
export function gather(tableIn, indexIn) {
  const a = toVar(tableIn, 'gather table');
  const index = toVar(indexIn, 'gather index');
  if (a.shape.length === 0) throw new Error('gather: the table cannot be a scalar');
  const shape = [...index.shape, ...a.shape.slice(1)];
  const rows = a.shape[0];
  const width = sizeOf(a.shape.slice(1));
  const m = index.value.data.length;
  const A = a.value.data;
  const I = index.value.data;
  const at = new Int32Array(m);
  const out = new Float64Array(m * width);
  const forward = () => {
    for (let i = 0; i < m; i++) {
      const r = I[i];
      if (!Number.isInteger(r) || r < 0 || r >= rows) {
        throw new Error(`gather: index ${r} at position ${i} is not a row of a table with ${rows} rows`);
      }
      at[i] = r;
      if (width === 1) out[i] = A[r];
      else out.set(A.subarray(r * width, r * width + width), i * width);
    }
  };
  return node({ data: out, shape }, [a, index], (g) => [(into) => {
    for (let i = 0; i < m; i++) {
      const to = at[i] * width;
      const from = i * width;
      for (let j = 0; j < width; j++) into[to + j] += g[from + j];
    }
  }, null], forward, { op: 'gather' });
}

export { zeros, sameShape };
