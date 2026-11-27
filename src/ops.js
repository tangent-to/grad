/**
 * Differentiable operations.
 *
 * Broadcasting is deliberately limited to three cases: a scalar against
 * anything; a vector against the last axis of a matrix or rank-3 tensor
 * (`[n, d]` or `[n, T, d]` with `[d]`); and a column against a matrix
 * (`[n, d]` with `[n, 1]`). Full numpy-style broadcasting would double the
 * size of every adjoint here for cases the suite's models do not use. The
 * scalar case covers a rate against a vector or a penalty against a matrix;
 * the row case is a bias added to every row of `X·W`, and its adjoint is a
 * column sum; the column case is one weight per row — a recurrent network's
 * padding mask applied to its state — and its adjoint is a row sum.
 */

import { MAX_RANK, sameShape, shapeStr, sizeOf, zeros } from './tensor.js';
import { node, requireSameShape, toVar } from './tape.js';

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
 * Fold a gradient contribution back onto a vector parent that was spread over
 * the rows of a matrix: each column sums back into its entry. `out` is the
 * parent-sized buffer the op allocated once.
 * @private
 */
function foldRows(contrib, out) {
  const d = out.length;
  out.fill(0);
  for (let o = 0; o < contrib.length; o += d) {
    for (let j = 0; j < d; j++) out[j] += contrib[o + j];
  }
  return out;
}

/**
 * Fold a gradient contribution back onto an `[n, 1]` column that was spread
 * across the `d` columns of a matrix: each row sums back into its entry.
 * @private
 */
function foldCols(contrib, out, d) {
  for (let i = 0, o = 0; i < out.length; i++, o += d) {
    let s = 0;
    for (let j = 0; j < d; j++) s += contrib[o + j];
    out[i] = s;
  }
  return out;
}

/** Is `v` a vector spread along the last axis of `w`? @private */
const spreadsAsRow = (v, w) =>
  v.shape.length === 1 && w.shape.length >= 2 && w.shape[w.shape.length - 1] === v.shape[0];

/** Is `v` an `[n, 1]` column spread across the columns of the `[n, d]` matrix `w`? @private */
const spreadsAsCol = (v, w) =>
  v.shape.length === 2 && w.shape.length === 2 && v.shape[1] === 1 && v.shape[0] === w.shape[0];

/**
 * Shape of an elementwise result: a scalar on either side, a vector against
 * the last axis of a matrix or rank-3 tensor, or a column against a matrix.
 * @private
 */
function broadcastShape(a, b, op) {
  if (a.shape.length === 0) return b.shape;
  if (b.shape.length === 0) return a.shape;
  if (sameShape(a.shape, b.shape)) return a.shape;
  if (spreadsAsRow(b, a) || spreadsAsCol(b, a)) return a.shape;
  if (spreadsAsRow(a, b) || spreadsAsCol(a, b)) return b.shape;
  requireSameShape(a, b, op);
  return a.shape;
}

/**
 * How an operand is read against the output shape. @private
 * @returns {'same'|'scalar'|'row'|'col'}
 */
function readMode(v, shape) {
  if (v.shape.length === 0) return 'scalar';
  if (sameShape(v.shape, shape)) return 'same';
  return v.shape.length === 1 ? 'row' : 'col';
}

/**
 * Stride for reading an operand elementwise: 0 for a broadcast scalar, which
 * pins every read to element 0, and 1 otherwise. Resolving the broadcast to a
 * stride ONCE lets the kernels below be a single flat loop instead of three
 * variants or a per-element branch.
 * @private
 */
const strideOf = (v) => (v.shape.length === 0 ? 0 : 1);

/**
 * A vector broadcast along the last axis is read through a buffer that holds
 * it tiled to the output's size, refilled on every forward pass; a column
 * broadcast across a matrix likewise, each entry repeated along its row. That
 * keeps the kernels to the two strides above instead of teaching each of them
 * a modulo, and it keeps the backward kernels reading the same buffer.
 * @private
 */
function tile(dst, src) {
  for (let o = 0; o < dst.length; o += src.length) dst.set(src, o);
}

/** Repeat each entry of an `[n, 1]` column `d` times. @private */
function tileCols(dst, src, d) {
  for (let i = 0, o = 0; i < src.length; i++, o += d) dst.fill(src[i], o, o + d);
}

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
    const d = shape.length === 0 ? 1 : shape[shape.length - 1];
    // A row- or column-broadcast operand reads through its tiled copy; the
    // others read their own storage, a scalar at stride 0.
    const ma = readMode(a, shape);
    const mb = readMode(b, shape);
    const tiled = (m) => m === 'row' || m === 'col';
    const A = tiled(ma) ? new Float64Array(n) : a.value.data;
    const B = tiled(mb) ? new Float64Array(n) : b.value.data;
    const sa = strideOf(a);
    const sb = strideOf(b);
    const out = new Float64Array(n);
    const forward = () => {
      if (ma === 'row') tile(A, a.value.data);
      else if (ma === 'col') tileCols(A, a.value.data, d);
      if (mb === 'row') tile(B, b.value.data);
      else if (mb === 'col') tileCols(B, b.value.data, d);
      fwd(out, A, B, sa, sb, n);
    };

    // Allocated once, refilled on every reverse pass. `backward()` accumulates a
    // node's contributions into its parents before touching the next node, so
    // there is no window in which these could be read stale.
    const ga = new Float64Array(n);
    const gb = new Float64Array(n);
    const fa = tiled(ma) ? new Float64Array(a.value.data.length) : null;
    const fb = tiled(mb) ? new Float64Array(b.value.data.length) : null;
    const fold = (g, m, f) => {
      if (m === 'row') return foldRows(g, f);
      if (m === 'col') return foldCols(g, f, d);
      return m === 'scalar' ? unbroadcastScalar(g) : g;
    };
    return node({ data: out, shape: shape.slice() }, [a, b], (g) => {
      bwd(g, ga, gb, A, B, out, sa, sb, n);
      return [fold(ga, ma, fa), fold(gb, mb, fb)];
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
  const scale = mean ? 1 / len : 1;
  const A = a.value.data;
  const out = new Float64Array(outer * inner);
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
    if (mean) for (let i = 0; i < out.length; i++) out[i] *= scale;
  };
  const ga = new Float64Array(A.length);
  return node({ data: out, shape }, [a], (g) => {
    for (let o = 0; o < outer; o++) {
      const base = o * len * inner;
      const ob = o * inner;
      for (let k = 0; k < len; k++) {
        const row = base + k * inner;
        for (let i = 0; i < inner; i++) ga[row + i] = g[ob + i] * scale;
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
 * Matrix product. Accepts matrix × matrix and matrix × vector; a vector on the
 * right is treated as a column, as in numpy.
 *
 * @param {Var|number[][]} aIn - (m × k)
 * @param {Var|number[][]|number[]} bIn - (k × n) or (k)
 * @returns {Var} (m × n) or (m)
 */
export function matmul(aIn, bIn) {
  const a = toVar(aIn, 'matmul left operand');
  const b = toVar(bIn, 'matmul right operand');
  if (a.shape.length !== 2) {
    throw new Error(`matmul: left operand must be a matrix, got ${shapeStr(a.shape)}`);
  }
  if (b.shape.length !== 1 && b.shape.length !== 2) {
    throw new Error(`matmul: right operand must be a matrix or vector, got ${shapeStr(b.shape)}`);
  }
  const [m, k] = a.shape;
  const vectorRhs = b.shape.length === 1;
  const k2 = b.shape[0];
  const n = vectorRhs ? 1 : b.shape[1];
  if (k !== k2) {
    throw new Error(
      `matmul: inner dimensions disagree — ${shapeStr(a.shape)} against ${shapeStr(b.shape)}`,
    );
  }

  const A = a.value.data;
  const B = b.value.data;
  const out = new Float64Array(m * n);
  const forward = () => {
    out.fill(0); // the kernel accumulates
    for (let i = 0; i < m; i++) {
      for (let p = 0; p < k; p++) {
        const aip = A[i * k + p];
        if (aip === 0) continue;
        for (let j = 0; j < n; j++) out[i * n + j] += aip * B[p * n + j];
      }
    }
  };

  const shape = vectorRhs ? [m] : [m, n];
  const ga = new Float64Array(m * k);
  const gb = new Float64Array(k * n);
  const Bt = new Float64Array(k * n);
  const gbt = new Float64Array(k * n);
  return node({ data: out, shape }, [a, b], (g) => {
    // Ā = Ḡ Bᵀ, B̄ = Aᵀ Ḡ, walked as (i, j, p) so a zero entry of Ḡ is
    // skipped whole: it contributes nothing, and a non-finite A or B meeting
    // it does not become NaN. The p loop runs along rows of A and of Ā; B is
    // read through its transpose and B̄ is accumulated transposed, so that
    // loop is contiguous in all four arrays instead of striding by n.
    for (let p = 0; p < k; p++) {
      for (let j = 0; j < n; j++) Bt[j * k + p] = B[p * n + j];
    }
    ga.fill(0);
    gbt.fill(0);
    for (let i = 0; i < m; i++) {
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
 * Matrix transpose.
 * @param {Var|number[][]} aIn
 * @returns {Var}
 */
export function transpose(aIn) {
  const a = toVar(aIn, 'transpose operand');
  if (a.shape.length !== 2) {
    throw new Error(`transpose: needs a matrix, got ${shapeStr(a.shape)}`);
  }
  const [m, n] = a.shape;
  const out = new Float64Array(m * n);
  const forward = () => {
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[j * m + i] = a.value.data[i * n + j];
  };
  const ga = new Float64Array(m * n);
  return node({ data: out, shape: [n, m] }, [a], (g) => {
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) ga[i * n + j] = g[j * m + i];
    return [ga];
  }, forward, { op: 'transpose' });
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
 * the same way: the last is what lets one matmul apply a weight matrix to
 * every time step of every sequence at once.
 *
 * @param {Var|number[]|number[][]} aIn
 * @param {number[]} shape - rank 0 to 3; must hold the same element count
 * @returns {Var}
 */
export function reshape(aIn, shape) {
  const a = toVar(aIn, 'reshape operand');
  if (!Array.isArray(shape) || shape.length > MAX_RANK || shape.some((d) => !Number.isInteger(d) || d < 1)) {
    throw new Error(`reshape: shape must be up to ${MAX_RANK} positive integers`);
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
 * Extract a contiguous submatrix (or subvector) — the differentiable form of
 * `rows.slice(...).map((r) => r.slice(...))`.
 *
 * A structural equation model needs it: Σ is built over latent AND observed
 * variables, then only the observed block is compared with the data.
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
  if (start.length !== rank || size.length !== rank) {
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

  const src = a.value.data;
  if (rank === 1) {
    const [s0] = start;
    const [n0] = size;
    const out = new Float64Array(n0);
    const forward = () => out.set(src.subarray(s0, s0 + n0));
    return node({ data: out, shape: [n0] }, [a], (g) => [(into) => {
      for (let i = 0; i < n0; i++) into[s0 + i] += g[i];
    }], forward, { op: 'slice', args: [start.slice(), size.slice()] });
  }
  if (rank === 3) return slice3(a, start, size);
  const [, n] = a.shape;
  const [s0, s1] = start;
  const [m0, n0] = size;
  const out = new Float64Array(m0 * n0);
  const forward = () => {
    for (let i = 0; i < m0; i++) {
      for (let j = 0; j < n0; j++) out[i * n0 + j] = src[(s0 + i) * n + (s1 + j)];
    }
  };
  return node({ data: out, shape: [m0, n0] }, [a], (g) => [(into) => {
    for (let i = 0; i < m0; i++) {
      for (let j = 0; j < n0; j++) into[(s0 + i) * n + (s1 + j)] += g[i * n0 + j];
    }
  }], forward, { op: 'slice', args: [start.slice(), size.slice()] });
}

/**
 * A rank-3 block: whole runs along the last axis are copied at once, so one
 * time step `[n, 1, F]` of a `[n, T, F]` batch costs n copies of F. Like
 * every slice, its adjoint adds into the parent's gradient in place rather
 * than returning a dense buffer: T time-step slices of one batch would
 * otherwise cost O(T²) in the reverse pass. @private
 */
function slice3(a, start, size) {
  const [, d1, d2] = a.shape;
  const [s0, s1, s2] = start;
  const [n0, n1, n2] = size;
  const src = a.value.data;
  const out = new Float64Array(n0 * n1 * n2);
  const forward = () => {
    for (let i = 0; i < n0; i++) {
      for (let j = 0; j < n1; j++) {
        const from = ((s0 + i) * d1 + (s1 + j)) * d2 + s2;
        out.set(src.subarray(from, from + n2), (i * n1 + j) * n2);
      }
    }
  };
  return node({ data: out, shape: [n0, n1, n2] }, [a], (g) => [(into) => {
    for (let i = 0; i < n0; i++) {
      for (let j = 0; j < n1; j++) {
        const o = (i * n1 + j) * n2;
        const to = ((s0 + i) * d1 + (s1 + j)) * d2 + s2;
        for (let k = 0; k < n2; k++) into[to + k] += g[o + k];
      }
    }
  }], forward, { op: 'slice', args: [start.slice(), size.slice()] });
}

/**
 * Join parts end to end. Scalars and vectors become one vector. Matrices join
 * along `axis`: 0 stacks their rows (same column count), 1 puts them side by
 * side (same row count), which is how a network joins the outputs of two
 * branches for every row of a batch. Rank-3 tensors join along any of their
 * three axes, every other axis agreeing: axis 1 of `[n, 1, H]` states is how
 * a recurrent network stacks its steps back into `[n, T, H]`. The adjoint
 * hands each part its slice.
 *
 * @param {Array<Var|number|number[]|number[][]>} parts
 * @param {{ axis?: number }} [options] - for matrices and rank 3; 0 by default
 * @returns {Var}
 */
export function concat(parts, options = {}) {
  if (!Array.isArray(parts) || parts.length === 0) {
    throw new Error('concat: needs a non-empty array of parts');
  }
  const vars = parts.map((p, i) => toVar(p, `concat part ${i}`));
  const rank = vars[0].shape.length;
  if (rank === 2) return concatMatrices(vars, options.axis ?? 0);
  if (rank === 3) return concat3(vars, options.axis ?? 0);
  for (const v of vars) {
    if (v.shape.length > 1) {
      throw new Error(
        `concat: parts must all be scalars and vectors, all matrices, or all rank 3; got ${shapeStr(v.shape)}`,
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

/** Matrices along axis 0 (rows) or 1 (columns). @private */
function concatMatrices(vars, axis) {
  if (axis !== 0 && axis !== 1) throw new Error('concat: axis must be 0 or 1');
  for (const v of vars) {
    if (v.shape.length !== 2) {
      throw new Error(`concat: parts must all be matrices, got ${shapeStr(v.shape)}`);
    }
    if (v.shape[1 - axis] !== vars[0].shape[1 - axis]) {
      throw new Error(
        `concat: parts must agree on axis ${1 - axis}, got ${shapeStr(vars[0].shape)} and ${shapeStr(v.shape)}`,
      );
    }
  }
  if (axis === 0) {
    // Row-major storage makes stacking rows a plain append.
    const cols = vars[0].shape[1];
    const rows = vars.reduce((a, v) => a + v.shape[0], 0);
    const out = new Float64Array(rows * cols);
    const forward = () => {
      let off = 0;
      for (const v of vars) {
        out.set(v.value.data, off);
        off += v.value.data.length;
      }
    };
    return node({ data: out, shape: [rows, cols] }, vars, (g) => {
      const contribs = [];
      let o = 0;
      for (const v of vars) {
        contribs.push(g.slice(o, o + v.value.data.length));
        o += v.value.data.length;
      }
      return contribs;
    }, forward, { op: 'concat', list: true, args: [{ axis: 0 }] });
  }
  const rows = vars[0].shape[0];
  const widths = vars.map((v) => v.shape[1]);
  const cols = widths.reduce((a, b) => a + b, 0);
  const out = new Float64Array(rows * cols);
  const forward = () => {
    let off = 0;
    for (let k = 0; k < vars.length; k++) {
      const src = vars[k].value.data;
      const w = widths[k];
      for (let i = 0; i < rows; i++) out.set(src.subarray(i * w, i * w + w), i * cols + off);
      off += w;
    }
  };
  const bufs = widths.map((w) => new Float64Array(rows * w));
  return node({ data: out, shape: [rows, cols] }, vars, (g) => {
    let off = 0;
    for (let k = 0; k < vars.length; k++) {
      const w = widths[k];
      const b = bufs[k];
      for (let i = 0; i < rows; i++) b.set(g.subarray(i * cols + off, i * cols + off + w), i * w);
      off += w;
    }
    return bufs;
  }, forward, { op: 'concat', list: true, args: [{ axis: 1 }] });
}

/**
 * Rank-3 tensors along `axis`. Seen as `[outer, block]`, where `outer` is the
 * product of the axes before `axis` and each part's block is everything from
 * `axis` on, every part is a run of `outer` contiguous blocks. @private
 */
function concat3(vars, axis) {
  if (axis !== 0 && axis !== 1 && axis !== 2) throw new Error('concat: axis must be 0, 1 or 2 for rank 3');
  const ref = vars[0].shape;
  for (const v of vars) {
    if (v.shape.length !== 3) {
      throw new Error(`concat: parts must all be rank 3, got ${shapeStr(v.shape)}`);
    }
    for (let k = 0; k < 3; k++) {
      if (k !== axis && v.shape[k] !== ref[k]) {
        throw new Error(
          `concat: parts must agree on every axis but ${axis}, got ${shapeStr(ref)} and ${shapeStr(v.shape)}`,
        );
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
      for (let o = 0; o < outer; o++) out.set(src.subarray(o * b, o * b + b), o * width + off);
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
 * Pick rows of a table by index: `out[i] = table[index[i]]`. The result has
 * the index's shape followed by a row's shape, so a `[V, D]` embedding table
 * gathered at `[n, T]` token ids gives `[n, T, D]`, and a `[K]` vector of
 * group effects gathered at `[n]` group labels gives `[n]` — a hierarchical
 * model's `z[site]`.
 *
 * The index is not differentiated. It may be a constant, or an input leaf
 * that changes between calls (a mini-batch of token ids); it is read and
 * checked on every forward pass. The adjoint scatter-adds back into the
 * table, so a row picked twice receives both contributions.
 *
 * @param {Var|number[]|number[][]} tableIn - rank 1 or 2
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
  if (shape.length > MAX_RANK) {
    throw new Error(
      `gather: a ${shapeStr(index.shape)} index into a ${shapeStr(a.shape)} table ` +
        `would have rank ${shape.length}; the most is ${MAX_RANK}`,
    );
  }
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
  const ga = new Float64Array(A.length);
  return node({ data: out, shape }, [a, index], (g) => {
    ga.fill(0);
    for (let i = 0; i < m; i++) {
      const to = at[i] * width;
      const from = i * width;
      for (let j = 0; j < width; j++) ga[to + j] += g[from + j];
    }
    return [ga, null];
  }, forward, { op: 'gather' });
}

export { zeros, sameShape };
