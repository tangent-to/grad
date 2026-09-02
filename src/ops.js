/**
 * Differentiable operations.
 *
 * Broadcasting is deliberately limited to "scalar against anything". Full
 * numpy-style broadcasting would double the size of every adjoint here for
 * cases the suite's models do not use; a scalar rate against a vector, or a
 * scalar penalty against a matrix, covers them.
 */

import { sameShape, shapeStr, sizeOf, zeros } from './tensor.js';
import { node, requireSameShape, toVar } from './tape.js';

/**
 * Fold a gradient contribution back onto a parent that was broadcast.
 * A scalar parent receives the SUM over the output it was spread across.
 * @private
 */
function unbroadcast(contrib, parentShape) {
  if (parentShape.length !== 0 || contrib.length === 1) return contrib;
  let s = 0;
  for (let i = 0; i < contrib.length; i++) s += contrib[i];
  return Float64Array.of(s);
}

/**
 * Shape of an elementwise result, allowing a scalar on either side.
 * @private
 */
function broadcastShape(a, b, op) {
  if (a.shape.length === 0) return b.shape;
  if (b.shape.length === 0) return a.shape;
  requireSameShape(a, b, op);
  return a.shape;
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
    const A = a.value.data;
    const B = b.value.data;
    const sa = strideOf(a);
    const sb = strideOf(b);
    const out = new Float64Array(n);
    const forward = () => fwd(out, A, B, sa, sb, n);
    forward();

    // Allocated once, refilled on every reverse pass. `backward()` accumulates a
    // node's contributions into its parents before touching the next node, so
    // there is no window in which these could be read stale.
    const ga = new Float64Array(n);
    const gb = new Float64Array(n);
    return node({ data: out, shape: shape.slice() }, [a, b], (g) => {
      bwd(g, ga, gb, A, B, out, sa, sb, n);
      return [unbroadcast(ga, a.shape), unbroadcast(gb, b.shape)];
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
    forward();
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
 * anything.
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
 * anything.
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
 * Sum every element to a scalar.
 * @param {Var|number[]|number[][]} aIn
 * @returns {Var}
 */
export function sum(aIn) {
  const a = toVar(aIn, 'sum operand');
  const n = a.value.data.length;
  const out = new Float64Array(1);
  const forward = () => {
    let s = 0;
    for (let i = 0; i < n; i++) s += a.value.data[i];
    out[0] = s;
  };
  forward();
  const ga = new Float64Array(n);
  return node({ data: out, shape: [] }, [a], (g) => {
    ga.fill(g[0]);
    return [ga];
  }, forward, { op: 'sum' });
}

/**
 * Mean of every element, as a scalar.
 * @param {Var|number[]|number[][]} aIn
 * @returns {Var}
 */
export function mean(aIn) {
  const a = toVar(aIn, 'mean operand');
  const n = a.value.data.length;
  const out = new Float64Array(1);
  const forward = () => {
    let s = 0;
    for (let i = 0; i < n; i++) s += a.value.data[i];
    out[0] = s / n;
  };
  forward();
  const ga = new Float64Array(n);
  return node({ data: out, shape: [] }, [a], (g) => {
    ga.fill(g[0] / n);
    return [ga];
  }, forward, { op: 'mean' });
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
  forward();

  const shape = vectorRhs ? [m] : [m, n];
  const ga = new Float64Array(m * k);
  const gb = new Float64Array(k * n);
  return node({ data: out, shape }, [a, b], (g) => {
    // Ā = Ḡ Bᵀ, B̄ = Aᵀ Ḡ
    ga.fill(0);
    gb.fill(0);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        const gij = g[i * n + j];
        if (gij === 0) continue;
        for (let p = 0; p < k; p++) {
          ga[i * k + p] += gij * B[p * n + j];
          gb[p * n + j] += A[i * k + p] * gij;
        }
      }
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
  forward();
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
  forward();
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
  forward();
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
  forward();

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
 * @param {Var|number[]|number[][]} aIn
 * @param {number[]} shape - rank 0, 1 or 2; must hold the same element count
 * @returns {Var}
 */
export function reshape(aIn, shape) {
  const a = toVar(aIn, 'reshape operand');
  if (!Array.isArray(shape) || shape.length > 2 || shape.some((d) => !Number.isInteger(d) || d < 1)) {
    throw new Error('reshape: shape must be up to two positive integers');
  }
  if (sizeOf(shape) !== a.value.data.length) {
    throw new Error(
      `reshape: cannot view ${a.value.data.length} elements as ${shapeStr(shape)}`,
    );
  }
  const out = new Float64Array(a.value.data.length);
  const forward = () => out.set(a.value.data);
  forward();
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
    forward();
    const ga = new Float64Array(src.length);
    return node({ data: out, shape: [n0] }, [a], (g) => {
      ga.fill(0);
      for (let i = 0; i < n0; i++) ga[s0 + i] = g[i];
      return [ga];
    }, forward, { op: 'slice', args: [start.slice(), size.slice()] });
  }
  const [m, n] = a.shape;
  const [s0, s1] = start;
  const [m0, n0] = size;
  const out = new Float64Array(m0 * n0);
  const forward = () => {
    for (let i = 0; i < m0; i++) {
      for (let j = 0; j < n0; j++) out[i * n0 + j] = src[(s0 + i) * n + (s1 + j)];
    }
  };
  forward();
  const ga = new Float64Array(m * n);
  return node({ data: out, shape: [m0, n0] }, [a], (g) => {
    ga.fill(0);
    for (let i = 0; i < m0; i++) {
      for (let j = 0; j < n0; j++) ga[(s0 + i) * n + (s1 + j)] = g[i * n0 + j];
    }
    return [ga];
  }, forward, { op: 'slice', args: [start.slice(), size.slice()] });
}

/**
 * Assemble scalar or vector Vars into one vector, end to end.
 *
 * The companion to `slice`, and what a vector-valued function needs to return:
 * an ODE right-hand side is written component by component and concatenated.
 *
 * @param {Array<Var|number|number[]>} parts
 * @returns {Var}
 */
export function concat(parts) {
  if (!Array.isArray(parts) || parts.length === 0) {
    throw new Error('concat: needs a non-empty array of parts');
  }
  const vars = parts.map((p, i) => toVar(p, `concat part ${i}`));
  for (const v of vars) {
    if (v.shape.length > 1) {
      throw new Error(`concat: parts must be scalars or vectors, got ${shapeStr(v.shape)}`);
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
  forward();
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

export { zeros, sameShape };
