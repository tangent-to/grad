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

/** Read element i of a tensor that may be a broadcast scalar. @private */
const at = (t, i) => (t.shape.length === 0 ? t.data[0] : t.data[i]);

/**
 * Build a binary elementwise op from its value and its two partial derivatives.
 * @private
 * @param {string} op - name, for error messages
 * @param {(x:number, y:number) => number} f - forward
 * @param {(x:number, y:number, out:number) => number} dfdx
 * @param {(x:number, y:number, out:number) => number} dfdy
 */
function binary(op, f, dfdx, dfdy) {
  return (aIn, bIn) => {
    const a = toVar(aIn, `${op} left operand`);
    const b = toVar(bIn, `${op} right operand`);
    const shape = broadcastShape(a, b, op);
    const n = sizeOf(shape);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = f(at(a.value, i), at(b.value, i));

    return node({ data: out, shape: shape.slice() }, [a, b], (g) => {
      const ga = new Float64Array(n);
      const gb = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const x = at(a.value, i);
        const y = at(b.value, i);
        ga[i] = g[i] * dfdx(x, y, out[i]);
        gb[i] = g[i] * dfdy(x, y, out[i]);
      }
      return [unbroadcast(ga, a.shape), unbroadcast(gb, b.shape)];
    });
  };
}

/**
 * Build a unary elementwise op from its value and derivative.
 * @private
 * @param {string} op
 * @param {(x:number) => number} f
 * @param {(x:number, out:number) => number} df
 */
function unary(op, f, df) {
  return (aIn) => {
    const a = toVar(aIn, `${op} operand`);
    const n = a.value.data.length;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = f(a.value.data[i]);
    return node({ data: out, shape: a.shape.slice() }, [a], (g) => {
      const ga = new Float64Array(n);
      for (let i = 0; i < n; i++) ga[i] = g[i] * df(a.value.data[i], out[i]);
      return [ga];
    });
  };
}

export const add = binary('add', (x, y) => x + y, () => 1, () => 1);
export const sub = binary('sub', (x, y) => x - y, () => 1, () => -1);
export const mul = binary('mul', (x, y) => x * y, (_x, y) => y, (x) => x);
export const div = binary('div', (x, y) => x / y, (_x, y) => 1 / y, (x, y) => -x / (y * y));

export const neg = unary('neg', (x) => -x, () => -1);
export const exp = unary('exp', Math.exp, (_x, out) => out);
export const log = unary('log', Math.log, (x) => 1 / x);
export const sqrt = unary('sqrt', Math.sqrt, (_x, out) => 1 / (2 * out));
export const square = unary('square', (x) => x * x, (x) => 2 * x);
export const tanh = unary('tanh', Math.tanh, (_x, out) => 1 - out * out);
export const sigmoid = unary('sigmoid', (x) => 1 / (1 + Math.exp(-x)), (_x, out) => out * (1 - out));

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
  return unary('pow', (x) => Math.pow(x, k), (x) => k * Math.pow(x, k - 1))(aIn);
}

/**
 * Sum every element to a scalar.
 * @param {Var|number[]|number[][]} aIn
 * @returns {Var}
 */
export function sum(aIn) {
  const a = toVar(aIn, 'sum operand');
  const n = a.value.data.length;
  let s = 0;
  for (let i = 0; i < n; i++) s += a.value.data[i];
  return node({ data: Float64Array.of(s), shape: [] }, [a], (g) => {
    const ga = new Float64Array(n);
    ga.fill(g[0]);
    return [ga];
  });
}

/**
 * Mean of every element, as a scalar.
 * @param {Var|number[]|number[][]} aIn
 * @returns {Var}
 */
export function mean(aIn) {
  const a = toVar(aIn, 'mean operand');
  const n = a.value.data.length;
  let s = 0;
  for (let i = 0; i < n; i++) s += a.value.data[i];
  return node({ data: Float64Array.of(s / n), shape: [] }, [a], (g) => {
    const ga = new Float64Array(n);
    ga.fill(g[0] / n);
    return [ga];
  });
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
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const aip = A[i * k + p];
      if (aip === 0) continue;
      for (let j = 0; j < n; j++) out[i * n + j] += aip * B[p * n + j];
    }
  }

  const shape = vectorRhs ? [m] : [m, n];
  return node({ data: out, shape }, [a, b], (g) => {
    // Ā = Ḡ Bᵀ, B̄ = Aᵀ Ḡ
    const ga = new Float64Array(m * k);
    const gb = new Float64Array(k * n);
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
  });
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
  let s = 0;
  for (let i = 0; i < n; i++) s += u.value.data[i] * v.value.data[i];
  return node({ data: Float64Array.of(s), shape: [] }, [u, v], (g) => {
    const gu = new Float64Array(n);
    const gv = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      gu[i] = g[0] * v.value.data[i];
      gv[i] = g[0] * u.value.data[i];
    }
    return [gu, gv];
  });
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
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[j * m + i] = a.value.data[i * n + j];
  return node({ data: out, shape: [n, m] }, [a], (g) => {
    const ga = new Float64Array(m * n);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) ga[i * n + j] = g[j * m + i];
    return [ga];
  });
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
  for (let i = 0; i < n; i++) out[i] = a.value.data[i * n + i];
  return node({ data: out, shape: [n] }, [a], (g) => {
    const ga = new Float64Array(n * n);
    for (let i = 0; i < n; i++) ga[i * n + i] = g[i];
    return [ga];
  });
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
  const out = a.value.data.slice();
  for (let i = 0; i < n; i++) out[i * n + i] += perRow ? alpha.value.data[i] : alpha.value.data[0];

  return node({ data: out, shape: [n, n] }, [a, alpha], (g) => {
    const gAlpha = new Float64Array(perRow ? n : 1);
    for (let i = 0; i < n; i++) {
      if (perRow) gAlpha[i] = g[i * n + i];
      else gAlpha[0] += g[i * n + i];
    }
    return [g.slice(), gAlpha];
  });
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
  return node({ data: a.value.data.slice(), shape: shape.slice() }, [a], (g) => [g.slice()]);
}

export { zeros, sameShape };
