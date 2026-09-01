/**
 * Differentiable linear algebra.
 *
 * Only TWO primitives carry a hand-written adjoint here: the Cholesky
 * factorization and the triangular solve. Everything a statistical model needs
 * — log-determinants, SPD solves, quadratic forms, Gaussian log-likelihoods —
 * composes from those two and the elementwise ops, and so inherits correct
 * derivatives for free. Each hand-derived adjoint is a place to be subtly
 * wrong, so there are as few as the problem allows.
 *
 * Forward factorizations come from @tangent.to/lina, which is scipy-validated.
 */

import { cholesky as linaCholesky } from '@tangent.to/lina';
import { shapeStr } from './tensor.js';
import { node, toVar } from './tape.js';
import { diagPart, log, mul, sum, transpose } from './ops.js';

/** Solve L X = B in place for lower-triangular L (forward substitution). @private */
function lowerSolve(L, B, n, nrhs) {
  const X = B.slice();
  for (let c = 0; c < nrhs; c++) {
    for (let i = 0; i < n; i++) {
      let s = X[i * nrhs + c];
      for (let k = 0; k < i; k++) s -= L[i * n + k] * X[k * nrhs + c];
      X[i * nrhs + c] = s / L[i * n + i];
    }
  }
  return X;
}

/** Solve U X = B for upper-triangular U (back substitution). @private */
function upperSolve(U, B, n, nrhs) {
  const X = B.slice();
  for (let c = 0; c < nrhs; c++) {
    for (let i = n - 1; i >= 0; i--) {
      let s = X[i * nrhs + c];
      for (let k = i + 1; k < n; k++) s -= U[i * n + k] * X[k * nrhs + c];
      X[i * nrhs + c] = s / U[i * n + i];
    }
  }
  return X;
}

/** Transpose a flat n×n matrix. @private */
function transposeFlat(A, n) {
  const T = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) T[j * n + i] = A[i * n + j];
  return T;
}

/**
 * Cholesky factorization: returns lower-triangular L with A = L Lᵀ.
 *
 * The adjoint is Murray (2016), *Differentiation of the Cholesky
 * decomposition*: with Φ(X) = tril(X) halved on the diagonal,
 *
 *     Ā = L⁻ᵀ Φ(Lᵀ L̄) L⁻¹, symmetrized.
 *
 * `A` is assumed symmetric — only its lower triangle affects the factor — so
 * the returned gradient is symmetrized: it is the derivative with respect to a
 * SYMMETRIC perturbation of A. That is what you want when A is a covariance
 * built by a kernel, which is every use here. Feeding a matrix whose two
 * triangles disagree is a modelling error, and lina's forward rejects it.
 *
 * @param {import('./tape.js').Var|number[][]} aIn - symmetric positive-definite matrix
 * @returns {import('./tape.js').Var} lower-triangular factor (n × n)
 */
export function cholesky(aIn) {
  const a = toVar(aIn, 'cholesky operand');
  if (a.shape.length !== 2 || a.shape[0] !== a.shape[1]) {
    throw new Error(`cholesky: needs a square matrix, got ${shapeStr(a.shape)}`);
  }
  const n = a.shape[0];

  // lina's public entry takes nested rows. Its flat kernel would save the round
  // trip, but is not exported; the conversion is a few percent of an O(n³)
  // factorization, and the validated forward is worth more than that.
  const nested = new Array(n);
  for (let i = 0; i < n; i++) nested[i] = Array.from(a.value.data.subarray(i * n, i * n + n));
  const Lnested = linaCholesky(nested);
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) L[i * n + j] = Lnested[i][j];

  return node({ data: L, shape: [n, n] }, [a], (gL) => {
    // P = Φ(Lᵀ L̄): lower triangle of LᵀL̄, diagonal halved.
    const P = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = 0;
        // (Lᵀ L̄)_ij = Σ_k L_ki L̄_kj, and L is lower so L_ki needs k >= i.
        for (let k = Math.max(i, j); k < n; k++) s += L[k * n + i] * gL[k * n + j];
        P[i * n + j] = i === j ? 0.5 * s : s;
      }
    }
    // Ā = L⁻ᵀ P L⁻¹, in two upper-triangular solves against Lᵀ.
    const Lt = transposeFlat(L, n);
    const X = upperSolve(Lt, P, n, n); // L⁻ᵀ P
    const Y = upperSolve(Lt, transposeFlat(X, n), n, n); // L⁻ᵀ Xᵀ = (L⁻ᵀ P L⁻¹)ᵀ
    const gA = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        // Symmetrize as we transpose back.
        gA[i * n + j] = 0.5 * (Y[j * n + i] + Y[i * n + j]);
      }
    }
    return [gA];
  });
}

/**
 * Solve a triangular system T X = B.
 *
 * @param {import('./tape.js').Var|number[][]} tIn - triangular matrix (n × n)
 * @param {import('./tape.js').Var|number[][]|number[]} bIn - right-hand side (n × k) or (n)
 * @param {Object} [opts]
 * @param {boolean} [opts.lower=true] - whether T is lower triangular
 * @returns {import('./tape.js').Var} solution, shaped like `b`
 */
export function triangularSolve(tIn, bIn, opts = {}) {
  const { lower = true } = opts;
  const t = toVar(tIn, 'triangularSolve matrix');
  const b = toVar(bIn, 'triangularSolve right-hand side');
  if (t.shape.length !== 2 || t.shape[0] !== t.shape[1]) {
    throw new Error(`triangularSolve: needs a square matrix, got ${shapeStr(t.shape)}`);
  }
  const n = t.shape[0];
  const vectorRhs = b.shape.length === 1;
  if (!vectorRhs && b.shape.length !== 2) {
    throw new Error(`triangularSolve: right-hand side must be a matrix or vector, got ${shapeStr(b.shape)}`);
  }
  if (b.shape[0] !== n) {
    throw new Error(
      `triangularSolve: ${shapeStr(t.shape)} cannot be applied to ${shapeStr(b.shape)}`,
    );
  }
  const k = vectorRhs ? 1 : b.shape[1];
  const T = t.value.data;
  const X = lower ? lowerSolve(T, b.value.data, n, k) : upperSolve(T, b.value.data, n, k);

  return node({ data: X, shape: vectorRhs ? [n] : [n, k] }, [t, b], (gX) => {
    // With X = T⁻¹B:  B̄ = T⁻ᵀ X̄  and  T̄ = −B̄ Xᵀ, kept to T's own triangle.
    const Tt = transposeFlat(T, n);
    const gB = lower ? upperSolve(Tt, gX, n, k) : lowerSolve(Tt, gX, n, k);
    const gT = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      const jHi = lower ? i : n - 1;
      const jLo = lower ? 0 : i;
      for (let j = jLo; j <= jHi; j++) {
        let s = 0;
        for (let c = 0; c < k; c++) s += gB[i * k + c] * X[j * k + c];
        gT[i * n + j] = -s;
      }
    }
    return [gT, gB];
  });
}

/**
 * Log-determinant of a symmetric positive-definite matrix.
 *
 * Composed as 2·Σ log Lᵢᵢ rather than given its own adjoint: the derivative
 * (A⁻¹) then falls out of the Cholesky adjoint, with one fewer formula to get
 * wrong and no second factorization.
 *
 * @param {import('./tape.js').Var|number[][]} aIn
 * @returns {import('./tape.js').Var} scalar
 */
export function logdetPSD(aIn) {
  return mul(2, sum(log(diagPart(cholesky(aIn)))));
}

/**
 * Solve A X = B for symmetric positive-definite A, via its Cholesky factor.
 * Composed from `cholesky` and two `triangularSolve`s, so it needs no adjoint
 * of its own.
 *
 * @param {import('./tape.js').Var|number[][]} aIn
 * @param {import('./tape.js').Var|number[][]|number[]} bIn
 * @returns {import('./tape.js').Var}
 */
export function solvePSD(aIn, bIn) {
  const L = cholesky(aIn);
  return triangularSolve(transpose(L), triangularSolve(L, bIn, { lower: true }), { lower: false });
}
