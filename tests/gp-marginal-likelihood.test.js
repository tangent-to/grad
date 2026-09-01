/**
 * The motivating case: the log marginal likelihood of a Gaussian process.
 *
 *     L(θ) = -½ yᵀK⁻¹y - ½ log|K| - n/2 log 2π,     K = K_signal(θ) + αI
 *
 * Hand-deriving ∂L/∂θ is the work this package exists to delete. In
 * @tangent.to/ds it runs to ~90 lines of kernel-specific closed forms, covers
 * only the Matérn family, and every other kernel falls back to a
 * derivative-free search that stops at a visibly worse optimum.
 *
 * Ground truth here is the exact identity
 *
 *     ∂L/∂θ = ½ tr((ααᵀ − K⁻¹) ∂K/∂θ),     α = K⁻¹y
 *
 * computed independently with lina — not finite differences, and not the code
 * under test. The second kernel below has no hand-derived gradient anywhere in
 * the suite; it gets exact derivatives from the same six lines.
 */

import { describe, expect, it } from 'vitest';
import { inv } from '@tangent.to/lina';
import {
  add, addDiag, div, dot, exp, logdetPSD, mul, neg, pow, solvePSD, square, sub,
  valueAndGrad,
} from '../src/index.js';
import { fdGrad } from './_fd.js';

// One-dimensional inputs, deterministic.
const X = [0, 0.6, 1.1, 1.9, 2.4, 3.2, 3.8, 4.5, 5.1, 6.0, 6.7, 7.3];
const y = X.map((x) => Math.sin(x) + 0.15 * Math.cos(4 * x));
const N = X.length;
const NOISE = 0.05;
const LOG2PI = Math.log(2 * Math.PI);

/** Squared distances — a constant of the problem, not a parameter. */
const D2 = X.map((xi) => X.map((xj) => (xi - xj) ** 2));

/**
 * The likelihood, written once. `kernel` returns the signal matrix as a Var;
 * everything after it is shared, and none of it was differentiated by hand.
 */
const logML = (kernel) => (p) => {
  const K = addDiag(kernel(p), NOISE);
  const quad = dot(y, solvePSD(K, y));
  return sub(mul(-0.5, quad), mul(0.5, logdetPSD(K)));
  // the -n/2·log2π constant is dropped: it has zero gradient and would only
  // blur the comparison with the reference below.
};

const rbf = (p) => mul(p.v, exp(neg(div(D2, mul(2, square(p.l))))));
// Rational quadratic. No closed-form gradient for this exists in the suite.
const rq = (p) => mul(p.v, pow(add(1, div(D2, mul(2, mul(p.a, square(p.l))))), -1));

/**
 * ∂L/∂θ by the exact trace identity, computed with lina.
 * @param {number[][]} Ksig - signal kernel at θ
 * @param {number[][][]} dK - ∂K/∂θ for each parameter, in order
 */
function referenceGrad(Ksig, dK) {
  const K = Ksig.map((row, i) => row.map((v, j) => (i === j ? v + NOISE : v)));
  const Kinv = inv(K);
  const alpha = Kinv.map((row) => row.reduce((s, v, j) => s + v * y[j], 0));
  // M = ααᵀ − K⁻¹
  const M = Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (_, j) => alpha[i] * alpha[j] - Kinv[i][j]));
  return dK.map((d) => {
    let tr = 0;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) tr += M[i][j] * d[j][i];
    return 0.5 * tr;
  });
}

describe('GP log marginal likelihood', () => {
  describe('RBF kernel', () => {
    const p = { l: 1.3, v: 0.9 };
    const Ksig = D2.map((row) => row.map((d) => p.v * Math.exp(-d / (2 * p.l ** 2))));

    it('reproduces the exact trace-identity gradient', () => {
      // ∂K/∂l = K·d²/l³   and   ∂K/∂v = K/v
      const dKdl = D2.map((row, i) => row.map((d, j) => Ksig[i][j] * d / p.l ** 3));
      const dKdv = Ksig.map((row) => row.map((k) => k / p.v));
      const [refL, refV] = referenceGrad(Ksig, [dKdl, dKdv]);

      const { gradient } = valueAndGrad(logML(rbf))(p);
      expect(gradient.l).toBeCloseTo(refL, 9);
      expect(gradient.v).toBeCloseTo(refV, 9);
    });

    it('agrees with the likelihood value computed directly', () => {
      const K = Ksig.map((row, i) => row.map((v, j) => (i === j ? v + NOISE : v)));
      const Kinv = inv(K);
      const alpha = Kinv.map((row) => row.reduce((s, v, j) => s + v * y[j], 0));
      const quad = y.reduce((s, yi, i) => s + yi * alpha[i], 0);
      // log|K| via the eigen-free route: 2Σ log L_ii is what logdetPSD computes.
      const { value } = valueAndGrad(logML(rbf))(p);
      const direct = -0.5 * quad - 0.5 * logDet(K);
      expect(value).toBeCloseTo(direct, 9);
    });

    it('matches finite differences', () => {
      const f = (v) => valueAndGrad(logML(rbf))({ l: v[0], v: v[1] }).value;
      const fd = fdGrad(f, [p.l, p.v]);
      const { gradient } = valueAndGrad(logML(rbf))(p);
      expect(gradient.l).toBeCloseTo(fd[0], 6);
      expect(gradient.v).toBeCloseTo(fd[1], 6);
    });
  });

  describe('rational quadratic kernel — no hand-derived gradient exists for it', () => {
    const p = { l: 1.1, v: 0.8, a: 1.7 };
    const Ksig = D2.map((row) =>
      row.map((d) => p.v / (1 + d / (2 * p.a * p.l ** 2))));

    it('reproduces the exact trace-identity gradient in all three parameters', () => {
      // k = v·(1+z)⁻¹ with z = d²/(2a l²)
      // ∂k/∂l = v·(1+z)⁻²·(d²/(a l³));  ∂k/∂v = k/v;  ∂k/∂a = v·(1+z)⁻²·(z/a)
      const dKdl = [], dKdv = [], dKda = [];
      for (let i = 0; i < N; i++) {
        dKdl.push([]); dKdv.push([]); dKda.push([]);
        for (let j = 0; j < N; j++) {
          const d = D2[i][j];
          const z = d / (2 * p.a * p.l ** 2);
          const s = 1 / ((1 + z) ** 2);
          dKdl[i].push(p.v * s * (d / (p.a * p.l ** 3)));
          dKdv[i].push(Ksig[i][j] / p.v);
          dKda[i].push(p.v * s * (z / p.a));
        }
      }
      const [refL, refV, refA] = referenceGrad(Ksig, [dKdl, dKdv, dKda]);

      const { gradient } = valueAndGrad(logML(rq))(p);
      expect(gradient.l).toBeCloseTo(refL, 9);
      expect(gradient.v).toBeCloseTo(refV, 9);
      expect(gradient.a).toBeCloseTo(refA, 9);
    });
  });

  it('differentiates a per-observation noise vector too', () => {
    // Heteroscedastic alpha: what ds gained by hand in its GP, free here.
    const p = { l: 1.3, v: 0.9, noise: X.map((_, i) => 0.02 + 0.01 * i) };
    const build = (q) => {
      const K = addDiag(mul(q.v, exp(neg(div(D2, mul(2, square(q.l)))))), q.noise);
      return sub(mul(-0.5, dot(y, solvePSD(K, y))), mul(0.5, logdetPSD(K)));
    };
    const { gradient } = valueAndGrad(build)(p);
    const f = (v) => valueAndGrad(build)({ ...p, noise: v }).value;
    fdGrad(f, p.noise).forEach((fi, i) => expect(gradient.noise[i]).toBeCloseTo(fi, 6));
  });
});

/** log|A| for the value check, straight from a Cholesky diagonal. */
function logDet(A) {
  const n = A.length;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let j = 0; j < n; j++) {
    let d = A[j][j];
    for (let k = 0; k < j; k++) d -= L[j][k] ** 2;
    L[j][j] = Math.sqrt(d);
    for (let i = j + 1; i < n; i++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = s / L[j][j];
    }
  }
  return 2 * L.reduce((s, row, i) => s + Math.log(row[i]), 0);
}
