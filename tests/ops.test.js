/**
 * Elementwise ops, reductions and matrix products, each checked against
 * central finite differences.
 */

import { describe, expect, it } from 'vitest';
import {
  add, addDiag, diagPart, div, dot, exp, log, matmul, mean, mul, neg, pow,
  reshape, sigmoid, sqrt, square, sub, sum, tanh, trace, transpose, valueAndGrad, variable,
} from '../src/index.js';
import { fdGrad } from './_fd.js';

/** Check grad(f) against finite differences at x. */
function agrees(build, x, digits = 7) {
  const { value, gradient } = valueAndGrad(build)(x);
  const flatX = Array.isArray(x) ? x : [x];
  const scalarIn = !Array.isArray(x);
  const fd = fdGrad((v) => valueAndGrad(build)(scalarIn ? v[0] : v).value, flatX);
  const g = Array.isArray(gradient) ? gradient : [gradient];
  g.forEach((gi, i) => expect(gi).toBeCloseTo(fd[i], digits));
  return value;
}

describe('elementwise ops', () => {
  const x = [0.7, -1.3, 2.1, 0.4];

  it.each([
    ['add + mul', (v) => sum(mul(add(v, 2), v))],
    ['sub + div', (v) => sum(div(sub(v, 0.25), add(square(v), 3)))],
    ['neg', (v) => sum(neg(v))],
    ['exp', (v) => sum(exp(v))],
    ['log of a positive expression', (v) => sum(log(add(square(v), 1)))],
    ['sqrt', (v) => sum(sqrt(add(square(v), 0.5)))],
    ['square', (v) => sum(square(v))],
    ['tanh', (v) => sum(tanh(v))],
    ['sigmoid', (v) => sum(sigmoid(v))],
    ['pow', (v) => sum(pow(add(square(v), 1), 1.5))],
    ['mean', (v) => mean(exp(v))],
    ['deep chain', (v) => sum(tanh(mul(exp(div(v, 3)), log(add(square(v), 2)))))],
  ])('%s matches finite differences', (_name, build) => {
    agrees(build, x);
  });

  it('broadcasts a scalar against a vector, both ways', () => {
    // The scalar's gradient must be the SUM over the elements it was spread to.
    const { gradient } = valueAndGrad((p) => sum(mul(p.k, [1, 2, 3])))({ k: 2 });
    expect(gradient.k).toBeCloseTo(6, 12);

    const g2 = valueAndGrad((p) => sum(div([2, 4, 6], p.k)))({ k: 2 });
    expect(g2.gradient.k).toBeCloseTo(-3, 10); // d/dk Σ(cᵢ/k) = -Σcᵢ/k² = -12/4
  });

  it('rejects a non-constant exponent with a usable message', () => {
    expect(() => pow(variable([1, 2]), variable(2))).toThrow(/exponent must be a finite number/);
  });
});

describe('reductions and matrix ops', () => {
  it('matmul (matrix × matrix) matches finite differences', () => {
    const B = [[0.5, -1], [2, 0.25], [1, 1.5]];
    const build = (v) => sum(matmul(reshape(v, [2, 3]), B));
    agrees(build, [1, 2, 3, 4, 5, 6]);
  });

  it('matmul (matrix × vector) matches finite differences', () => {
    const build = (v) => sum(square(matmul(reshape(v, [2, 3]), [1, -2, 0.5])));
    agrees(build, [1, 2, 3, 4, 5, 6]);
  });

  it('propagates into BOTH matmul operands', () => {
    // A one-sided adjoint still passes a test that only varies one operand.
    const { gradient } = valueAndGrad((p) => sum(matmul(p.A, p.B)))({
      A: [[1, 2], [3, 4]],
      B: [[5, 6], [7, 8]],
    });
    // d/dA Σ(AB) = 1·Bᵀ summed over rows; d/dB = Aᵀ·1
    expect(gradient.A).toEqual([[11, 15], [11, 15]]);
    expect(gradient.B).toEqual([[4, 4], [6, 6]]);
  });

  it('dot, transpose, diagPart and trace match finite differences', () => {
    agrees((v) => dot(v, [1, -2, 0.5, 3]), [0.7, -1.3, 2.1, 0.4]);
    agrees((v) => sum(square(transpose(reshape(v, [2, 3])))), [1, 2, 3, 4, 5, 6]);
    agrees((v) => sum(exp(diagPart(reshape(v, [3, 3])))), [1, 2, 3, 4, 5, 6, 7, 8, 9].map((z) => z / 10));
    agrees((v) => trace(reshape(v, [3, 3])), [1, 2, 3, 4, 5, 6, 7, 8, 9].map((z) => z / 10));
  });

  it('addDiag differentiates in the matrix and in the scalar', () => {
    const A = [[2, 0.3], [0.3, 1.5]];
    const { gradient } = valueAndGrad((p) => sum(square(addDiag(A, p.alpha))))({ alpha: 0.1 });
    // d/dα Σ(A+αI)² = 2·Σ_diag (A_ii + α) = 2·(2.1 + 1.6)
    expect(gradient.alpha).toBeCloseTo(2 * (2.1 + 1.6), 10);
  });

  it('addDiag takes a per-row noise vector', () => {
    const A = [[2, 0.3], [0.3, 1.5]];
    const { gradient } = valueAndGrad((p) => sum(square(addDiag(A, p.a))))({ a: [0.1, 0.2] });
    expect(gradient.a[0]).toBeCloseTo(2 * 2.1, 10);
    expect(gradient.a[1]).toBeCloseTo(2 * 1.7, 10);
  });
});

describe('valueAndGrad', () => {
  it('returns the gradient in the shape the input arrived in', () => {
    const r = valueAndGrad((p) => add(square(p.mu), square(p.sigma)))({ mu: 3, sigma: 4 });
    expect(r.value).toBe(25);
    expect(r.gradient).toEqual({ mu: 6, sigma: 8 });
  });

  it('gives an untouched parameter a zero gradient rather than omitting it', () => {
    // A sampler indexes every parameter by name; a missing key would crash it.
    const r = valueAndGrad((p) => square(p.used))({ used: 2, unused: [1, 2] });
    expect(r.gradient.used).toBe(4);
    expect(r.gradient.unused).toEqual([0, 0]);
  });

  it('refuses a non-scalar objective, naming the fix', () => {
    expect(() => valueAndGrad((v) => square(v))([1, 2])).toThrow(/must return a scalar/);
  });

  it('refuses an objective that bypassed the ops', () => {
    expect(() => valueAndGrad(() => 42)([1, 2])).toThrow(/must return a Var/);
  });
});
