/**
 * Row broadcasting: a vector against the rows of a matrix, on every binary
 * elementwise op, either side. Checked the way every adjoint here is checked,
 * against central finite differences, and through `compile` on a second
 * point, since the tiled buffer is refilled on replay and a stale one would
 * pass the first call.
 */

import { describe, expect, it } from 'vitest';
import {
  add, compile, compileFromJSON, div, matmul, maximum, minimum, mul, sub, sum, square, valueAndGrad,
} from '../src/index.js';
import { fdGrad } from './_fd.js';

const M = [[1, 2, 3], [4, 5, 6]];
const ops = { add, sub, mul, div, maximum, minimum };

// Objective over (matrix, vector) with a nonlinearity after, so the adjoint of
// the broadcast is exercised through a non-trivial upstream gradient.
const objective = (op, side) => (p) => {
  const m = p.m, v = p.v;
  return sum(square(side === 'right' ? op(m, v) : op(v, m)));
};
const flatOf = (p) => [...p.m.flat(), ...p.v];
const unflat = (f) => ({ m: [[f[0], f[1], f[2]], [f[3], f[4], f[5]]], v: [f[6], f[7], f[8]] });

describe('broadcast: a vector over the rows of a matrix', () => {
  for (const [name, op] of Object.entries(ops)) {
    for (const side of ['right', 'left']) {
      it(`${name}, vector on the ${side}`, () => {
        const f = objective(op, side);
        // Values kept away from ties for maximum/minimum, and from zero for div.
        const p = { m: M, v: [0.7, 3.2, 8.1] };
        const vg = valueAndGrad(f);
        const got = vg(p);
        const fd = fdGrad((x) => vg.value(unflat(x)), flatOf(p));
        // Relative: the objective is O(10³) here and central differences carry
        // their own rounding error at that magnitude.
        [...got.gradient.m.flat(), ...got.gradient.v].forEach((g, i) =>
          expect(Math.abs(g - fd[i]) / (1 + Math.abs(fd[i]))).toBeLessThan(1e-6));
      });
    }
  }

  it('the forward value is what the tiling promises', () => {
    const out = valueAndGrad((p) => sum(add(p.m, [10, 20, 30])));
    expect(out.value({ m: M })).toBe(1 + 2 + 3 + 4 + 5 + 6 + 2 * 60);
    expect(add(M, [10, 20, 30]).data).toEqual(Float64Array.from([11, 22, 33, 14, 25, 36]));
  });

  it('the adjoint of the vector is the column sum', () => {
    const g = valueAndGrad((p) => sum(mul(p.b, [[1, 2], [3, 4], [5, 6]])))({ b: [1, 1] }).gradient.b;
    expect(g).toEqual([9, 12]);
  });

  it('a bias in a dense layer, the case this exists for', () => {
    const X = [[1, 2], [3, 4], [5, 6]];
    const p = { W: [[0.1, 0.2], [0.3, 0.4]], b: [1, -1] };
    const f = (q) => sum(square(add(matmul(X, q.W), q.b)));
    const vg = valueAndGrad(f);
    const got = vg(p);
    const flat = [0.1, 0.2, 0.3, 0.4, 1, -1];
    const un = (x) => ({ W: [[x[0], x[1]], [x[2], x[3]]], b: [x[4], x[5]] });
    const fd = fdGrad((x) => vg.value(un(x)), flat);
    [...got.gradient.W.flat(), ...got.gradient.b].forEach((g, i) => expect(g).toBeCloseTo(fd[i], 6));
  });

  it('replays correctly at a second point, and serializes', () => {
    const f = (p) => sum(square(sub(mul(p.m, p.v), div(p.v, p.m))));
    const c = compile(f);
    const plain = valueAndGrad(f);
    const points = [{ m: M, v: [1, 2, 3] }, { m: [[2, 2, 2], [3, 3, 3]], v: [0.5, 1.5, 2.5] }];
    for (const p of points) {
      const a = c(p), b = plain(p);
      expect(a.value).toBeCloseTo(b.value, 12);
      a.gradient.v.forEach((g, i) => expect(g).toBeCloseTo(b.gradient.v[i], 12));
    }
    const again = compileFromJSON(c.toJSON());
    expect(again(points[1]).value).toBe(c(points[1]).value);
  });

  it('refuses a vector whose length is not the row length', () => {
    expect(() => add(M, [1, 2])).toThrow(/shapes .* do not match/);
    expect(() => add([1, 2], M)).toThrow(/shapes .* do not match/);
  });

  it('does not broadcast a column: a vector of the row count is refused', () => {
    // [2×3] against [2] is ambiguous with numpy's trailing-axis rule and is
    // not the case this package broadcasts.
    expect(() => add(M, [1, 2])).toThrow();
  });
});
