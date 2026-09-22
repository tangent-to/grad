/**
 * concat on matrices, along rows or columns: the shape, the adjoint against
 * finite differences, the replay, and the serialized axis.
 */
import { describe, expect, it } from 'vitest';
import { compile, compileFromJSON, concat, matmul, mul, sum, square, valueAndGrad } from '../src/index.js';
import { fdGrad } from './_fd.js';

const A = [[1, 2], [3, 4], [5, 6]];        // 3×2
const B = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]; // 3×3
const C = [[7, 8]];                          // 1×2

describe('concat: matrices', () => {
  it('along columns puts parts side by side, row by row', () => {
    const v = concat([A, B], { axis: 1 });
    expect(v.shape).toEqual([3, 5]);
    expect(Array.from(v.data.subarray(0, 5))).toEqual([1, 2, 10, 20, 30]);
    expect(Array.from(v.data.subarray(10, 15))).toEqual([5, 6, 70, 80, 90]);
  });

  it('along rows stacks them', () => {
    const v = concat([A, C]);
    expect(v.shape).toEqual([4, 2]);
    expect(Array.from(v.data.subarray(6, 8))).toEqual([7, 8]);
  });

  it('adjoints match finite differences on both axes', () => {
    const w = [[1, -1, 0.5, 2, -2], [0.3, 0.7, -0.4, 1, 1]];
    const f1 = (p) => sum(square(matmul(concat([p.a, p.b], { axis: 1 }), [[1], [2], [3], [4], [5]])));
    const un1 = (v) => ({ a: [[v[0], v[1]], [v[2], v[3]], [v[4], v[5]]], b: [[v[6], v[7], v[8]], [v[9], v[10], v[11]], [v[12], v[13], v[14]]] });
    const x1 = [...A.flat(), ...B.flat()];
    const vg1 = valueAndGrad(f1);
    const g1 = vg1(un1(x1)).gradient;
    const fd1 = fdGrad((v) => vg1.value(un1(v)), x1);
    const rel = (g, r) => Math.abs(g - r) / (1 + Math.abs(r));
    [...g1.a.flat(), ...g1.b.flat()].forEach((g, i) => expect(rel(g, fd1[i])).toBeLessThan(1e-6));

    const f0 = (p) => sum(square(mul(concat([p.a, p.c]), w[0].slice(0, 2))));
    const un0 = (v) => ({ a: [[v[0], v[1]], [v[2], v[3]], [v[4], v[5]]], c: [[v[6], v[7]]] });
    const x0 = [...A.flat(), ...C.flat()];
    const vg0 = valueAndGrad(f0);
    const g0 = vg0(un0(x0)).gradient;
    const fd0 = fdGrad((v) => vg0.value(un0(v)), x0);
    [...g0.a.flat(), ...g0.c.flat()].forEach((g, i) => expect(rel(g, fd0[i])).toBeLessThan(1e-6));
  });

  it('replays and serializes with its axis', () => {
    const f = (p) => sum(square(concat([p.a, p.b], { axis: 1 })));
    const c = compile(f);
    const p1 = { a: A, b: B }, p2 = { a: A.map((r) => r.map((v) => -v)), b: B };
    expect(c(p2).value).toBeCloseTo(valueAndGrad(f)(p2).value, 10);
    const json = c.toJSON();
    expect(json.nodes.find((n) => n.op === 'concat').args).toEqual([{ axis: 1 }]);
    expect(compileFromJSON(json)(p1).value).toBe(c(p1).value);
  });

  it('refuses mismatched parts and mixed ranks', () => {
    expect(() => concat([A, C], { axis: 1 })).toThrow(/agree on axis 0/);
    expect(() => concat([A, [1, 2]])).toThrow(/must all be matrices/);
    expect(() => concat([[1, 2], A])).toThrow(/scalars and vectors/);
  });
});
