/**
 * Edges of the n-dimensional work: what the reference fixtures do not reach.
 * Regressions found while testing it, plans saved by the released 0.3.2,
 * error paths, high and degenerate ranks, non-finite values, and a value
 * consumed both directly and broadcast.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  add, asTensor, compile, compileFromJSON, concat, div, gather, matmul, max, mean, mul, reshape,
  slice, square, sub, sum, tanh, toNested, transpose, valueAndGrad, valueAndGradFns, variable,
} from '../src/index.js';
import { checkOp, filled, flatten, rng } from './_check.js';

const r = rng(11);

describe('regressions found while testing', () => {
  it('a cached gradient notices an in-place edit three levels deep', () => {
    // The cache's defensive copy used to stop at two levels, so it shared the
    // caller's inner arrays and saw the edit on both sides.
    const fns = valueAndGradFns((p) => sum(square(p.x)));
    const x = [[[1, 2], [3, 4]]];
    expect(fns.value({ x })).toBe(30);
    x[0][1][1] = 10;
    expect(fns.value({ x })).toBe(114);
    expect(fns.gradient({ x }).x).toEqual([[[2, 4], [6, 20]]]);
  });

  it('a compiled replay refuses a ragged value instead of reusing the last one', () => {
    // The signature reads only the first row, and the write used to trust it:
    // [[1, 2], [3]] answered 10, the previous call's sum.
    const c = compile((p) => sum(p.m));
    expect(c({ m: [[1, 2], [3, 4]] }).value).toBe(10);
    expect(() => c({ m: [[1, 2], [3]] })).toThrow(/parameter "m" is not rectangular \(\[1\] has length 1, expected 2\)/);
    expect(c({ m: [[1, 1], [1, 1]] }).value).toBe(4);

    const c3 = compile((p, i) => sum(mul(p.w, i.X)));
    c3({ w: 2 }, { X: [[[1, 2]], [[3, 4]]] });
    expect(() => c3({ w: 2 }, { X: [[[1, 2]], [[3]]] })).toThrow(/input "X" is not rectangular \(\[1\]\[0\]/);
  });

  it('a compiled replay refuses a tensor whose data does not fill its shape', () => {
    const c = compile((p) => sum(p.t));
    c({ t: { data: new Float64Array(6), shape: [2, 3] } });
    expect(() => c({ t: { data: new Float64Array(5), shape: [2, 3] } })).toThrow(/holds 5 values, not 6/);
  });
});

describe('plans saved by grad 0.3.2', () => {
  const { plans } = JSON.parse(readFileSync(new URL('./fixtures/plans-0.3.2.json', import.meta.url), 'utf8'));

  for (const [name, plan] of Object.entries(plans)) {
    it(`${name} rebuilds and gives 0.3.2's value and gradient`, () => {
      const rebuilt = compileFromJSON(structuredClone(plan.json));
      const got = plan.inputs ? rebuilt(plan.params, plan.inputs) : rebuilt(plan.params);
      expect(got.value).toBeCloseTo(plan.value, 12);
      const want = flatten(Object.values(plan.gradient));
      flatten(Object.values(got.gradient)).forEach((g, i) => expect(g).toBeCloseTo(want[i], 12));
    });
  }
});

describe('broadcasting refusals', () => {
  it('refuses incompatible axes and says which', () => {
    expect(() => add(filled([2, 3, 4], r), filled([3, 2], r)))
      .toThrow(/shapes \[2×3×4\] and \[3×2\] do not match: aligned from the right, axis 2 is 4 against 2/);
  });

  it('refuses a result that is neither operand: the [n] against [n, 1] mistake', () => {
    const y = [1, 2, 3];
    const residual = [[0.1], [0.2], [0.3]];
    expect(() => sub(y, residual)).toThrow(/would both have to be broadcast, to \[3×3\]/);
    expect(() => mul(filled([2, 1], r), filled([1, 3], r))).toThrow(/would both have to be broadcast/);
    // Written on purpose, the outer combination is a product.
    expect(matmul(residual, [y]).shape).toEqual([3, 3]);
  });

  it('accepts every degenerate one-sided case', () => {
    expect(add(5, 2).shape).toEqual([]);
    expect(add([1], 2).shape).toEqual([1]);
    expect(add(filled([3, 1], r), [[1]]).shape).toEqual([3, 1]);
    expect(add([[1]], filled([1, 1, 1], r)).shape).toEqual([1, 1, 1]);
    expect(toNested(mul(filled([1, 3, 1], r), [[[2]]]).value).length).toBe(1);
  });
});

describe('a value used both directly and broadcast', () => {
  it('accumulates both paths: x / sum(x, axis) is a row normalization', () => {
    const f = (p) => sum(square(div(p.x, sum(p.x, { axis: 1, keepDims: true }))));
    const x = filled([3, 4], () => 1 + Math.abs(r()));
    checkOp(f, { x }, { x: filled([3, 4], () => 1 + Math.abs(r())) });
  });

  it('a bias broadcast into two different nodes gets both gradients', () => {
    const f = (p) => add(sum(square(add(p.h, p.b))), sum(tanh(mul(p.g, p.b))));
    checkOp(f, { h: filled([2, 3, 4], r), g: filled([5, 4], r), b: filled([4], r) },
      { h: filled([2, 3, 4], r), g: filled([5, 4], r), b: filled([4], r) });
  });
});

describe('high and degenerate ranks', () => {
  it('chains every structural op through rank 6', () => {
    const shape = [2, 1, 3, 1, 2, 2];
    const f = (p) => {
      const t = transpose(p.x, [5, 3, 1, 0, 4, 2]);               // [2, 1, 1, 2, 2, 3]
      const b = mul(t, p.w);                                      // w is [2, 3]: broadcast
      const s = slice(b, [0, 0, 0, 1, 0, 0], [2, 1, 1, 1, 2, 3]);
      const c = concat([s, s], { axis: 3 });                      // [2, 1, 1, 2, 2, 3]
      return sum(square(max(reshape(c, [4, 6]), { axis: 0 })));
    };
    checkOp(f, { x: filled(shape, r), w: filled([2, 3], r) }, { x: filled(shape, r), w: filled([2, 3], r) });
  });

  it('handles all-singleton shapes', () => {
    const one = [[[0.7]]];
    const f = (p) => add(sum(transpose(p.a)), mean(p.a, { axis: 1, keepDims: true }), max(p.a, { axis: 2 }));
    expect(f({ a: variable(one) }).shape).toEqual([1, 1, 1]); // [] + [1,1,1] + [1,1]
    checkOp((p) => sum(f(p)), { a: one }, { a: [[[-0.3]]] });
    expect(slice(one, [0, 0, 0], [1, 1, 1]).data[0]).toBe(0.7);
  });

  it('reads Float64Array rows at any depth', () => {
    const x = [[Float64Array.of(1, 2), Float64Array.of(3, 4)]];
    expect(asTensor(x).shape).toEqual([1, 2, 2]);
    const c = compile((p) => sum(p.x));
    expect(c({ x }).value).toBe(10);
    expect(c({ x: [[Float64Array.of(1, 1), Float64Array.of(1, 1)]] }).value).toBe(4);
  });

  it('returns a rank-4 tensor parameter its gradient as a tensor', () => {
    const t = { data: Float64Array.from(flatten(filled([2, 1, 3, 2], r))), shape: [2, 1, 3, 2] };
    const { gradient } = valueAndGrad((p) => sum(square(p.t)))({ t });
    expect(gradient.t.shape).toEqual([2, 1, 3, 2]);
    gradient.t.data.forEach((g, i) => expect(g).toBeCloseTo(2 * t.data[i], 14));
  });

  it('keeps a plan per rank-4 shape, alternating without rebuilding wrongly', () => {
    const c = compile((p, i) => sum(mul(p.w, i.X)));
    const full = filled([2, 3, 2, 2], r), part = filled([1, 3, 2, 2], r);
    const w = filled([2], r);
    const direct = (X) => valueAndGrad((p, i) => sum(mul(p.w, i.X)))({ w }, { X });
    for (const X of [full, part, full, part]) expect(c({ w }, { X }).value).toBeCloseTo(direct(X).value, 12);
  });
});

describe('non-finite values', () => {
  it('a zero meeting an infinity in matmul stays zero, batched or not', () => {
    // The kernel skips a zero factor, so 0 · ∞ never becomes NaN; a mask
    // that zeroes a row must keep working when the masked values are huge.
    const A = [[0, 1], [1, 0]];
    const B = [[Infinity, 2], [3, 4]];
    expect(toNested(matmul(A, B).value)).toEqual([[3, 4], [Infinity, 2]]);
    const batched = matmul([A, A], [B, B]);
    expect(flatten(toNested(batched.value))).toEqual([3, 4, Infinity, 2, 3, 4, Infinity, 2]);
  });

  it('a zero gradient meeting an infinity leaves the other gradient finite', () => {
    const { gradient } = valueAndGrad((p) => sum(mul(matmul(p.A, [[Infinity, 1], [1, 1]]), [[0, 1], [0, 1]])))({
      A: [[1, 2], [3, 4]],
    });
    expect(flatten(gradient.A).every(Number.isFinite)).toBe(true);
  });

  it('NaN propagates through broadcasting and reductions', () => {
    expect(Number.isNaN(sum(add(filled([2, 3], r), [0, NaN, 0])).data[0])).toBe(true);
    expect(Number.isNaN(max(mul([[1, 2], [3, 4]], [[NaN], [1]]), { axis: 1 }).data[0])).toBe(true);
  });
});

describe('error paths', () => {
  const X = filled([2, 3, 4], r);

  it('transpose refuses anything but a permutation', () => {
    expect(() => transpose(X, [0, 1])).toThrow(/permutation of 0…2/);
    expect(() => transpose(X, [0, 1, 1])).toThrow(/permutation/);
    expect(() => transpose(X, [0, 1, 3])).toThrow(/permutation/);
    expect(() => transpose(X, 'reverse')).toThrow(/permutation/);
    expect(() => transpose([1, 2, 3])).toThrow(/at least two axes/);
  });

  it('matmul names what disagrees', () => {
    expect(() => matmul([1, 2], [[1], [2]])).toThrow(/at least two axes/);
    expect(() => matmul([[1, 2]], 3)).toThrow(/cannot be a scalar/);
    expect(() => matmul(X, filled([5, 2], r))).toThrow(/inner dimensions disagree/);
    expect(() => matmul(X, filled([3, 4, 2], r))).toThrow(/batch axes disagree/);
    expect(() => matmul(filled([3, 4], r), filled([2, 4, 5], r))).toThrow(/batch axes disagree/);
  });

  it('concat, slice and reshape refuse malformed arguments', () => {
    expect(() => concat([])).toThrow(/non-empty array/);
    expect(() => concat([X, X], { axis: -1 })).toThrow(/axis must be an integer from 0 to 2/);
    expect(() => concat([X, X], { axis: 1.5 })).toThrow(/axis must be an integer/);
    expect(() => concat([X, filled([2, 3], r)])).toThrow(/must all be rank-3 tensors/);
    expect(() => slice(X, [0, 0], [1, 1])).toThrow(/need 3 entries/);
    expect(() => slice(X, [0, -1, 0], [1, 1, 1])).toThrow(/non-negative integer/);
    expect(() => slice(X, [0, 0, 0], 4)).toThrow(/need 3 entries/);
    expect(() => reshape(X, 24)).toThrow(/array of positive integers/);
  });

  it('asTensor refuses empty axes and non-numbers deep down', () => {
    expect(() => asTensor([[[]]])).toThrow(/empty axis/);
    expect(() => asTensor([[[1, 'a']]])).toThrow(/\[0\]\[0\]\[1\] is not a number/);
    expect(() => asTensor([[[1]], 2])).toThrow(/not rectangular/);
  });

  it('gather checks an input index on every replay, not only the first', () => {
    const c = compile((p, i) => sum(gather(p.e, i.ids)));
    const e = filled([3, 2], r);
    c({ e }, { ids: [0, 2] });
    expect(() => c({ e }, { ids: [0, 3] })).toThrow(/index 3 at position 1 is not a row of a table with 3 rows/);
    expect(c({ e }, { ids: [1, 1] }).value).toBeCloseTo(2 * (e[1][0] + e[1][1]), 14);
  });

  it('gather reports a zero gradient for an index passed as a parameter', () => {
    const { gradient } = valueAndGrad((p) => sum(gather(p.e, p.i)))({ e: [[1, 2], [3, 4]], i: [1, 0] });
    expect(gradient.i).toEqual([0, 0]);
    expect(gradient.e).toEqual([[1, 1], [1, 1]]);
  });
});
