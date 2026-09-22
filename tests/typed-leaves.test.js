/**
 * Tensors as leaves: a parameter given as `{ data: Float64Array, shape }`
 * gets its gradient back in that form, and a plan neither aliases nor
 * converts it. The gradient it returns is a copy, so a caller keeping it
 * across the next call is not reading a buffer being refilled.
 */

import { describe, expect, it } from 'vitest';
import { compile, compileFromJSON, isTensor, matmul, mul, splitValueAndGrad, square, sum, tensor, valueAndGrad } from '../src/index.js';

const W = tensor([0.1, 0.2, 0.3, 0.4], [2, 2]);
const b = tensor([1, -1], [2]);
const X = [[1, 2], [3, 4]];
const f = (p) => sum(square(mul(matmul(X, p.W), p.b)));
const nested = { W: [[0.1, 0.2], [0.3, 0.4]], b: [1, -1] };

describe('typed leaves', () => {
  it('returns the gradient of a tensor parameter as a tensor of the same shape', () => {
    const got = valueAndGrad(f)({ W, b });
    const ref = valueAndGrad(f)(nested);
    expect(got.value).toBe(ref.value);
    expect(isTensor(got.gradient.W)).toBe(true);
    expect(got.gradient.W.shape).toEqual([2, 2]);
    expect(Array.from(got.gradient.W.data)).toEqual(ref.gradient.W.flat());
    expect(Array.from(got.gradient.b.data)).toEqual(ref.gradient.b);
  });

  it('mixes forms in one map, and a single tensor parameter works too', () => {
    const got = valueAndGrad(f)({ W, b: [1, -1] });
    expect(isTensor(got.gradient.W)).toBe(true);
    expect(Array.isArray(got.gradient.b)).toBe(true);
    const single = valueAndGrad((v) => sum(square(v)))(tensor([1, 2, 3], [3]));
    expect(Array.from(single.gradient.data)).toEqual([2, 4, 6]);
  });

  it('a compiled plan replays tensors and returns a fresh gradient each call', () => {
    const c = compile(f);
    const g1 = c({ W, b }).gradient.W;
    const W2 = tensor([1, 1, 1, 1], [2, 2]);
    const g2 = c({ W: W2, b }).gradient.W;
    expect(g1.data).not.toBe(g2.data);                        // a copy, not the plan's buffer
    expect(Array.from(g1.data)).toEqual(valueAndGrad(f)(nested).gradient.W.flat());
    expect(Array.from(W.data)).toEqual([0.1, 0.2, 0.3, 0.4]); // the caller's tensor untouched
  });

  it('a tensor and its nested form share a plan', () => {
    const c = compile(f);
    c({ W, b });
    c(nested);
    expect(c.toJSON().nodes.filter((n) => n.kind === 'param')).toHaveLength(2);
  });

  it('a zero gradient for an untouched tensor parameter is a tensor of zeros', () => {
    const got = valueAndGrad((p) => sum(square(p.b)))({ W, b });
    expect(isTensor(got.gradient.W)).toBe(true);
    expect(Array.from(got.gradient.W.data)).toEqual([0, 0, 0, 0]);
  });

  it('an untouched matrix parameter given as rows gets zeros as rows', () => {
    const got = valueAndGrad((p) => sum(square(p.b)))(nested);
    expect(got.gradient.W).toEqual([[0, 0], [0, 0]]);
  });

  it('works as an input, and through a rebuilt plan', () => {
    const c = compile((p, d) => sum(mul(matmul(d.X, p.W), p.b)));
    const Xt = tensor([1, 2, 3, 4], [2, 2]);
    const a = c({ W, b }, { X: Xt });
    const again = compileFromJSON(c.toJSON());
    expect(again({ W, b }, { X: Xt }).value).toBe(a.value);
    expect(again({ W, b }, { X }).value).toBe(a.value);
  });

  it('splitValueAndGrad compares tensors by value for its one-entry cache', () => {
    let calls = 0;
    const counted = (p) => { calls++; return valueAndGrad(f)(p); };
    const { value, gradient } = splitValueAndGrad(counted);
    value({ W, b });
    gradient({ W: tensor([0.1, 0.2, 0.3, 0.4], [2, 2]), b });   // equal values, fresh object
    expect(calls).toBe(1);
    gradient({ W: tensor([0.1, 0.2, 0.3, 0.5], [2, 2]), b });
    expect(calls).toBe(2);
  });
});
