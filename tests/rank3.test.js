/**
 * grad 0.4: rank-3 tensors for batches of sequences, column broadcasting,
 * axis reductions and gather.
 *
 * Every op is held to the same three checks: its adjoint against central
 * finite differences, a compiled plan replayed at a SECOND point against a
 * fresh tape, and that plan written to JSON and rebuilt.
 */
import { describe, expect, it } from 'vitest';
import {
  add, asTensor, compile, compileFromJSON, concat, div, exp, gather, log, matmul, max, maximum,
  mean, mul, reshape, slice, square, sub, sum, tanh, toNested, valueAndGrad,
} from '../src/index.js';
import { fdGrad } from './_fd.js';

// ---- helpers ---------------------------------------------------------------

/** Deterministic values in (-1, 1). */
function rng(seed = 3) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296) * 2 - 1;
}

/** A nested array of the given shape, filled from `next`. */
function filled(shape, next) {
  if (shape.length === 1) return Array.from({ length: shape[0] }, next);
  return Array.from({ length: shape[0] }, () => filled(shape.slice(1), next));
}

const flatten = (x) => (Array.isArray(x) ? x.flatMap(flatten) : [x]);

/** Rebuild a nested array shaped like `like` from a flat list. */
function unflatten(flat, like) {
  let i = 0;
  const walk = (l) => (Array.isArray(l) ? l.map(walk) : flat[i++]);
  return walk(like);
}

/** Flatten a `{name: nested}` map in key order, and its inverse. */
function packer(params) {
  const keys = Object.keys(params);
  return {
    pack: (p) => keys.flatMap((k) => flatten(p[k])),
    unpack: (v) => {
      const out = {};
      let o = 0;
      for (const k of keys) {
        const n = flatten(params[k]).length;
        out[k] = unflatten(v.slice(o, o + n), params[k]);
        o += n;
      }
      return out;
    },
  };
}

const rel = (g, r) => Math.abs(g - r) / (1 + Math.abs(r));

/**
 * The three checks every op here must pass. `p1` and `p2` are two points with
 * the same shapes; `feed1`/`feed2` optional inputs.
 */
function checkOp(f, p1, p2, feed1, feed2) {
  const { pack, unpack } = packer(p1);
  const vg = valueAndGrad(f);

  // 1. Adjoint against finite differences, at p1.
  const got = vg(p1, feed1);
  const fd = fdGrad((v) => vg.value(unpack(v), feed1), pack(p1));
  pack(got.gradient).forEach((g, i) => expect(rel(g, fd[i])).toBeLessThan(1e-6));

  // 2. A plan built at p1 and replayed at p2 matches a fresh tape at p2.
  const c = compile(f);
  c(p1, feed1);
  const fresh = vg(p2, feed2);
  const replay = c(p2, feed2);
  expect(replay.value).toBeCloseTo(fresh.value, 12);
  pack(replay.gradient).forEach((g, i) => expect(g).toBeCloseTo(pack(fresh.gradient)[i], 12));

  // 3. Written out, cloned the way postMessage would, and rebuilt.
  const json = structuredClone(c.toJSON());
  const rebuilt = compileFromJSON(json)(p2, feed2);
  expect(rebuilt.value).toBeCloseTo(fresh.value, 12);
  pack(rebuilt.gradient).forEach((g, i) => expect(g).toBeCloseTo(pack(fresh.gradient)[i], 12));
  return json;
}

const r = rng();
const X = filled([2, 3, 4], r); // [n, T, F]
const X2 = filled([2, 3, 4], r);

// ---- storage ---------------------------------------------------------------

describe('rank-3 storage', () => {
  it('reads and writes an array of matrices', () => {
    const t = asTensor(X);
    expect(t.shape).toEqual([2, 3, 4]);
    expect(t.data[1 * 12 + 2 * 4 + 3]).toBe(X[1][2][3]);
    expect(toNested(t)).toEqual(X);
  });

  it('refuses ragged blocks and rows', () => {
    expect(() => asTensor([[[1, 2]], [[1, 2], [3, 4]]])).toThrow(/block 1 has 2 rows, expected 1/);
    expect(() => asTensor([[[1, 2], [3]]])).toThrow(/block 0, row 1 has length 1, expected 2/);
  });

  it('hands a rank-3 parameter its gradient back nested', () => {
    const { gradient } = valueAndGrad((p) => sum(square(p.x)))({ x: X });
    expect(gradient.x).toEqual(X.map((m) => m.map((row) => row.map((v) => 2 * v))));
  });

  it('keeps a rank-3 plan apart from a matrix plan of the same leading shape', () => {
    // Before 0.4 both signed as "2x3", and the second call replayed the first
    // call's graph on the wrong storage.
    const c = compile((p) => sum(p.x));
    expect(c({ x: X }).value).toBeCloseTo(flatten(X).reduce((a, b) => a + b), 12);
    expect(c({ x: [[1, 2, 3], [4, 5, 6]] }).value).toBe(21);
    expect(c({ x: X2 }).value).toBeCloseTo(flatten(X2).reduce((a, b) => a + b), 12);
  });
});

// ---- reshape ---------------------------------------------------------------

describe('reshape on rank 3', () => {
  it('moves between [n,T,H], [n,T·H] and [n·T,H] without reordering', () => {
    const flat = flatten(X);
    for (const shape of [[2, 12], [6, 4], [2, 3, 4], [24], [4, 3, 2]]) {
      const v = reshape(X, shape);
      expect(v.shape).toEqual(shape);
      expect(Array.from(v.data)).toEqual(flat);
    }
  });

  it('still refuses a size mismatch and rank 4', () => {
    expect(() => reshape(X, [5, 5])).toThrow(/cannot view 24 elements/);
    expect(() => reshape(X, [2, 3, 2, 2])).toThrow(/up to 3 positive integers/);
  });

  it('differentiates and replays', () => {
    const W = filled([4, 5], r);
    // One matmul for every step of every sequence: [n·T, F] · [F, H].
    const f = (p) => sum(square(reshape(matmul(reshape(p.x, [6, 4]), p.w), [2, 3, 5])));
    checkOp(f, { x: X, w: W }, { x: X2, w: filled([4, 5], r) });
  });
});

// ---- slice -----------------------------------------------------------------

describe('slice on rank 3', () => {
  it('takes one time step and a window', () => {
    const step = slice(X, [0, 1, 0], [2, 1, 4]);
    expect(step.shape).toEqual([2, 1, 4]);
    expect(toNested(step.value)).toEqual([[X[0][1]], [X[1][1]]]);

    const win = slice(X, [1, 1, 1], [1, 2, 2]);
    expect(toNested(win.value)).toEqual([[X[1][1].slice(1, 3), X[1][2].slice(1, 3)]]);
  });

  it('accumulates overlapping windows into the same entries', () => {
    // Each slice adds into its parent's gradient in place; two that overlap
    // must both land, on every rank.
    const f = (p) => add(
      sum(square(slice(p.x, [0, 0, 1], [2, 2, 3]))),
      sum(tanh(slice(p.x, [1, 1, 0], [1, 2, 4]))),
      sum(square(slice(p.m, [1, 0], [2, 2]))), sum(slice(p.m, [0, 1], [3, 2])),
      sum(square(slice(p.v, [1], [3]))), sum(slice(p.v, [0], [2])),
    );
    checkOp(f, { x: X, m: filled([3, 3], r), v: filled([4], r) }, { x: X2, m: filled([3, 3], r), v: filled([4], r) });
  });

  it('refuses a window past the end', () => {
    expect(() => slice(X, [0, 2, 0], [2, 2, 4])).toThrow(/exceeds axis 1/);
  });

  it('differentiates and replays: the last step of a tanh recurrence', () => {
    const f = (p) => {
      const step = (t) => reshape(slice(p.x, [0, t, 0], [2, 1, 4]), [2, 4]);
      let h = tanh(step(0));
      for (let t = 1; t < 3; t++) h = tanh(add(mul(h, 0.5), step(t)));
      return sum(square(h));
    };
    const json = checkOp(f, { x: X }, { x: X2 });
    const starts = json.nodes.filter((n) => n.op === 'slice').map((n) => n.args[0][1]);
    expect(starts.sort()).toEqual([0, 1, 2]);
  });
});

// ---- concat ----------------------------------------------------------------

describe('concat on rank 3', () => {
  it('stacks T states of [n,1,H] along time', () => {
    const steps = [0, 1, 2].map((t) => slice(X, [0, t, 0], [2, 1, 4]));
    const back = concat(steps, { axis: 1 });
    expect(back.shape).toEqual([2, 3, 4]);
    expect(Array.from(back.data)).toEqual(flatten(X));
  });

  it('joins along axes 0 and 2', () => {
    const a = filled([1, 3, 4], r);
    expect(toNested(concat([X, a]).value)).toEqual([...X, ...a]);
    const b = filled([2, 3, 1], r);
    const c = concat([X, b], { axis: 2 });
    expect(c.shape).toEqual([2, 3, 5]);
    expect(toNested(c.value)[1][2]).toEqual([...X[1][2], b[1][2][0]]);
  });

  it('refuses parts that disagree off the axis, and mixed ranks', () => {
    expect(() => concat([X, filled([2, 2, 3], r)], { axis: 1 })).toThrow(/agree on every axis but 1/);
    expect(() => concat([X, [[1, 2, 3, 4]]])).toThrow(/must all be rank 3/);
    expect(() => concat([X, X], { axis: 3 })).toThrow(/axis must be 0, 1 or 2/);
  });

  it('differentiates and replays on every axis', () => {
    const B = filled([2, 3, 4], r);
    for (const axis of [0, 1, 2]) {
      const w = filled([axis === 2 ? 8 : 4], r);
      const f = (p) => sum(mul(square(concat([p.a, tanh(p.b)], { axis })), w));
      const json = checkOp(f, { a: X, b: B }, { a: X2, b: X });
      expect(json.nodes.find((n) => n.op === 'concat').args).toEqual([{ axis }]);
    }
  });
});

// ---- broadcasting ----------------------------------------------------------

describe('column broadcast: [n, d] against [n, 1]', () => {
  const H = [[1, 2, 3], [4, 5, 6]];
  const m = [[1], [0]];

  it('scales each row by its own entry, on either side', () => {
    expect(toNested(mul(H, m).value)).toEqual([[1, 2, 3], [0, 0, 0]]);
    expect(toNested(mul(m, H).value)).toEqual([[1, 2, 3], [0, 0, 0]]);
    expect(toNested(sub(m, H).value)).toEqual([[0, -1, -2], [-4, -5, -6]]);
  });

  it('freezes masked rows: m·h + (1 − m)·h_prev', () => {
    const prev = [[9, 9, 9], [7, 7, 7]];
    const next = add(mul(m, H), mul(sub(1, m), prev));
    expect(toNested(next.value)).toEqual([[1, 2, 3], [7, 7, 7]]);
  });

  it('refuses a column of the wrong height and a row-shaped [1, d]', () => {
    expect(() => mul(H, [[1], [0], [1]])).toThrow(/do not match/);
    expect(() => mul(H, [[1, 2, 3]])).toThrow(/do not match/);
  });

  it('differentiates and replays through every binary op', () => {
    const p1 = { h: filled([3, 4], r), c: filled([3, 1], r).map(([v]) => [v + 2]) };
    const p2 = { h: filled([3, 4], r), c: filled([3, 1], r).map(([v]) => [v + 2]) };
    for (const op of [add, sub, mul, div, maximum]) {
      checkOp((p) => sum(square(op(p.h, p.c))), p1, p2);
      checkOp((p) => sum(square(op(p.c, p.h))), p1, p2);
    }
  });

  it('takes the mask as an input that changes between calls', () => {
    const f = (p, i) => sum(square(mul(p.h, i.mask)));
    const p = { h: filled([3, 4], r) };
    checkOp(f, p, p, { mask: [[1], [0], [1]] }, { mask: [[0], [1], [1]] });
  });
});

describe('row broadcast on rank 3: a bias on every step', () => {
  it('adds a [H] vector along the last axis of [n, T, H]', () => {
    const b = [10, 20, 30, 40];
    const y = toNested(add(X, b).value);
    expect(y[1][2]).toEqual(X[1][2].map((v, k) => v + b[k]));
  });

  it('differentiates and replays', () => {
    checkOp((p) => sum(square(tanh(add(p.x, p.b)))), { x: X, b: [0.1, -0.2, 0.3, 0.4] }, { x: X2, b: [1, 2, 3, 4] });
  });
});

// ---- axis reductions -------------------------------------------------------

describe('axis reductions', () => {
  const M = [[1, 5, 2], [7, 3, 3]];

  it('sum, mean and max agree with loops on a matrix', () => {
    expect(toNested(sum(M, { axis: 0 }).value)).toEqual([8, 8, 5]);
    expect(toNested(sum(M, { axis: 1 }).value)).toEqual([8, 13]);
    expect(toNested(mean(M, { axis: 1 }).value)).toEqual([8 / 3, 13 / 3]);
    expect(toNested(max(M, { axis: 0 }).value)).toEqual([7, 5, 3]);
    expect(toNested(max(M, { axis: 1 }).value)).toEqual([5, 7]);
    expect(max(M).data[0]).toBe(7);
  });

  it('keepDims leaves the axis at length 1, ready to broadcast back', () => {
    const s = sum(M, { axis: 1, keepDims: true });
    expect(s.shape).toEqual([2, 1]);
    const share = div(M, s);
    expect(toNested(share.value)[1]).toEqual([7 / 13, 3 / 13, 3 / 13]);
    expect(sum(M, { axis: 0, keepDims: true }).shape).toEqual([1, 3]);
  });

  it('reduces a rank-3 batch over time', () => {
    const m = mean(X, { axis: 1 });
    expect(m.shape).toEqual([2, 4]);
    expect(toNested(m.value)[1][2]).toBeCloseTo((X[1][0][2] + X[1][1][2] + X[1][2][2]) / 3, 14);
  });

  it('without options, still reduces to a scalar and serializes as before', () => {
    const c = compile((p) => add(sum(p.x), mean(p.x)));
    c({ x: M });
    const ops = c.toJSON().nodes.filter((n) => n.kind === 'op' && n.op !== 'add');
    expect(ops).toHaveLength(2);
    expect(ops).toContainEqual({ kind: 'op', op: 'sum', parents: [0] });
    expect(ops).toContainEqual({ kind: 'op', op: 'mean', parents: [0] });
  });

  it('refuses an axis the operand does not have', () => {
    expect(() => sum(M, { axis: 2 })).toThrow(/axis 2 does not exist on a \[2×3\]/);
    expect(() => max(M, { axis: -1 })).toThrow(/axis -1 does not exist/);
    expect(() => mean(M, { keepDims: true })).toThrow(/keepDims needs an axis/);
    expect(() => sum(M, 1)).toThrow(/options must be an object/);
  });

  it('max sends its adjoint to the first winner, and propagates NaN', () => {
    const { gradient } = valueAndGrad((p) => sum(max(p.x, { axis: 1 })))({ x: [[2, 2, 1], [0, 4, 4]] });
    expect(gradient.x).toEqual([[1, 0, 0], [0, 1, 0]]);
    expect(Number.isNaN(max([[1, NaN, 3]], { axis: 1 }).data[0])).toBe(true);
    expect(Number.isNaN(max([NaN, 1]).data[0])).toBe(true);
  });

  it('differentiate and replay along every axis, with and without keepDims', () => {
    const p1 = { x: X }, p2 = { x: X2 };
    for (const op of [sum, mean, max]) {
      for (const axis of [0, 1, 2]) {
        for (const keepDims of [false, true]) {
          const json = checkOp((p) => sum(square(op(tanh(p.x), { axis, keepDims }))), p1, p2);
          expect(json.nodes.find((n) => n.op === op.name && n.args).args).toEqual([{ axis, keepDims }]);
        }
      }
    }
  });

  it('build a stable per-row log-sum-exp that matches the direct form', () => {
    const Z = filled([3, 5], r);
    const lse = (p) => {
      const m = max(p.z, { axis: 1, keepDims: true });
      return add(m, log(sum(exp(sub(p.z, m)), { axis: 1, keepDims: true })));
    };
    const direct = Z.map((row) => Math.log(row.reduce((a, v) => a + Math.exp(v), 0)));
    flatten(toNested(lse({ z: asTensor(Z) }).value)).forEach((v, i) => expect(v).toBeCloseTo(direct[i], 12));
    checkOp((p) => sum(square(lse(p))), { z: Z }, { z: filled([3, 5], r) });
  });
});

// ---- gather ----------------------------------------------------------------

describe('gather', () => {
  const table = [[1, 2], [3, 4], [5, 6]];

  it('picks rows, shaped as the index followed by a row', () => {
    expect(toNested(gather(table, [2, 0, 2]).value)).toEqual([[5, 6], [1, 2], [5, 6]]);
    const tokens = gather(table, [[0, 1], [2, 2]]);
    expect(tokens.shape).toEqual([2, 2, 2]);
    expect(toNested(tokens.value)[1]).toEqual([[5, 6], [5, 6]]);
    expect(toNested(gather([10, 20, 30], [1, 1, 0]).value)).toEqual([20, 20, 10]);
    expect(gather([10, 20, 30], 2).data[0]).toBe(30);
  });

  it('scatter-adds the adjoint, so a row picked twice gets both', () => {
    const { gradient } = valueAndGrad((p) => sum(gather(p.t, [2, 0, 2])))({ t: table });
    expect(gradient.t).toEqual([[1, 1], [0, 0], [2, 2]]);
  });

  it('refuses a bad index, a scalar table, and a result past rank 3', () => {
    expect(() => gather(table, [0, 3])).toThrow(/index 3 at position 1 is not a row of a table with 3 rows/);
    expect(() => gather(table, [0.5])).toThrow(/index 0.5/);
    expect(() => gather(table, [-1])).toThrow(/index -1/);
    expect(() => gather(5, [0])).toThrow(/cannot be a scalar/);
    expect(() => gather(table, [[[0]]])).toThrow(/would have rank 4/);
  });

  it('differentiates and replays: a hierarchical z[site]', () => {
    const f = (p) => sum(square(sub([1, 2, 0.5, -1, 3], add(p.mu, gather(p.z, [0, 1, 1, 2, 0])))));
    checkOp(f, { mu: 0.3, z: [0.1, -0.4, 0.8] }, { mu: -1, z: [2, 1, 0] });
  });

  it('reads a fresh index each call when it is an input: an embedding over mini-batches', () => {
    const E1 = filled([5, 3], r), E2 = filled([5, 3], r);
    const f = (p, i) => sum(square(tanh(gather(p.e, i.ids))));
    checkOp(f, { e: E1 }, { e: E2 }, { ids: [[0, 4], [2, 2]] }, { ids: [[1, 3], [3, 0]] });
  });
});
