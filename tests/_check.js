/**
 * The checks every shape op is held to, shared by the rank-3 and nD tests:
 * the adjoint against central finite differences, a compiled plan replayed
 * at a SECOND point against a fresh tape, and that plan written to JSON,
 * cloned the way postMessage would, and rebuilt.
 */
import { expect } from 'vitest';
import { compile, compileFromJSON, valueAndGrad } from '../src/index.js';
import { fdGrad } from './_fd.js';

/** Deterministic values in (-1, 1). */
export function rng(seed = 3) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296) * 2 - 1;
}

/** A nested array of the given shape, filled from `next`. */
export function filled(shape, next) {
  if (shape.length === 1) return Array.from({ length: shape[0] }, next);
  return Array.from({ length: shape[0] }, () => filled(shape.slice(1), next));
}

export const flatten = (x) => (Array.isArray(x) ? x.flatMap(flatten) : [x]);

/** Rebuild a nested array shaped like `like` from a flat list. */
export function unflatten(flat, like) {
  let i = 0;
  const walk = (l) => (Array.isArray(l) ? l.map(walk) : flat[i++]);
  return walk(like);
}

/** Flatten a `{name: nested}` map in key order, and its inverse. */
export function packer(params) {
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

export const rel = (g, r) => Math.abs(g - r) / (1 + Math.abs(r));

/**
 * The three checks every op here must pass. `p1` and `p2` are two points with
 * the same shapes; `feed1`/`feed2` optional inputs.
 */
export function checkOp(f, p1, p2, feed1, feed2) {
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

