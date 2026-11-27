/**
 * Strided copies: the one loop behind broadcasting, `slice` and `transpose`
 * at any rank.
 *
 * Every tensor is contiguous row-major storage. A view of one — a window of
 * it, its axes permuted, or it repeated along broadcast axes — is the same
 * storage read through a different stride per axis, a stride of 0 repeating
 * an entry. So each of those ops is a copy from a strided view into a
 * contiguous buffer, and its adjoint the reverse: an accumulation from a
 * contiguous gradient back through the same strides.
 *
 * Both loops walk the destination one row (a run along the last axis) at a
 * time with an odometer over the other axes, and specialize the run on the
 * last axis's stride: 1 is a block copy, 0 a fill or a sum, anything else a
 * strided loop. A run is the unit, never an element, for the same reason the
 * elementwise kernels in `ops.js` are whole loops.
 */

/**
 * Row-major strides of a shape: how far apart, in elements, two neighbours
 * along each axis are.
 * @param {number[]} shape
 * @returns {number[]}
 */
export function stridesOf(shape) {
  const s = new Array(shape.length);
  let acc = 1;
  for (let k = shape.length - 1; k >= 0; k--) {
    s[k] = acc;
    acc *= shape[k];
  }
  return s;
}

/**
 * The same walk with fewer, longer rows: length-1 axes dropped (their index
 * is always 0, so their stride never matters), then each pair of adjacent
 * axes merged where the outer stride is the inner stride times the inner
 * length, which is exactly when the two read as one axis. A `[1]` bias
 * broadcast over a `[256, 1]` column becomes one fill of 256 instead of 256
 * fills of 1. Called once when a node is built, not on every pass.
 * @param {number[]} shape
 * @param {number[]} strides
 * @returns {{ shape: number[], strides: number[] }}
 */
export function simplify(shape, strides) {
  const sh = [];
  const st = [];
  for (let k = 0; k < shape.length; k++) {
    if (shape[k] === 1) continue;
    const last = sh.length - 1;
    if (last >= 0 && st[last] === strides[k] * shape[k]) {
      sh[last] *= shape[k];
      st[last] = strides[k];
    } else {
      sh.push(shape[k]);
      st.push(strides[k]);
    }
  }
  return { shape: sh, strides: st };
}

/**
 * Fill contiguous `dst`, of shape `shape`, from `src` read at
 * `base + Σ idx[k]·strides[k]`.
 * @param {Float64Array} dst
 * @param {Float64Array} src
 * @param {number[]} shape - the shape being written
 * @param {number[]} strides - one per axis of `shape`, into `src`
 * @param {number} base - offset into `src` of the first element
 */
export function copyStrided(dst, src, shape, strides, base) {
  const R = shape.length;
  if (R === 0) {
    dst[0] = src[base];
    return;
  }
  const last = shape[R - 1];
  const ls = strides[R - 1];
  const rows = dst.length / last;
  const idx = new Int32Array(R);
  let off = base;
  for (let r = 0, o = 0; r < rows; r++, o += last) {
    if (ls === 1) dst.set(src.subarray(off, off + last), o);
    else if (ls === 0) dst.fill(src[off], o, o + last);
    else for (let j = 0, p = off; j < last; j++, p += ls) dst[o + j] = src[p];
    for (let k = R - 2; k >= 0; k--) {
      off += strides[k];
      if (++idx[k] < shape[k]) break;
      off -= strides[k] * shape[k];
      idx[k] = 0;
    }
  }
}

/**
 * The adjoint of {@link copyStrided}: add contiguous `src`, of shape
 * `shape`, into `dst` at `base + Σ idx[k]·strides[k]`. Where a stride is 0
 * the entries that were repeated sum back into the one they came from.
 * @param {Float64Array} dst
 * @param {Float64Array} src
 * @param {number[]} shape
 * @param {number[]} strides
 * @param {number} base
 */
export function addStrided(dst, src, shape, strides, base) {
  const R = shape.length;
  if (R === 0) {
    dst[base] += src[0];
    return;
  }
  const last = shape[R - 1];
  const ls = strides[R - 1];
  const rows = src.length / last;
  const idx = new Int32Array(R);
  let off = base;
  for (let r = 0, o = 0; r < rows; r++, o += last) {
    if (ls === 1) {
      for (let j = 0; j < last; j++) dst[off + j] += src[o + j];
    } else if (ls === 0) {
      let s = 0;
      for (let j = 0; j < last; j++) s += src[o + j];
      dst[off] += s;
    } else {
      for (let j = 0, p = off; j < last; j++, p += ls) dst[p] += src[o + j];
    }
    for (let k = R - 2; k >= 0; k--) {
      off += strides[k];
      if (++idx[k] < shape[k]) break;
      off -= strides[k] * shape[k];
      idx[k] = 0;
    }
  }
}
