/**
 * Writes tests/fixtures/plans-0.3.2.json: compiled plans serialized by grad
 * 0.3.2, with the value and gradient 0.3.2 computed for each. The test that
 * reads it checks that a plan saved by the released version still rebuilds,
 * and gives the same numbers, under the current one.
 *
 * Run once against a copy of 0.3.2's sources:
 *   git archive v0.3.2 src | tar -x -C /tmp/grad-0.3.2
 *   node tests/references/plans_0_3_2.mjs /tmp/grad-0.3.2/src/index.js
 */
import { writeFileSync } from 'node:fs';

const g = await import(process.argv[2]);
const { add, sub, mul, div, sum, mean, square, exp, log, tanh, matmul, transpose, reshape, slice, concat,
  addDiag, logdetPSD, solvePSD, maximum, compile } = g;

const X = [[1, 2], [3, -1], [0.5, 4]];
const objectives = {
  // Row broadcast, matmul, reductions without options.
  regression: [(p, i) => mean(square(sub(add(matmul(i.X, p.w), p.b), i.y))),
    { w: [[0.3], [-0.2]], b: [0.1] }, { X, y: [[1], [2], [0.5]] }],
  // transpose (no args in 0.3.2's plans), slice and both concat axes.
  shapes: [(p) => add(
    sum(square(concat([transpose(slice(p.m, [0, 0], [2, 3])), concat([p.m, p.m], { axis: 1 })], { axis: 1 }))),
    sum(mul(concat([p.m, slice(p.m, [1, 0], [2, 3])]), [1, -2, 0.5])),
  ),
    { m: [[1, 2, 3], [4, 5, 6], [7, 8, 9]] }],
  // reshape, elementwise chain, maximum with a scalar.
  elementwise: [(p) => sum(maximum(tanh(reshape(p.v, [2, 3])), 0.1)), { v: [0.5, -1, 2, 0.1, -0.3, 1.5] }],
  // The GP shape: addDiag, logdetPSD, solvePSD.
  gp: [(p) => {
    const K = addDiag(mul(exp(p.logAmp), [[1, 0.5, 0.2], [0.5, 1, 0.5], [0.2, 0.5, 1]]), exp(p.logNoise));
    const y = [0.3, -0.1, 0.8];
    return add(mul(0.5, logdetPSD(K)), mul(0.5, sum(mul(y, solvePSD(K, y)))));
  }, { logAmp: 0.2, logNoise: -1 }],
  // div and log, a scalar parameter.
  ratio: [(p) => sum(log(div(add(p.a, [1, 2, 3]), p.s))), { a: [0.5, 0.6, 0.7], s: 2 }],
};

const out = { generator: 'plans_0_3_2.mjs', grad: '0.3.2', plans: {} };
for (const [name, [f, params, inputs]] of Object.entries(objectives)) {
  const c = compile(f);
  const r = inputs ? c(params, inputs) : c(params);
  out.plans[name] = { json: c.toJSON(), params, inputs: inputs ?? null, value: r.value, gradient: r.gradient };
}
writeFileSync(new URL('../fixtures/plans-0.3.2.json', import.meta.url), JSON.stringify(out));
console.log(Object.keys(out.plans).length, 'plans written');
