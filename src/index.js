/**
 * @tangent.to/grad - Reverse-mode automatic differentiation for JavaScript (ESM)
 *
 * An array-valued tape over plain nested arrays, with adjoints for the linear
 * algebra statistical models are made of. Built so that a log-likelihood
 * written once yields its own exact gradient, instead of being hand-derived
 * per model or approximated by finite differences.
 *
 * @example
 * import { valueAndGrad, cholesky, logdetPSD } from '@tangent.to/grad';
 *
 * const nll = (theta) => ... ;            // scalar objective, built from ops
 * const { value, gradient } = valueAndGrad(nll)([1.0, 0.5]);
 */

export { Var, variable } from './tape.js';
export { asTensor, toNested, tensor, zeros, eye } from './tensor.js';
export {
  add, sub, mul, div, neg,
  exp, log, sqrt, square, pow, tanh, sigmoid,
  sum, mean,
  matmul, dot, transpose, diagPart, trace, addDiag, reshape,
} from './ops.js';
export { cholesky, triangularSolve, logdetPSD, solvePSD } from './linalg.js';
export { grad, valueAndGrad, valueAndGradFns } from './api.js';
