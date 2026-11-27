import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

// `npm run test:dist` runs the same suite against the minified bundle that is
// published, instead of the sources: GRAD_TEST_DIST points every test's
// `../src/index.js` import at `dist/index.js`.
const dist = process.env.GRAD_TEST_DIST
  ? [{ find: /^\.\.\/src\/index\.js$/, replacement: fileURLToPath(new URL('./dist/index.js', import.meta.url)) }]
  : [];

export default defineConfig({
  resolve: { alias: dist },
  test: {
    include: ['tests/**/*.test.js'],
  },
});
