const test = require('node:test');
const assert = require('node:assert/strict');
const IF = require('../algorithms/isolation-forest/iforest.js');
const M = require('../assets/js/core/metrics.js');
const D = require('../assets/js/core/datasets.js');

const near = (a, b, eps, msg) => assert.ok(Math.abs(a - b) <= eps, `${msg || ''} ${a} vs ${b}`);

test('c(n): valores de referencia del paper', () => {
  assert.equal(IF.cFactor(1), 0);
  assert.equal(IF.cFactor(2), 1);
  near(IF.cFactor(256), 10.2448, 0.01, 'c(256)');
});

test('determinismo: misma semilla → mismos scores; otra semilla → distintos', () => {
  const { X, n } = D.generate('blob', 300, 0.05, 1);
  const a = IF.fit(X, n, 2, { seed: 5, nEstimators: 30 }).scores(X, n);
  const b = IF.fit(X, n, 2, { seed: 5, nEstimators: 30 }).scores(X, n);
  const c = IF.fit(X, n, 2, { seed: 6, nEstimators: 30 }).scores(X, n);
  assert.deepEqual(Array.from(a), Array.from(b));
  assert.notDeepEqual(Array.from(a), Array.from(c));
});

test('los primeros T árboles no cambian al subir n_estimators', () => {
  const { X, n } = D.generate('blob', 200, 0.05, 2);
  const f10 = IF.fit(X, n, 2, { seed: 3, nEstimators: 10 });
  const f50 = IF.fit(X, n, 2, { seed: 3, nEstimators: 50 });
  for (let t = 0; t < 10; t++) assert.equal(f10.pathLength(t, X, 8), f50.pathLength(t, X, 8));
});

test('estructura del árbol: cada nivel conserva ψ puntos y respeta max_depth', () => {
  const { X, n } = D.generate('moons', 400, 0.05, 3);
  const f = IF.fit(X, n, 2, { seed: 1, nEstimators: 5, maxSamples: 64 });
  assert.equal(f.maxDepth, 6);
  for (const tr of f.trees) {
    let leafTotal = 0;
    for (let i = 0; i < tr.nNodes; i++) {
      assert.ok(tr.depth[i] <= f.maxDepth);
      if (tr.feat[i] < 0) leafTotal += tr.size[i];
      else assert.equal(tr.size[i], tr.size[tr.left[i]] + tr.size[tr.right[i]]);
    }
    assert.equal(leafTotal, 64);
  }
});

test('las anomalías reciben scores mayores (ROC-AUC alto) en los 5 datasets', () => {
  for (const id of Object.keys(D.DEFS)) {
    const { X, y, n } = D.generate(id, 500, 0.06, 7);
    const s = IF.fit(X, n, 2, { seed: 42, nEstimators: 100 }).scores(X, n);
    const auc = M.rocAuc(s, y);
    assert.ok(auc > 0.75, `${id}: AUC=${auc.toFixed(3)}`);
  }
});

test('métricas: AUC por rangos == AUC por trapecios; AP en [0,1]', () => {
  const { X, y, n } = D.generate('twoblobs', 400, 0.08, 4);
  const s = IF.fit(X, n, 2, { seed: 1, nEstimators: 50 }).scores(X, n);
  const c = M.curves(s, y);
  near(c.auc, M.rocAuc(s, y), 1e-9, 'auc');
  assert.ok(c.ap > 0 && c.ap <= 1);
});

test('métricas: caso trivial perfectamente separable', () => {
  const s = [0.9, 0.8, 0.3, 0.2, 0.1], y = [1, 1, 0, 0, 0];
  assert.equal(M.rocAuc(s, y), 1);
  assert.equal(M.averagePrecision(s, y), 1);
  const c = M.confusion(s, y, 0.5);
  assert.deepEqual([c.tp, c.fp, c.fn, c.tn], [2, 0, 0, 3]);
});

test('contamination solo mueve el umbral: marca exactamente round(n·c) puntos', () => {
  const { X, n } = D.generate('blob', 500, 0.06, 9);
  const s = IF.fit(X, n, 2, { seed: 2, nEstimators: 50 }).scores(X, n);
  const thr = IF.threshold(s, 0.1);
  assert.equal(Array.from(s).filter((v) => v > thr).length, 50);
  assert.equal(IF.threshold(s, 'auto'), 0.5);
});

test('max_features=0.5 en 2D → cada árbol usa una sola variable', () => {
  const { X, n } = D.generate('blob', 200, 0.05, 1);
  const f = IF.fit(X, n, 2, { seed: 1, nEstimators: 10, maxFeatures: 0.5 });
  assert.equal(f.nFeatures, 1);
  for (let t = 0; t < 10; t++) {
    const used = new Set();
    for (let i = 0; i < f.trees[t].nNodes; i++) if (f.trees[t].feat[i] >= 0) used.add(f.trees[t].feat[i]);
    assert.ok(used.size <= 1);
  }
});

test('threshold con empates en la frontera marca el grupo empatado (no cero puntos)', () => {
  const scores = new Float64Array([0.9, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2]);
  const thr = IF.threshold(scores, 0.3);            // k = 3, pero hay 4 empatados en 0.8
  const flagged = Array.from(scores).filter((v) => v > thr).length;
  assert.equal(flagged, 5);
  assert.ok(flagged >= 3);
});
