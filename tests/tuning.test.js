const test = require('node:test');
const assert = require('node:assert/strict');
const IF = require('../algorithms/isolation-forest/iforest.js');
const M = require('../assets/js/core/metrics.js');
const D = require('../assets/js/core/datasets.js');
const U = require('../assets/js/core/unsup.js');
const TU = require('../assets/js/core/tuning.js');
global.window = { MLLab: {} };
require('../algorithms/isolation-forest/landscape.js');
const LS = global.window.MLLab.landscapes;

test('unsup: jaccard, spearman y estabilidad', () => {
  assert.equal(U.jaccard([1, 2, 3], [1, 2, 3]), 1);
  assert.equal(U.jaccard([1, 2], [3, 4]), 0);
  assert.equal(U.jaccard([1, 2, 3], [2, 3, 4]), 0.5);
  assert.ok(Math.abs(U.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12);
  assert.ok(Math.abs(U.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12);
  const s = Float64Array.from([5, 4, 3, 2, 1]);
  assert.deepEqual(U.stability([s, s, s], 2), { jaccard: 1, spearman: 1 });
});

test('unsup: Mass-Volume separa un buen detector de uno aleatorio (signo correcto)', () => {
  const { X, n } = D.generate('blob', 400, 0.05, 3);
  const rng = IF.mulberry32(1), uni = U.uniformPoints(1500, 2, rng);
  const f = IF.fit(X, n, 2, { seed: 1, nEstimators: 100 });
  const good = U.massVolume(f.scores(X, n), f.scores(uni, 1500));
  const rnd = U.massVolume(Float64Array.from({ length: n }, () => rng()), Float64Array.from({ length: 1500 }, () => rng()));
  assert.ok(good.mvArea < rnd.mvArea - 0.3, `MV good=${good.mvArea} random=${rnd.mvArea}`);
  assert.ok(good.emArea > rnd.emArea * 1.5, `EM good=${good.emArea} random=${rnd.emArea}`);
});

test('iforest: prefijos de árboles == bosque de T árboles', () => {
  const { X, n } = D.generate('moons', 300, 0.05, 2);
  const f300 = IF.fit(X, n, 2, { seed: 4, nEstimators: 300, maxSamples: 64 });
  const f30 = IF.fit(X, n, 2, { seed: 4, nEstimators: 30, maxSamples: 64 });
  const [eh30] = f300.meanPathAtCheckpoints(X, n, [30]);
  const ref = f30.meanPathLengths(X, n);
  for (let i = 0; i < n; i += 17) assert.ok(Math.abs(eh30[i] - ref[i]) < 1e-9);
});

test('paisaje precalculado: una celda coincide con el cálculo en vivo (reproducibilidad)', () => {
  const L = LS.masking, cfg = { psi: 64, mf: 1, bs: false };
  const { X, y, n } = D.generate('masking', L.n, L.frac, L.dataSeed);
  let ap = 0;
  for (const seed of L.seeds) ap += M.curves(IF.fit(X, n, 2, { seed, nEstimators: 30, maxSamples: 64 }).scores(X, n), y).ap;
  ap /= L.seeds.length;
  assert.ok(Math.abs(ap - TU.measure(L, cfg, 30).ap) < 6e-5, `${ap} vs ${TU.measure(L, cfg, 30).ap}`);
});

test('tuning: todos los métodos respetan el presupuesto y devuelven configuraciones válidas', () => {
  const L = LS.blob, valid = new Set(TU.allConfigs().map(TU.cfgId));
  for (const budget of [400, 1500, 3000]) {
    for (const obj of ['ap', 'mv', 'jac']) {
      const f = TU.OBJECTIVES[obj].f;
      const runs = [TU.runGrid(L, f, budget), TU.runRandom(L, f, budget, IF.mulberry32(3)), TU.runBayes(L, f, budget, IF.mulberry32(3)), TU.runHalving(L, f, budget, IF.mulberry32(3))];
      for (const r of runs) {
        assert.ok(r.traj.length > 0);
        assert.ok(r.traj[r.traj.length - 1].cost <= budget);
        r.traj.forEach((p) => assert.ok(valid.has(TU.cfgId(p.cfg))));
      }
    }
  }
});

test('tuning: random y bayes no repiten configuraciones; halving promueve con más árboles', () => {
  const L = LS.masking, f = TU.OBJECTIVES.ap.f;
  for (const run of [TU.runRandom(L, f, 3000, IF.mulberry32(9)), TU.runBayes(L, f, 3000, IF.mulberry32(9))]) {
    const ids = run.evals.map((e) => TU.cfgId(e.cfg));
    assert.equal(new Set(ids).size, ids.length);
  }
  const h = TU.runHalving(L, f, 1500, IF.mulberry32(9));
  const maxT = Math.max(...h.evals.map((e) => e.T)), minT = Math.min(...h.evals.map((e) => e.T));
  assert.ok(maxT > minT);
  assert.ok(h.evals.filter((e) => e.T === maxT).length < h.evals.filter((e) => e.T === minT).length);
});

test('tuning: grid explora todas las dimensiones con presupuesto suficiente', () => {
  const p = TU.gridPlan(16);
  assert.equal(p.mf.length, 2); assert.equal(p.bs.length, 2); assert.ok(p.psi.length * 4 <= 16);
});

test('tuning: en masking el óptimo está en el interior y hay que buscarlo (defecto ≪ oráculo)', () => {
  const r = TU.reference(LS.masking);
  assert.ok(r.oracle.psi < 256 && r.oracle.psi > 8);
  assert.ok(r.oracleAP - r.defaultAP > 0.05);
});
