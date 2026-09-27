/*
 * Precalcula el "paisaje" de hiperparámetros de Isolation Forest para cada dataset del laboratorio.
 * Para cada configuración (ψ × max_features × bootstrap) y cada T ∈ {3,10,30,100,300}, promedia 3 semillas:
 *   ap, auc (con etiquetas) · mv, jac, spr, agree (sin etiquetas).
 * Uso:  node tools/build-landscape.js   →  algorithms/isolation-forest/landscape.js
 */
const fs = require('fs'), path = require('path');
const IF = require('../algorithms/isolation-forest/iforest.js');
const M = require('../assets/js/core/metrics.js');
const D = require('../assets/js/core/datasets.js');
const U = require('../assets/js/core/unsup.js');
const TU = require('../assets/js/core/tuning.js');

const N = 500, FRAC = 0.06, DATA_SEED = 11, SEEDS = [1, 2, 3], K = 25, NUNI = 1500;
const out = {};
const t0 = Date.now();
for (const ds of Object.keys(D.DEFS)) {
  const { X, y, n } = D.generate(ds, N, FRAC, DATA_SEED);
  const uni = U.uniformPoints(NUNI, 2, IF.mulberry32(99));
  const Xq = new Float64Array((n + NUNI) * 2); Xq.set(X); Xq.set(uni, n * 2);
  const knnTop = U.topK(U.knnScores(X, n, 2, 10), K);
  const cells = {};
  for (const cfg of TU.allConfigs()) {
    const perT = TU.SPACE.T.map(() => ({ sc: [], ap: [], auc: [], mv: [], ag: [] }));
    for (const seed of SEEDS) {
      const f = IF.fit(X, n, 2, { nEstimators: 300, maxSamples: cfg.psi, maxFeatures: cfg.mf, bootstrap: cfg.bs, seed });
      const ehs = f.meanPathAtCheckpoints(Xq, n + NUNI, TU.SPACE.T);
      ehs.forEach((eh, ti) => {
        const s = Float64Array.from(eh, (v) => f.scoreFromMeanPath(v));
        const sd = s.subarray(0, n), su = s.subarray(n);
        const cv = M.curves(sd, y), mvr = U.massVolume(sd, su);
        const P = perT[ti]; P.sc.push(sd.slice()); P.ap.push(cv.ap); P.auc.push(cv.auc); P.mv.push(mvr.mvArea);
        P.ag.push(U.jaccard(U.topK(sd, K), knnTop));
      });
    }
    const mean = (a) => a.reduce((x, v) => x + v, 0) / a.length;
    TU.SPACE.T.forEach((T, ti) => {
      const P = perT[ti], st = U.stability(P.sc, K);
      cells[TU.key(cfg, T)] = [mean(P.ap), mean(P.auc), mean(P.mv), st.jaccard, st.spearman, mean(P.ag)].map((v) => +v.toFixed(4));
    });
  }
  out[ds] = { n, frac: FRAC, dataSeed: DATA_SEED, seeds: SEEDS, k: K, cells };
  console.log(ds, 'listo', ((Date.now() - t0) / 1000).toFixed(1) + 's');
}
const file = path.join(__dirname, '..', 'algorithms', 'isolation-forest', 'landscape.js');
fs.writeFileSync(file, '/* Generado por tools/build-landscape.js — no editar a mano. */\n(window.MLLab = window.MLLab || {}).landscapes = ' + JSON.stringify(out) + ';\n');
console.log('escrito', file, (fs.statSync(file).size / 1024).toFixed(0) + ' KB');
