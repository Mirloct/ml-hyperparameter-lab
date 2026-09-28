/*
 * Gradient Boosting al estilo XGBoost (Chen & Guestrin, 2016) — implementación didáctica en JavaScript puro.
 *
 * Boosting de Newton de segundo orden sobre pérdida logística (clasificación binaria):
 *   - cada árbol se ajusta al gradiente g_i y hessiano h_i del residuo actual, no a las etiquetas;
 *   - la ganancia de cada corte y el peso de cada hoja usan el objetivo REGULARIZADO de XGBoost:
 *       Gain = ½[ GL²/(HL+λ) + GR²/(HR+λ) − (GL+GR)²/(H+λ) ] − γ        peso de hoja w = −G/(H+λ)
 *   - la predicción final es sigmoide( base + η · Σ árboles ), con η = learning_rate (shrinkage).
 *
 * Convenciones de nombres = las de la librería real (xgboost / sklearn.ensemble.GradientBoostingClassifier):
 *   n_estimators, learning_rate, max_depth, min_child_weight, gamma, lambda (reg_lambda), subsample.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.MLLab = root.MLLab || {}; root.MLLab.GBM = factory(); }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /** PRNG determinista (mulberry32) — misma convención que en Isolation Forest y el VAE. */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  /** Semilla independiente por ronda: los primeros T árboles no cambian al subir n_estimators. */
  function mixSeed(seed, i) {
    let h = (Math.imul(seed | 0, 0x9E3779B1) ^ Math.imul((i + 0x7F4A7C15) | 0, 0x85EBCA6B)) >>> 0;
    h = Math.imul(h ^ (h >>> 15), 0x2C1B3C6D) >>> 0;
    h = Math.imul(h ^ (h >>> 12), 0x297A2D39) >>> 0;
    return (h ^ (h >>> 15)) >>> 0;
  }
  const sigmoid = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
  const clip = (v, a, b) => Math.min(b, Math.max(a, v));

  /** Ajusta UN árbol de regresión a (gradiente, hessiano) con partición exacta (greedy), al estilo XGBoost. */
  function buildTree(X, d, idx, g, h, opts) {
    const { maxDepth, lambda, gamma, minChildWeight, cols } = opts;
    const feat = [], thr = [], left = [], right = [], weight = [];
    const gainAcc = new Float64Array(d);
    const alloc = () => { feat.push(-1); thr.push(0); left.push(-1); right.push(-1); weight.push(0); return feat.length - 1; };
    const sumGH = (ids) => { let G = 0, H = 0; for (let k = 0; k < ids.length; k++) { G += g[ids[k]]; H += h[ids[k]]; } return [G, H]; };
    const leafW = (G, H) => -G / (H + lambda);

    const root = alloc();
    const stack = [[root, idx, 0]];
    while (stack.length) {
      const [node, ids, depth] = stack.pop();
      const [G, H] = sumGH(ids);
      weight[node] = leafW(G, H);
      if (depth >= maxDepth || ids.length < 2 || H < 2 * minChildWeight) continue; // hoja: no hay margen para dos hijos válidos

      // mejor corte exacto: para cada variable, ordena por valor y barre G/H acumulados de izquierda a derecha
      let bestGain = gamma, bestF = -1, bestT = 0, bestL = null, bestR = null;
      for (let ci = 0; ci < cols.length; ci++) {
        const f = cols[ci];
        const sorted = ids.slice().sort((a, b) => X[a * d + f] - X[b * d + f]);
        let Gl = 0, Hl = 0;
        for (let k = 0; k < sorted.length - 1; k++) {
          const i = sorted[k];
          Gl += g[i]; Hl += h[i];
          const v0 = X[i * d + f], v1 = X[sorted[k + 1] * d + f];
          if (v1 === v0) continue;                          // no cortar entre valores idénticos
          const Hr = H - Hl;
          if (Hl < minChildWeight || Hr < minChildWeight) continue;
          const Gr = G - Gl;
          const gain = 0.5 * ((Gl * Gl) / (Hl + lambda) + (Gr * Gr) / (Hr + lambda) - (G * G) / (H + lambda)) - gamma;
          if (gain > bestGain) { bestGain = gain; bestF = f; bestT = (v0 + v1) / 2; bestL = sorted.slice(0, k + 1); bestR = sorted.slice(k + 1); }
        }
      }
      if (bestF < 0) continue;                              // ninguna partición supera γ: se queda como hoja (poda)
      gainAcc[bestF] += bestGain;
      const l = alloc(), r = alloc();
      feat[node] = bestF; thr[node] = bestT; left[node] = l; right[node] = r;
      stack.push([l, bestL, depth + 1], [r, bestR, depth + 1]);
    }
    return {
      feat: Int16Array.from(feat), thr: Float64Array.from(thr),
      left: Int32Array.from(left), right: Int32Array.from(right), weight: Float64Array.from(weight),
      gainAcc, nNodes: feat.length,
    };
  }

  function predictTree(tr, X, d, off) {
    let node = 0;
    while (tr.feat[node] >= 0) node = X[off + tr.feat[node]] < tr.thr[node] ? tr.left[node] : tr.right[node];
    return tr.weight[node];
  }

  function logloss(F, y, n) {
    let s = 0;
    for (let i = 0; i < n; i++) { const p = clip(sigmoid(F[i]), 1e-7, 1 - 1e-7); s -= y[i] ? Math.log(p) : Math.log(1 - p); }
    return s / n;
  }
  function accuracy(F, y, n) {
    let c = 0;
    for (let i = 0; i < n; i++) if ((F[i] > 0 ? 1 : 0) === y[i]) c++;
    return c / n;
  }

  class GBM {
    /**
     * @param {Float64Array} X fila-mayor n×d   @param {Uint8Array} y ∈{0,1}
     * @param opts { nEstimators, learningRate, maxDepth, minChildWeight, gamma, lambda, subsample, seed }
     * @param {Array<{X,y,n}>} [evalSets] conjuntos adicionales (p. ej. validación) evaluados en cada ronda
     */
    static fit(X, y, n, d, opts, evalSets) {
      const o = Object.assign({ nEstimators: 50, learningRate: 0.3, maxDepth: 2, minChildWeight: 1, gamma: 0, lambda: 1, subsample: 1, seed: 0 }, opts);
      const m = new GBM();
      m.d = d; m.opts = o; m.n = n;
      const pbar = clip(Array.from(y).reduce((s, v) => s + v, 0) / n, 1e-6, 1 - 1e-6);
      m.base = Math.log(pbar / (1 - pbar));
      m.trees = [];
      m.gainImportance = new Float64Array(d);
      const F = new Float64Array(n).fill(m.base);
      const sets = evalSets || [];
      const evalF = sets.map((s) => new Float64Array(s.n).fill(m.base));
      m.hist = [];
      const cols = Array.from({ length: d }, (_, i) => i);
      const allIdx = Array.from({ length: n }, (_, i) => i);
      for (let t = 0; t < o.nEstimators; t++) {
        const rng = mulberry32(mixSeed(o.seed, t));
        const g = new Float64Array(n), h = new Float64Array(n);
        for (let i = 0; i < n; i++) { const p = sigmoid(F[i]); g[i] = p - y[i]; h[i] = Math.max(p * (1 - p), 1e-6); }
        let idx = allIdx;
        if (o.subsample < 1) { idx = allIdx.filter(() => rng() < o.subsample); if (idx.length < 4) idx = allIdx; }
        const tree = buildTree(X, d, idx, g, h, { maxDepth: o.maxDepth, lambda: o.lambda, gamma: o.gamma, minChildWeight: o.minChildWeight, cols });
        for (let f = 0; f < d; f++) m.gainImportance[f] += tree.gainAcc[f];
        m.trees.push(tree);
        for (let i = 0; i < n; i++) F[i] += o.learningRate * predictTree(tree, X, d, i * d);
        const row = { round: t + 1, train: logloss(F, y, n), trainAcc: accuracy(F, y, n) };
        sets.forEach((s, si) => {
          for (let i = 0; i < s.n; i++) evalF[si][i] += o.learningRate * predictTree(tree, s.X, d, i * d);
          row['val' + si] = logloss(evalF[si], s.y, s.n);
          row['val' + si + 'Acc'] = accuracy(evalF[si], s.y, s.n);
        });
        m.hist.push(row);
      }
      return m;
    }
    predictRaw(X, i0) { let f = this.base; for (let t = 0; t < this.trees.length; t++) f += this.opts.learningRate * predictTree(this.trees[t], X, this.d, i0); return f; }
    predictProba(X, n) { const out = new Float64Array(n); for (let i = 0; i < n; i++) out[i] = sigmoid(this.predictRaw(X, i * this.d)); return out; }
    /** Predicciones tras cada uno de los primeros T árboles en `roundsList` (crecientes): una pasada, no T pasadas. */
    stagedProba(X, n, roundsList) {
      const F = new Float64Array(n).fill(this.base), out = [];
      let ri = 0;
      for (let t = 0; t < this.trees.length && ri < roundsList.length; t++) {
        for (let i = 0; i < n; i++) F[i] += this.opts.learningRate * predictTree(this.trees[t], X, this.d, i * this.d);
        if (t + 1 === roundsList[ri]) { out.push(Float64Array.from(F, sigmoid)); ri++; }
      }
      return out;
    }
    /** Nodos visitados por un punto en el árbol t (para dibujar su recorrido). */
    pathNodes(t, x) {
      const tr = this.trees[t], path = [0];
      let node = 0;
      while (tr.feat[node] >= 0) { node = x[tr.feat[node]] < tr.thr[node] ? tr.left[node] : tr.right[node]; path.push(node); }
      return path;
    }
  }
  GBM.mulberry32 = mulberry32;
  GBM.sigmoid = sigmoid;
  GBM.logloss = logloss;
  GBM.accuracy = accuracy;
  return GBM;
});
