/*
 * Isolation Forest (Liu, Ting & Zhou, 2008) — implementación didáctica en JavaScript puro.
 *
 * Sigue las mismas convenciones que scikit-learn (IsolationForest):
 *   - max_samples  = ψ, tamaño del submuestreo por árbol
 *   - max_depth    = ceil(log2(ψ)) (derivado, no es un hiperparámetro)
 *   - max_features = fracción de columnas que cada árbol puede usar
 *   - bootstrap    = submuestreo con/sin reemplazo
 *   - contamination = 'auto' (umbral fijo 0.5) o una proporción en (0, 0.5]
 *
 * Score de anomalía (convención del paper, en (0,1)):  s(x, ψ) = 2^( -E[h(x)] / c(ψ) )
 *   s → 1   : muy anómalo (se aísla con pocos cortes)
 *   s ≈ 0.5 : sin estructura clara
 *   s → 0   : muy normal (necesita muchos cortes)
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.MLLab = root.MLLab || {}; root.MLLab.IsolationForest = factory(); }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const EULER = 0.5772156649015329;

  /** c(n): longitud media de camino de una búsqueda fallida en un BST de n nodos. */
  function cFactor(n) {
    if (n <= 1) return 0;
    if (n === 2) return 1;
    return 2 * (Math.log(n - 1) + EULER) - (2 * (n - 1)) / n;
  }

  /** PRNG determinista (mulberry32). */
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

  /** Semilla independiente por árbol: los primeros T árboles no cambian al subir n_estimators. */
  function mixSeed(seed, i) {
    let h = (Math.imul(seed | 0, 0x9E3779B1) ^ Math.imul((i + 0x7F4A7C15) | 0, 0x85EBCA6B)) >>> 0;
    h = Math.imul(h ^ (h >>> 15), 0x2C1B3C6D) >>> 0;
    h = Math.imul(h ^ (h >>> 12), 0x297A2D39) >>> 0;
    return (h ^ (h >>> 15)) >>> 0;
  }

  function drawSample(n, psi, bootstrap, rng) {
    const out = new Int32Array(psi);
    if (bootstrap) {
      for (let i = 0; i < psi; i++) out[i] = Math.floor(rng() * n);
      return out;
    }
    if (psi >= n) { for (let i = 0; i < n; i++) out[i] = i; return out; }
    const pool = new Int32Array(n);
    for (let i = 0; i < n; i++) pool[i] = i;
    for (let i = 0; i < psi; i++) {           // Fisher–Yates parcial
      const j = i + Math.floor(rng() * (n - i));
      const tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
      out[i] = pool[i];
    }
    return out;
  }

  function drawFeatures(d, k, rng) {
    const pool = [];
    for (let i = 0; i < d; i++) pool.push(i);
    for (let i = 0; i < k; i++) {
      const j = i + Math.floor(rng() * (d - i));
      const tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
    }
    return pool.slice(0, k);
  }

  /** Construye un árbol de aislamiento (nodos en arreglos planos para recorrerlos rápido). */
  function buildTree(X, d, sample, feats, maxDepth, rng) {
    const feat = [], thr = [], left = [], right = [], size = [], depth = [];
    const alloc = () => {
      feat.push(-1); thr.push(0); left.push(-1); right.push(-1); size.push(0); depth.push(0);
      return feat.length - 1;
    };
    const stack = [[alloc(), Array.from(sample), 0]];
    while (stack.length) {
      const [node, idx, dep] = stack.pop();
      size[node] = idx.length;
      depth[node] = dep;
      if (dep >= maxDepth || idx.length <= 1) continue;      // hoja

      // 1) elegir al azar una variable no constante en este nodo
      const order = feats.slice();
      let f = -1, lo = 0, hi = 0;
      while (order.length) {
        const cand = order.splice(Math.floor(rng() * order.length), 1)[0];
        let mn = Infinity, mx = -Infinity;
        for (let k = 0; k < idx.length; k++) {
          const v = X[idx[k] * d + cand];
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }
        if (mx > mn) { f = cand; lo = mn; hi = mx; break; }
      }
      if (f < 0) continue;                                    // todos idénticos → hoja

      // 2) valor de corte uniforme entre el mínimo y el máximo
      let t = lo + rng() * (hi - lo);
      if (t <= lo) t = (lo + hi) / 2;
      const L = [], R = [];
      for (let k = 0; k < idx.length; k++) (X[idx[k] * d + f] < t ? L : R).push(idx[k]);

      const l = alloc(), r = alloc();
      feat[node] = f; thr[node] = t; left[node] = l; right[node] = r;
      stack.push([l, L, dep + 1], [r, R, dep + 1]);
    }
    return {
      feat: Int16Array.from(feat), thr: Float64Array.from(thr),
      left: Int32Array.from(left), right: Int32Array.from(right),
      size: Int32Array.from(size), depth: Int16Array.from(depth),
      nNodes: feat.length,
    };
  }

  class IsolationForest {
    /**
     * @param {Float64Array} X  datos en fila-mayor (n × d)
     * @param {number} n
     * @param {number} d
     * @param {object} opts { nEstimators, maxSamples, maxFeatures, bootstrap, seed }
     */
    static fit(X, n, d, opts) {
      const o = Object.assign({ nEstimators: 100, maxSamples: 256, maxFeatures: 1, bootstrap: false, seed: 0 }, opts);
      const f = new IsolationForest();
      f.d = d;
      f.nEstimators = Math.max(1, Math.floor(o.nEstimators));
      f.psi = Math.max(1, Math.min(Math.floor(o.maxSamples), n));
      f.maxDepth = Math.ceil(Math.log2(Math.max(f.psi, 2)));
      f.nFeatures = Math.max(1, Math.min(d, Math.floor(o.maxFeatures * d + 1e-9)));
      f.bootstrap = !!o.bootstrap;
      f.cPsi = cFactor(f.psi);
      f.cTab = new Float64Array(f.psi + 2);
      for (let i = 0; i < f.cTab.length; i++) f.cTab[i] = cFactor(i);
      f.trees = [];
      f.samples = [];
      f.features = [];
      for (let t = 0; t < f.nEstimators; t++) {
        const rng = mulberry32(mixSeed(o.seed, t));
        const sample = drawSample(n, f.psi, f.bootstrap, rng);
        const feats = drawFeatures(d, f.nFeatures, rng);
        f.samples.push(sample);
        f.features.push(feats);
        f.trees.push(buildTree(X, d, sample, feats, f.maxDepth, rng));
      }
      return f;
    }

    /** Profundidad h_t(x) en el árbol t (incluye la corrección c(size) de la hoja). */
    pathLength(t, X, off) {
      const tr = this.trees[t], d = this.d;
      let node = 0;
      while (tr.feat[node] >= 0) {
        node = X[off + tr.feat[node]] < tr.thr[node] ? tr.left[node] : tr.right[node];
      }
      return tr.depth[node] + this.cTab[tr.size[node]];
    }

    /** Nodos visitados por un punto en el árbol t (para dibujar el camino de aislamiento). */
    pathNodes(t, x) {
      const tr = this.trees[t];
      const path = [0];
      let node = 0;
      while (tr.feat[node] >= 0) {
        node = x[tr.feat[node]] < tr.thr[node] ? tr.left[node] : tr.right[node];
        path.push(node);
      }
      return path;
    }

    /** E[h(x)] para cada fila de Xq. */
    meanPathLengths(Xq, m) {
      const d = this.d, T = this.trees.length;
      const acc = new Float64Array(m);
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < m; i++) acc[i] += this.pathLength(t, Xq, i * d);
      }
      for (let i = 0; i < m; i++) acc[i] /= T;
      return acc;
    }

    /**
     * E[h(x)] usando solo los primeros T árboles, para varios T a la vez (una sola pasada).
     * Como los árboles no dependen de n_estimators, el bosque de 300 contiene a los de 3, 10, 30, 100…
     * @param {number[]} checkpoints  T crecientes
     * @returns {Float64Array[]} un arreglo de E[h] por checkpoint
     */
    meanPathAtCheckpoints(Xq, m, checkpoints) {
      const d = this.d, acc = new Float64Array(m), out = [];
      let ci = 0;
      for (let t = 0; t < this.trees.length && ci < checkpoints.length; t++) {
        for (let i = 0; i < m; i++) acc[i] += this.pathLength(t, Xq, i * d);
        if (t + 1 === checkpoints[ci]) {
          const eh = new Float64Array(m);
          for (let i = 0; i < m; i++) eh[i] = acc[i] / (t + 1);
          out.push(eh); ci++;
        }
      }
      return out;
    }

    /** h_t(x) para todos los árboles, dado un único punto. */
    pathLengthsOf(x) {
      const out = new Float64Array(this.trees.length);
      for (let t = 0; t < out.length; t++) out[t] = this.pathLength(t, x, 0);
      return out;
    }

    scoreFromMeanPath(eh) {
      return this.cPsi > 0 ? Math.pow(2, -eh / this.cPsi) : 0.5;
    }

    /** Score de anomalía s ∈ (0,1) para cada fila de Xq. */
    scores(Xq, m) {
      const eh = this.meanPathLengths(Xq, m);
      const s = new Float64Array(m);
      for (let i = 0; i < m; i++) s[i] = this.scoreFromMeanPath(eh[i]);
      return s;
    }

    /**
     * Umbral de decisión. 'auto' → 0.5 (offset_ = -0.5 en sklearn).
     * Numérico → cuantil de los scores de entrenamiento: marca round(n·c) puntos. Si hay empates de score
     * justo en la frontera (típico con pocos árboles o ψ pequeño) el modelo no puede distinguirlos, así que
     * se marca todo el grupo empatado en lugar de ninguno.
     */
    static threshold(scores, contamination) {
      if (contamination === 'auto') return 0.5;
      const n = scores.length;
      const k = Math.min(Math.max(Math.round(n * contamination), 1), n - 1);
      const sorted = Array.from(scores).sort((a, b) => b - a);
      if (sorted[k - 1] === sorted[k]) return sorted[k - 1] - 1e-9;
      return (sorted[k - 1] + sorted[k]) / 2;
    }
  }

  IsolationForest.cFactor = cFactor;
  IsolationForest.mulberry32 = mulberry32;
  return IsolationForest;
});
