/*
 * Indicadores para evaluar un detector de anomalías SIN etiquetas.
 *   - Estabilidad entre semillas: Jaccard@k del conjunto marcado y correlación de rangos (Spearman).
 *   - Mass-Volume (MV) y Excess-Mass (EM): Goix (2016), estimados por Monte-Carlo en el cuadrado unidad.
 *   - Acuerdo con otro detector (distancia al k-ésimo vecino) medido con Jaccard@k.
 * Score alto = más anómalo en todas las funciones.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.MLLab = root.MLLab || {}; root.MLLab.unsup = factory(); }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /** Rangos promedio (empates comparten el rango medio). */
  function ranks(a) {
    const n = a.length, idx = Array.from({ length: n }, (_, i) => i).sort((x, y) => a[x] - a[y]);
    const r = new Float64Array(n);
    let i = 0;
    while (i < n) {
      let j = i;
      while (j + 1 < n && a[idx[j + 1]] === a[idx[i]]) j++;
      const avg = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) r[idx[k]] = avg;
      i = j + 1;
    }
    return r;
  }
  function pearson(a, b) {
    const n = a.length; let ma = 0, mb = 0;
    for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    let sab = 0, saa = 0, sbb = 0;
    for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
    return saa && sbb ? sab / Math.sqrt(saa * sbb) : 0;
  }
  const spearman = (a, b) => pearson(ranks(a), ranks(b));

  /** Índices de los k scores más altos (los k puntos que el modelo marcaría). */
  function topK(scores, k) {
    return Array.from(scores, (_, i) => i).sort((a, b) => scores[b] - scores[a]).slice(0, k);
  }
  function jaccard(A, B) {
    const sa = new Set(A);
    let inter = 0;
    for (const v of B) if (sa.has(v)) inter++;
    const uni = sa.size + new Set(B).size - inter;
    return uni ? inter / uni : 1;
  }

  /** Estabilidad entre varias corridas (p. ej. semillas): media de todos los pares. */
  function stability(scoresList, k) {
    const tops = scoresList.map((s) => topK(s, k));
    let j = 0, r = 0, c = 0;
    for (let a = 0; a < scoresList.length; a++) for (let b = a + 1; b < scoresList.length; b++) {
      j += jaccard(tops[a], tops[b]); r += spearman(scoresList[a], scoresList[b]); c++;
    }
    return c ? { jaccard: j / c, spearman: r / c } : { jaccard: 1, spearman: 1 };
  }

  /**
   * Curvas MV y EM. dataScores: scores de ANOMALÍA en los datos; uniScores: scores en puntos uniformes del
   * cuadrado unidad (volumen 1). Devuelve las áreas del criterio de Goix (MV: menor es mejor; EM: mayor es mejor).
   */
  function massVolume(dataScores, uniScores) {
    // MV/EM trabajan con una función de "normalidad" (alto = denso): se invierte el signo del score de anomalía.
    const n = dataScores.length, M = uniScores.length;
    const sd = Array.from(dataScores, (v) => -v).sort((a, b) => b - a);
    const su = Array.from(uniScores, (v) => -v).sort((a, b) => b - a);
    const volAt = (u) => { let lo = 0, hi = M; while (lo < hi) { const m = (lo + hi) >> 1; if (su[m] >= u) lo = m + 1; else hi = m; } return lo / M; };

    // MV(α): menor volumen de un conjunto de nivel con masa ≥ α, α ∈ [0.9, 0.999]
    const alphas = [], mv = [];
    for (let a = 0.9; a <= 0.999 + 1e-9; a += 0.001) {
      const k = Math.min(n, Math.ceil(a * n));
      alphas.push(a); mv.push(volAt(sd[k - 1]));
    }
    const mvArea = mv.reduce((s, v) => s + v, 0) / mv.length;   // media de MV sobre el intervalo

    // EM(t) = sup_u { masa(u) − t·vol(u) }, t ∈ [0, EM^-1(0.9)]
    const cand = [];
    for (let q = 1; q <= 100; q++) { const k = Math.max(1, Math.round((q / 100) * n)); cand.push({ mass: k / n, vol: volAt(sd[k - 1]) }); }
    const em = (t) => { let best = 0; for (const c of cand) { const v = c.mass - t * c.vol; if (v > best) best = v; } return best; };
    let tHi = 1;
    while (em(tHi) > 0.9 && tHi < 1e4) tHi *= 2;
    let lo = 0, hi = tHi;
    for (let it = 0; it < 40; it++) { const mid = (lo + hi) / 2; if (em(mid) > 0.9) lo = mid; else hi = mid; }
    const t90 = hi, steps = 60;
    let emArea = 0;
    for (let i = 0; i < steps; i++) emArea += em(((i + 0.5) / steps) * t90) * (t90 / steps);
    return { alphas, mv, mvArea, emArea, t90 };
  }

  /** Detector de referencia: distancia al k-ésimo vecino (2D u otra dimensión, O(n²)). */
  function knnScores(X, n, d, k) {
    const out = new Float64Array(n), buf = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let c = 0; c < d; c++) { const df = X[i * d + c] - X[j * d + c]; s += df * df; }
        buf[j] = s;
      }
      out[i] = Math.sqrt(Array.from(buf).sort((a, b) => a - b)[Math.min(k, n - 1)]);
    }
    return out;
  }

  /** n·d valores uniformes en [0,1]^d con PRNG determinista. */
  function uniformPoints(m, d, rng) {
    const P = new Float64Array(m * d);
    for (let i = 0; i < P.length; i++) P[i] = rng();
    return P;
  }

  return { ranks, spearman, pearson, topK, jaccard, stability, massVolume, knnScores, uniformPoints };
});
