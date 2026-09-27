/*
 * Optimización de hiperparámetros sobre un "paisaje" tabular precalculado (como en los benchmarks
 * tabulares de HPO): cada configuración ya tiene su calidad medida, así que podemos simular cientos de
 * búsquedas por método y comparar de forma justa cuánto cuesta encontrar una buena configuración.
 *
 * Espacio: ψ (max_samples) × max_features × bootstrap. n_estimators (T) NO se optimiza: se usa como
 * recurso/fidelidad (Successive Halving) o se fija en un valor razonable (Probst & Boulesteix, 2018).
 * Costo de una evaluación = nº de árboles entrenados (T).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.MLLab = root.MLLab || {}; root.MLLab.tuning = factory(); }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const SPACE = {
    psi: [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 500],
    mf: [0.5, 1],
    bs: [false, true],
    T: [3, 10, 30, 100, 300],
  };
  const DEFAULT_CFG = { psi: 256, mf: 1, bs: false };           // valores por defecto de scikit-learn
  const EVAL_T = 100;                                           // fidelidad estándar (n_estimators por defecto)
  const FULL_T = 300;                                           // fidelidad de despliegue (verdad de referencia)

  const key = (c, T) => `${c.psi}|${c.mf}|${c.bs ? 1 : 0}|${T}`;
  function allConfigs() {
    const out = [];
    for (const psi of SPACE.psi) for (const mf of SPACE.mf) for (const bs of SPACE.bs) out.push({ psi, mf, bs });
    return out;
  }
  const cfgId = (c) => `${c.psi}|${c.mf}|${c.bs ? 1 : 0}`;
  const encode = (c) => [
    (Math.log2(c.psi) - Math.log2(SPACE.psi[0])) / (Math.log2(SPACE.psi[SPACE.psi.length - 1]) - Math.log2(SPACE.psi[0])),
    c.mf === 1 ? 1 : 0, c.bs ? 1 : 0,
  ];

  /** Accede a la medición de una configuración a una fidelidad T. cells: { "psi|mf|bs|T": [ap,auc,mv,jac,spr,agree] } */
  const FIELDS = ['ap', 'auc', 'mv', 'jac', 'spr', 'agree'];
  function measure(L, c, T) {
    const v = L.cells[key(c, T)];
    const o = {}; FIELDS.forEach((f, i) => { o[f] = v[i]; });
    return o;
  }
  /** Objetivo que "ve" el tuner (siempre mayor = mejor). */
  const OBJECTIVES = {
    ap: { label: 'AP (con etiquetas)', needsLabels: true, f: (m) => m.ap },
    mv: { label: 'Mass-Volume (sin etiquetas)', needsLabels: false, f: (m) => -m.mv },
    jac: { label: 'Estabilidad Jaccard@k (sin etiquetas)', needsLabels: false, f: (m) => m.jac },
    agree: { label: 'Acuerdo con kNN (sin etiquetas)', needsLabels: false, f: (m) => m.agree },
  };

  /* ---------- utilidades ---------- */
  function shuffle(a, rng) { const r = a.slice(); for (let i = r.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [r[i], r[j]] = [r[j], r[i]]; } return r; }
  const argmax = (arr, f) => { let bi = -1, bv = -Infinity; arr.forEach((x, i) => { const v = f(x); if (v > bv) { bv = v; bi = i; } }); return bi; };

  /**
   * Cada método devuelve una trayectoria: [{cost, cfg}] donde cfg es el INCUMBENTE (la configuración que el
   * tuner recomendaría) después de gastar `cost` árboles. También devuelve `evals` (lo probado, en orden).
   */
  function runRandom(L, obj, budget, rng) {
    const evals = [], traj = [];
    let cost = 0, best = null, bestV = -Infinity;
    for (const c of shuffle(allConfigs(), rng)) {
      if (cost + EVAL_T > budget) break;
      const v = obj(measure(L, c, EVAL_T)); cost += EVAL_T; evals.push({ cfg: c, T: EVAL_T, v, cost });
      if (v > bestV) { bestV = v; best = c; }
      traj.push({ cost, cfg: best });
    }
    return { traj, evals };
  }

  function gridPlan(B) {
    const b = B >= 8 ? 2 : B >= 4 ? 2 : 1, c = B >= 8 ? 2 : 1;
    let a = Math.max(1, Math.min(SPACE.psi.length, Math.floor(B / (b * c))));
    const pick = (arr, k) => (k >= arr.length ? arr.slice() : k === 1 ? [arr[Math.floor(arr.length / 2)]] : Array.from({ length: k }, (_, i) => arr[Math.round((i * (arr.length - 1)) / (k - 1))]));
    return { psi: pick(SPACE.psi, a), mf: pick(SPACE.mf, b), bs: pick(SPACE.bs, c) };
  }
  function runGrid(L, obj, budget) {
    const plan = gridPlan(Math.floor(budget / EVAL_T)), evals = [], traj = [];
    let cost = 0, best = null, bestV = -Infinity;
    outer: for (const psi of plan.psi) for (const mf of plan.mf) for (const bs of plan.bs) {
      if (cost + EVAL_T > budget) break outer;
      const c = { psi, mf, bs }, v = obj(measure(L, c, EVAL_T)); cost += EVAL_T; evals.push({ cfg: c, T: EVAL_T, v, cost });
      if (v > bestV) { bestV = v; best = c; }
      traj.push({ cost, cfg: best });
    }
    return { traj, evals, plan };
  }

  /* ----- Proceso Gaussiano mínimo (RBF ARD, hiperparámetros fijos) + Expected Improvement ----- */
  const LS = [0.28, 0.6, 1.4];                                   // escalas: ψ importa, bootstrap casi no
  function kern(a, b) { let s = 0; for (let i = 0; i < 3; i++) { const d = (a[i] - b[i]) / LS[i]; s += d * d; } return Math.exp(-0.5 * s); }
  function cholesky(K, n) {
    const Lm = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = K[i][j]; for (let k = 0; k < j; k++) s -= Lm[i][k] * Lm[j][k];
      Lm[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / Lm[j][j];
    }
    return Lm;
  }
  function solveL(Lm, b, n) { const y = new Float64Array(n); for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= Lm[i][k] * y[k]; y[i] = s / Lm[i][i]; } return y; }
  function solveLT(Lm, y, n) { const x = new Float64Array(n); for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= Lm[k][i] * x[k]; x[i] = s / Lm[i][i]; } return x; }
  const phi = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
  function Phi(z) { // erf de Abramowitz-Stegun
    const t = 1 / (1 + 0.2316419 * Math.abs(z)), d = 0.3989423 * Math.exp((-z * z) / 2);
    const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return z > 0 ? 1 - p : p;
  }
  function runBayes(L, obj, budget, rng, nInit = 3) {
    const cands = allConfigs(), X = cands.map(encode), evals = [], traj = [], used = new Set(), idxs = [], ys = [];
    let cost = 0, best = null, bestV = -Infinity;
    const evalIdx = (i) => {
      if (cost + EVAL_T > budget) return false;
      const c = cands[i], v = obj(measure(L, c, EVAL_T)); cost += EVAL_T; used.add(i); idxs.push(i); ys.push(v);
      evals.push({ cfg: c, T: EVAL_T, v, cost });
      if (v > bestV) { bestV = v; best = c; }
      traj.push({ cost, cfg: best });
      return true;
    };
    for (const i of shuffle(cands.map((_, k) => k), rng).slice(0, nInit)) if (!evalIdx(i)) break;
    while (cost + EVAL_T <= budget && used.size < cands.length) {
      const n = idxs.length, mu0 = ys.reduce((s, v) => s + v, 0) / n;
      const sd = Math.sqrt(ys.reduce((s, v) => s + (v - mu0) ** 2, 0) / n) || 1;
      const yn = ys.map((v) => (v - mu0) / sd);
      const K = idxs.map((a) => idxs.map((b) => kern(X[a], X[b])));
      for (let i = 0; i < n; i++) K[i][i] += 1e-4;
      const Lm = cholesky(K, n), alpha = solveLT(Lm, solveL(Lm, yn, n), n), ybest = Math.max(...yn);
      let bi = -1, bei = -Infinity;
      for (let c = 0; c < cands.length; c++) {
        if (used.has(c)) continue;
        const ks = idxs.map((a) => kern(X[c], X[a]));
        let mu = 0; for (let i = 0; i < n; i++) mu += ks[i] * alpha[i];
        const v = solveL(Lm, ks, n); let q = 0; for (let i = 0; i < n; i++) q += v[i] * v[i];
        const s = Math.sqrt(Math.max(1 - q, 1e-12)), z = (mu - ybest) / s, ei = (mu - ybest) * Phi(z) + s * phi(z);
        if (ei > bei) { bei = ei; bi = c; }
      }
      if (bi < 0 || !evalIdx(bi)) break;
    }
    return { traj, evals };
  }

  /**
   * Successive Halving con T como recurso (η = 3): evalúa muchas configuraciones con pocos árboles y solo
   * promueve el tercio superior a más árboles. El incumbente sale del rung más alto alcanzado.
   */
  function runHalving(L, obj, budget, rng, eta = 3) {
    const rungs = SPACE.T, cands = shuffle(allConfigs(), rng);
    const perConfig = (n0) => { let n = n0, c = 0; for (const T of rungs) { c += n * T; n = Math.max(1, Math.floor(n / eta)); } return c; };
    let n0 = Math.min(cands.length, 3); while (n0 < cands.length && perConfig(n0 + 1) <= budget) n0++;
    let alive = cands.slice(0, n0), cost = 0;
    const evals = [], traj = [];
    for (let r = 0; r < rungs.length && alive.length; r++) {
      const T = rungs[r], scored = [];
      let inc = null, incV = -Infinity;
      for (const c of alive) {
        if (cost + T > budget) break;
        const v = obj(measure(L, c, T)); cost += T; scored.push({ c, v }); evals.push({ cfg: c, T, v, cost });
        if (v > incV) { incV = v; inc = c; }
        traj.push({ cost, cfg: inc });
      }
      scored.sort((a, b) => b.v - a.v);
      alive = scored.slice(0, Math.max(1, Math.floor(scored.length / eta))).map((s) => s.c);
      if (scored.length < 2) break;
    }
    return { traj, evals, n0 };
  }

  /** Simula R corridas y resume el AP real (a T=300) del incumbente en una malla de costos. */
  function simulate(L, method, objKey, budget, R, rng, grid) {
    const obj = OBJECTIVES[objKey].f, truth = (c) => measure(L, c, FULL_T).ap;
    const curves = [];
    const runs = method === 'grid' ? 1 : R;
    let sample = null;
    for (let r = 0; r < runs; r++) {
      const res = method === 'grid' ? runGrid(L, obj, budget) : method === 'random' ? runRandom(L, obj, budget, rng)
        : method === 'bayes' ? runBayes(L, obj, budget, rng) : runHalving(L, obj, budget, rng);
      if (r === 0) sample = res;
      const vals = grid.map((g) => { let cur = null; for (const p of res.traj) { if (p.cost <= g) cur = p; else break; } return cur ? truth(cur.cfg) : NaN; });
      curves.push(vals);
    }
    const q = (arr, p) => { const a = arr.filter(Number.isFinite).sort((x, y) => x - y); if (!a.length) return NaN; const i = (a.length - 1) * p, lo = Math.floor(i); return a[lo] + (a[Math.min(a.length - 1, lo + 1)] - a[lo]) * (i - lo); };
    const med = [], lo = [], hi = [];
    grid.forEach((_, gi) => { const col = curves.map((c) => c[gi]); med.push(q(col, 0.5)); lo.push(q(col, 0.25)); hi.push(q(col, 0.75)); });
    return { med, lo, hi, sample, finalCost: grid[grid.length - 1] };
  }

  /** Verdad de referencia: mejor configuración (a T=300) y valor por defecto. */
  function reference(L) {
    const cfgs = allConfigs();
    const bi = argmax(cfgs, (c) => measure(L, c, FULL_T).ap);
    return { oracle: cfgs[bi], oracleAP: measure(L, cfgs[bi], FULL_T).ap, defaultAP: measure(L, DEFAULT_CFG, FULL_T).ap };
  }

  return { SPACE, DEFAULT_CFG, EVAL_T, FULL_T, FIELDS, OBJECTIVES, allConfigs, cfgId, measure, key, gridPlan, runGrid, runRandom, runBayes, runHalving, simulate, reference };
});
