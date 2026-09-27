/*
 * Métricas de evaluación para detección de anomalías (score alto = más anómalo, y=1 anómalo).
 * Todo se calcula desde cero para que el código sea legible en clase.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.MLLab = root.MLLab || {}; root.MLLab.metrics = factory(); }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  function counts(y) {
    let pos = 0;
    for (let i = 0; i < y.length; i++) pos += y[i] ? 1 : 0;
    return { pos, neg: y.length - pos };
  }

  /** Curvas ROC y PR en un solo barrido (los empates de score se agrupan). */
  function curves(scores, y) {
    const n = scores.length;
    const { pos, neg } = counts(y);
    if (!pos || !neg) return null;
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);

    const fpr = [0], tpr = [0], thrRoc = [Infinity];
    const recall = [0], precision = [1], thrPr = [Infinity];
    let tp = 0, fp = 0, ap = 0, prevRecall = 0;
    for (let k = 0; k < n; k++) {
      const i = order[k];
      if (y[i]) tp++; else fp++;
      const last = k === n - 1 || scores[order[k + 1]] !== scores[i];
      if (!last) continue;
      const r = tp / pos, p = tp / (tp + fp);
      fpr.push(fp / neg); tpr.push(r); thrRoc.push(scores[i]);
      recall.push(r); precision.push(p); thrPr.push(scores[i]);
      ap += (r - prevRecall) * p;       // definición de sklearn: Σ (R_n − R_{n−1}) · P_n
      prevRecall = r;
    }
    let auc = 0;
    for (let k = 1; k < fpr.length; k++) auc += ((fpr[k] - fpr[k - 1]) * (tpr[k] + tpr[k - 1])) / 2;
    return { fpr, tpr, thrRoc, recall, precision, thrPr, auc, ap, pos, neg };
  }

  /** Matriz de confusión y derivados para un umbral (predice anómalo si score > thr). */
  function confusion(scores, y, thr) {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    for (let i = 0; i < scores.length; i++) {
      const pred = scores[i] > thr;
      if (pred && y[i]) tp++;
      else if (pred) fp++;
      else if (y[i]) fn++;
      else tn++;
    }
    const precision = tp + fp ? tp / (tp + fp) : 0;
    const recall = tp + fn ? tp / (tp + fn) : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    return { tp, fp, fn, tn, precision, recall, f1, flagged: tp + fp };
  }

  /** ROC-AUC vía estadístico de rangos (Mann–Whitney), rápido para los barridos. */
  function rocAuc(scores, y) {
    const n = scores.length;
    const { pos, neg } = counts(y);
    if (!pos || !neg) return NaN;
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[a] - scores[b]);
    let rankSum = 0, i = 0;
    while (i < n) {
      let j = i;
      while (j + 1 < n && scores[order[j + 1]] === scores[order[i]]) j++;
      const avgRank = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) if (y[order[k]]) rankSum += avgRank;
      i = j + 1;
    }
    return (rankSum - (pos * (pos + 1)) / 2) / (pos * neg);
  }

  function averagePrecision(scores, y) {
    const c = curves(scores, y);
    return c ? c.ap : NaN;
  }

  return { curves, confusion, rocAuc, averagePrecision, counts };
});
