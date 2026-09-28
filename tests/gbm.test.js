/* Pruebas del Gradient Boosting: matemática del objetivo regularizado y comportamiento esperado. */
const test = require('node:test');
const assert = require('node:assert');
const GBM = require('../algorithms/gradient-boosting/gbm.js');
const DATA = require('../algorithms/gradient-boosting/data.js');

const fit = (d, opts) => GBM.fit(d.train.X, d.train.y, d.train.n, 2, opts, [{ X: d.val.X, y: d.val.y, n: d.val.n }]);
const last = (m) => m.hist[m.hist.length - 1];

test('sigmoide y logloss: casos de referencia', () => {
  assert.ok(Math.abs(GBM.sigmoid(0) - 0.5) < 1e-12);
  assert.ok(GBM.sigmoid(50) > 0.999 && GBM.sigmoid(-50) < 0.001);
  // predicción constante 0.5 (raw = 0) sobre cualquier etiqueta ⇒ logloss = ln 2
  const F = new Float64Array(4), y = Uint8Array.from([0, 1, 0, 1]);
  assert.ok(Math.abs(GBM.logloss(F, y, 4) - Math.log(2)) < 1e-12);
});

test('el modelo sin rondas predice la proporción base de la clase', () => {
  const d = DATA.generate('blobs', 200, 3, 0, 0.5);
  const m = fit(d, { nEstimators: 0 });
  const p = m.predictProba(d.train.X, d.train.n);
  const base = Array.from(d.train.y).reduce((s, v) => s + v, 0) / d.train.n;
  for (let i = 0; i < d.train.n; i++) assert.ok(Math.abs(p[i] - base) < 1e-9, `p=${p[i]} base=${base}`);
});

test('entrenar baja la pérdida de entrenamiento de forma monótona', () => {
  const d = DATA.generate('moons', 250, 5, 0, 0.5);
  const m = fit(d, { nEstimators: 40, learningRate: 0.2, maxDepth: 3 });
  for (let i = 1; i < m.hist.length; i++) {
    assert.ok(m.hist[i].train <= m.hist[i - 1].train + 1e-9, `ronda ${i}: ${m.hist[i - 1].train} → ${m.hist[i].train}`);
  }
  assert.ok(last(m).train < m.hist[0].train * 0.5);
});

test('determinismo: misma semilla → mismo modelo; otra semilla con subsample → distinto', () => {
  const d = DATA.generate('moons', 200, 7, 0, 0.5);
  const a = fit(d, { nEstimators: 20, subsample: 0.6, seed: 1 });
  const b = fit(d, { nEstimators: 20, subsample: 0.6, seed: 1 });
  const c = fit(d, { nEstimators: 20, subsample: 0.6, seed: 2 });
  assert.equal(last(a).train, last(b).train);
  assert.notEqual(last(a).train, last(c).train);
});

test('los primeros T árboles no cambian al pedir más rondas', () => {
  const d = DATA.generate('blobs', 200, 11, 0, 0.5);
  const corto = fit(d, { nEstimators: 10, subsample: 0.7 });
  const largo = fit(d, { nEstimators: 30, subsample: 0.7 });
  for (let t = 0; t < 10; t++) {
    assert.deepStrictEqual(Array.from(largo.trees[t].thr), Array.from(corto.trees[t].thr), 'árbol ' + t);
    assert.deepStrictEqual(Array.from(largo.trees[t].weight), Array.from(corto.trees[t].weight), 'árbol ' + t);
  }
  assert.ok(Math.abs(largo.hist[9].train - corto.hist[9].train) < 1e-12);
});

test('peso de hoja = −G/(H+λ): lambda encoge las hojas hacia cero', () => {
  const d = DATA.generate('blobs', 200, 13, 0, 0.5);
  const sinReg = fit(d, { nEstimators: 1, maxDepth: 1, lambda: 0, learningRate: 1 });
  const conReg = fit(d, { nEstimators: 1, maxDepth: 1, lambda: 50, learningRate: 1 });
  const maxW = (m) => Math.max(...Array.from(m.trees[0].weight, Math.abs));
  assert.ok(maxW(conReg) < maxW(sinReg), `${maxW(conReg)} debería ser menor que ${maxW(sinReg)}`);
});

test('gamma poda: con un umbral alto ningún corte se acepta y el árbol queda en una hoja', () => {
  const d = DATA.generate('blobs', 200, 17, 0, 0.5);
  const libre = fit(d, { nEstimators: 1, maxDepth: 3, gamma: 0 });
  const podado = fit(d, { nEstimators: 1, maxDepth: 3, gamma: 1e6 });
  assert.ok(libre.trees[0].nNodes > 1, 'sin gamma debería cortar');
  assert.equal(podado.trees[0].nNodes, 1, 'con gamma enorme no debería cortar');
});

test('min_child_weight impide hojas sostenidas por pocos puntos', () => {
  const d = DATA.generate('blobs', 200, 19, 0, 0.5);
  const libre = fit(d, { nEstimators: 1, maxDepth: 4, minChildWeight: 0.01 });
  const exigente = fit(d, { nEstimators: 1, maxDepth: 4, minChildWeight: 1e6 });
  assert.ok(libre.trees[0].nNodes > exigente.trees[0].nNodes);
  assert.equal(exigente.trees[0].nNodes, 1);
});

test('un tocón (profundidad 1) no puede resolver el XOR; profundidad 2 sí', () => {
  const d = DATA.generate('xor', 400, 23, 0, 0.5);
  const tocon = fit(d, { nEstimators: 60, maxDepth: 1, learningRate: 0.3 });
  const hondo = fit(d, { nEstimators: 60, maxDepth: 2, learningRate: 0.3 });
  assert.ok(last(tocon).val0Acc < 0.65, 'el tocón no debería superar el azar por mucho: ' + last(tocon).val0Acc);
  assert.ok(last(hondo).val0Acc > 0.85, 'con profundidad 2 debería resolverlo: ' + last(hondo).val0Acc);
});

test('con ruido de etiqueta, entrenar de más sobreajusta: la validación empeora tras su mínimo', () => {
  const d = DATA.generate('blobs', 200, 29, 0.25, 0.5);
  const m = fit(d, { nEstimators: 150, maxDepth: 6, learningRate: 0.3, lambda: 0.1 });
  let bi = 0;
  m.hist.forEach((h, i) => { if (h.val0 < m.hist[bi].val0) bi = i; });
  assert.ok(bi < m.hist.length - 1, 'el mínimo de validación debería estar antes de la última ronda');
  assert.ok(last(m).val0 > m.hist[bi].val0 * 1.2, 'la validación debería empeorar claramente tras el mínimo');
  assert.ok(last(m).train < m.hist[bi].train, 'mientras el entrenamiento sigue bajando');
});

test('learning_rate bajo necesita más rondas para el mismo ajuste', () => {
  const d = DATA.generate('moons', 300, 31, 0, 0.5);
  const rapido = fit(d, { nEstimators: 10, learningRate: 0.5, maxDepth: 3 });
  const lento10 = fit(d, { nEstimators: 10, learningRate: 0.05, maxDepth: 3 });
  const lento100 = fit(d, { nEstimators: 100, learningRate: 0.05, maxDepth: 3 });
  assert.ok(last(lento10).train > last(rapido).train, 'con 10 rondas, η bajo va por detrás');
  assert.ok(last(lento100).train < last(lento10).train, 'con 100 rondas alcanza el ajuste');
});

test('stagedProba coincide con entrenar hasta esa ronda', () => {
  const d = DATA.generate('moons', 200, 37, 0, 0.5);
  const m = fit(d, { nEstimators: 20, learningRate: 0.3, maxDepth: 2 });
  const staged = m.stagedProba(d.val.X, d.val.n, [5, 20]);
  const corto = fit(d, { nEstimators: 5, learningRate: 0.3, maxDepth: 2 });
  const p5 = corto.predictProba(d.val.X, d.val.n);
  for (let i = 0; i < d.val.n; i++) assert.ok(Math.abs(staged[0][i] - p5[i]) < 1e-12);
  const p20 = m.predictProba(d.val.X, d.val.n);
  for (let i = 0; i < d.val.n; i++) assert.ok(Math.abs(staged[1][i] - p20[i]) < 1e-12);
});

test('datos: coordenadas en [0,1], validación independiente y ruido de etiqueta aplicado', () => {
  const d = DATA.generate('circles', 300, 41, 0.2, 0.5);
  assert.ok(d.train.n > 0 && d.val.n > 0);
  for (let i = 0; i < d.train.n * 2; i++) assert.ok(d.train.X[i] >= 0 && d.train.X[i] <= 1);
  assert.ok(d.train.flipped > 0, 'debería haber etiquetas cambiadas');
  const frac = d.train.flipped / d.train.n;
  assert.ok(frac > 0.1 && frac < 0.32, 'la fracción cambiada debería rondar el 20%: ' + frac);
  assert.notDeepStrictEqual(Array.from(d.val.X.slice(0, 10)), Array.from(d.train.X.slice(0, 10)));
});

test('clase rara: el generador respeta la proporción pedida', () => {
  const d = DATA.generate('blobs', 600, 43, 0, 0.06);
  const frac = Array.from(d.train.y).reduce((s, v) => s + v, 0) / d.train.n;
  assert.ok(frac > 0.03 && frac < 0.10, 'proporción de clase 1: ' + frac);
});

test('importancia por ganancia: en el XOR ambas variables participan', () => {
  const d = DATA.generate('xor', 400, 47, 0, 0.5);
  const m = fit(d, { nEstimators: 40, maxDepth: 2, learningRate: 0.3 });
  const [g0, g1] = Array.from(m.gainImportance);
  assert.ok(g0 > 0 && g1 > 0, `ambas variables deberían aportar: ${g0}, ${g1}`);
  assert.ok(Math.min(g0, g1) / Math.max(g0, g1) > 0.2, 'y de forma comparable');
});
