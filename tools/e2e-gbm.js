/*
 * Prueba end-to-end del laboratorio de Gradient Boosting.
 *   node tools/serve.js &  ;  NODE_PATH=<ruta a node_modules> node tools/e2e-gbm.js [url]
 */
const puppeteer = require('puppeteer-core');
const assert = require('assert');
const url = process.argv[2] || 'http://localhost:8080/algorithms/gradient-boosting/';
const exe = process.env.BROWSER || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const b = await puppeteer.launch({ executablePath: exe, headless: 'new' });
  const p = await b.newPage(); const errs = [];
  p.on('console', (m) => { if (['error', 'warning'].includes(m.type())) errs.push(m.type() + ': ' + m.text()); });
  p.on('pageerror', (e) => errs.push('PAGEERROR: ' + e.message));
  await p.setViewport({ width: 1440, height: 900 });
  await p.goto(url, { waitUntil: 'networkidle0' });
  await p.addStyleTag({ content: 'html{scroll-behavior:auto!important}' });
  const ok = (m) => console.log('  ✔', m);
  const S = () => p.evaluate(() => {
    const L = window.MLLab.gbLab, m = L.metrics();
    return { T: L.state.nEstimators, trees: L.model.trees.length, train: m.trainLoss, val: m.valLoss, gap: m.gap, auc: m.auc, ap: m.ap, acc: m.valAcc, best: L.bestRound(), depth: L.state.maxDepth, ds: L.state.dataset };
  });
  const settle = async (ms = 12000) => { const t0 = Date.now(); let prev = -1; while (Date.now() - t0 < ms) { await wait(500); const s = await S(); if (s.T === prev) return s; prev = s.T; } return S(); };

  // 1. Entrena al cargar
  let s = await settle();
  assert.ok(s.T >= 40 && s.trees === s.T, `rondas ${s.T} / árboles ${s.trees}`);
  assert.ok(s.train < 0.3 && s.auc > 0.9, `train=${s.train} auc=${s.auc}`);
  ok(`entrena al cargar: ${s.T} rondas, logloss train ${s.train.toFixed(3)} / val ${s.val.toFixed(3)}, AUC ${s.auc.toFixed(3)}`);

  // 2. Todas las gráficas se dibujan
  for (const [sel, what] of [['#curveChart svg', 'curva de pérdida'], ['#impChart svg', 'importancia'], ['#histChart svg', 'histograma'], ['#rocChart svg', 'ROC'], ['#prChart svg', 'PR'], ['#calChart svg', 'calibración'], ['#treeChart svg', 'árbol']]) {
    assert.ok(await p.$(sel), 'falta ' + what);
  }
  const mapPx = await p.evaluate(() => { const c = document.querySelector('#mapCanvas'); return c.width * c.height; });
  assert.ok(mapPx > 100000, 'mapa de decisión vacío: ' + mapPx);
  ok('todas las gráficas dibujadas (mapa de decisión ' + mapPx + ' px)');

  // 3. Añadir rondas baja la pérdida de entrenamiento
  const before = s.train;
  await p.evaluate(() => document.querySelector('.train-btn[data-ep="50"]').click());
  s = await settle(20000);
  assert.ok(s.T >= 90 && s.train <= before + 1e-9, `${before} → ${s.train}`);
  ok(`+50 rondas: ${s.T} totales, logloss de entrenamiento ${s.train.toFixed(3)}`);

  // 4. Sobreajuste con ruido de etiqueta (el escenario guiado)
  await p.evaluate(() => document.querySelector('#lessonChips button[data-id=overfit]').click());
  await wait(600);
  s = await settle(25000);
  assert.ok(s.best < s.T * 0.8, `el mínimo de validación (${s.best}) debería estar antes del final (${s.T})`);
  assert.ok(s.gap > 0.2, 'no hay brecha train-val: ' + s.gap);
  assert.ok(/[Ss]obreajuste/.test(await p.$eval('#reading', (e) => e.textContent)), 'la lectura no avisa del sobreajuste');
  ok(`sobreajuste: mínimo en la ronda ${s.best}/${s.T}, brecha ${s.gap.toFixed(2)}, y la lectura lo avisa`);

  // 5. El tablero (XOR) necesita profundidad ≥ 2
  await p.evaluate(() => document.querySelector('#lessonChips button[data-id=xor-depth]').click());
  await wait(600);
  let sTocon = await settle(20000);
  assert.equal(sTocon.depth, 1);
  assert.ok(sTocon.acc < 0.7, 'el tocón no debería resolver el XOR: ' + sTocon.acc);
  await p.evaluate(() => {
    const r = document.querySelector('.pcard[data-id=maxDepth] input[type=range]');
    r.value = 1; r.dispatchEvent(new Event('input', { bubbles: true }));     // max_depth = 2
  });
  await wait(500);
  await p.evaluate(() => document.querySelector('.train-btn[data-ep="50"]').click());
  const sHondo = await settle(20000);
  assert.equal(sHondo.depth, 2);
  assert.ok(sHondo.acc > sTocon.acc + 0.15, `profundidad 2 debería mejorar: ${sTocon.acc} → ${sHondo.acc}`);
  ok(`tablero XOR: exactitud ${sTocon.acc.toFixed(2)} con tocones → ${sHondo.acc.toFixed(2)} con profundidad 2`);

  // 6. Los 7 escenarios se ejecutan
  for (const id of ['first', 'lr-tradeoff', 'overfit', 'xor-depth', 'gamma-prune', 'imbalanced', 'subsample-var']) {
    await p.evaluate((i) => document.querySelector(`#lessonChips button[data-id=${i}]`).click(), id);
    await wait(500);
    s = await settle(25000);
    console.log(`    escenario ${id}: ${s.ds} · ${s.T} rondas · train ${s.train.toFixed(3)} · val ${s.val.toFixed(3)} · AUC ${s.auc.toFixed(3)}`);
    assert.ok(Number.isFinite(s.val), 'escenario ' + id);
  }
  ok('7 escenarios ejecutados');

  // 7. Barrido
  await p.evaluate(() => document.querySelector('.pcard[data-id=lr] .p-head').click());
  await wait(1000);
  for (let i = 0; i < 60; i++) { await wait(700); if (!/[Cc]alculando/.test(await p.$eval('#sweepChart', (e) => e.textContent))) break; }
  const circles = await p.$$eval('#sweepChart svg circle', (c) => c.length);
  const ins = await p.$eval('#sweepInsight', (e) => e.textContent);
  assert.ok(circles > 4, 'barrido con ' + circles + ' puntos');
  assert.ok(ins.length > 30, 'sin lectura del barrido');
  ok('barrido de learning_rate: ' + circles + ' puntos · ' + ins.slice(0, 70).trim() + '…');

  // 8. Cuadrícula learning_rate × n_estimators
  await p.evaluate(() => document.querySelector('#hmRun').click());
  for (let i = 0; i < 90; i++) { await wait(700); if (!(await p.$eval('#hmRun', (e) => e.disabled))) break; }
  const rects = await p.$$eval('#hmChart svg rect', (r) => r.length);
  const hmRead = await p.$eval('#hmRead', (e) => e.textContent);
  assert.ok(rects >= 30, 'cuadrícula incompleta: ' + rects);
  assert.ok(/learning_rate/.test(hmRead), hmRead);
  ok('cuadrícula 5×6 calculada · ' + hmRead.slice(0, 90).trim() + '…');

  // 9. Estándar "qué no optimizar", casos reales y glosario
  assert.equal(await p.$$eval('[data-not-tune] .nt-card', (c) => c.length), 3);
  await p.evaluate(() => document.querySelector('#casesTabs button[data-id=credito]').click());
  await wait(300);
  const caso = await p.$eval('#caseBody', (e) => e.textContent);
  assert.ok(/[Cc]alibraci/.test(caso), caso.slice(0, 80));
  assert.ok(/Por qué Gradient Boosting/.test(caso), 'la etiqueta del caso no usa el nombre del modelo');
  assert.ok(/boosting/i.test(await p.$eval('[data-term=boosting]', (e) => e.getAttribute('data-tip'))));
  assert.ok(/ds=/.test(await p.evaluate(() => location.hash)));
  ok('bloque "qué no optimizar" (3 tarjetas), casos reales, glosario y configuración en la URL');

  console.log(errs.length ? 'ERRORES DE CONSOLA:\n' + errs.join('\n') : '  ✔ sin errores de consola');
  await b.close();
  process.exit(errs.length ? 1 : 0);
})().catch((e) => { console.error('FALLÓ:', e.message); process.exit(1); });
