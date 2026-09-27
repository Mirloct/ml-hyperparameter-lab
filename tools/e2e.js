/*
 * Prueba de humo end-to-end (puppeteer-core): ejercita controles, escenarios, carrera de aislamiento,
 * indicadores sin etiquetas, arena de optimización y casos, y falla si hay errores de consola.
 *   node tools/serve.js &   ;   NODE_PATH=<ruta a node_modules> node tools/e2e.js [url]
 */
const puppeteer = require('puppeteer-core');
const assert = require('assert');
const url = process.argv[2] || 'http://localhost:8080/algorithms/isolation-forest/';
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
  await wait(3500);
  const M = () => p.evaluate(() => { const L = window.MLLab.lab, m = L.model, u = L.state.unsup; return { auc: m.cv.auc, ap: m.cv.ap, f1: m.conf.f1, flagged: m.conf.flagged, T: m.f.nEstimators, psi: m.f.psi, jac: u && u.jac, mv: u && u.mv, patk: m.patk }; });
  const ok = (m) => console.log('  ✔', m);

  // 1. Modelo y contamination
  let m0 = await M(); assert(m0.auc > 0.9); ok(`modelo base AUC=${m0.auc.toFixed(3)} AP=${m0.ap.toFixed(3)}`);
  await p.evaluate(() => document.querySelector('.pcard[data-id=contamination] [data-auto]').click()); await wait(700);
  const m1 = await M(); assert(Math.abs(m1.auc - m0.auc) < 1e-12 && Math.abs(m1.ap - m0.ap) < 1e-12); ok('contamination no altera AUC/AP');
  // 2. Indicadores sin etiquetas calculados
  await wait(1500);
  const m2 = await M(); assert(Number.isFinite(m2.jac) && Number.isFinite(m2.mv) && m2.jac > 0 && m2.mv > 0 && m2.mv < 1); ok(`sin etiquetas: Jaccard=${m2.jac.toFixed(2)} MV=${m2.mv.toFixed(2)} P@k=${m2.patk.toFixed(2)}`);
  const tiles = await p.$$eval('#tilesUnsup .t-val', (e) => e.map((x) => x.textContent)); assert(tiles.every((t) => /^\d/.test(t)), 'tiles: ' + tiles); ok('tiles sin etiquetas: ' + tiles.join(' '));
  assert((await p.$$eval('#jacChart svg rect', (r) => r.length)) > 20); ok('matriz de Jaccard dibujada');
  // 3. Controles
  await p.$eval('.pcard[data-id=nEstimators] input[type=range]', (e) => { e.value = 0; e.dispatchEvent(new Event('input', { bubbles: true })); }); await wait(900);
  assert.strictEqual((await M()).T, 1); ok('n_estimators=1');
  await p.evaluate(() => { document.querySelector('.pcard[data-id=maxFeatures] [data-v="0.5"]').click(); document.querySelector('.pcard[data-id=bootstrap] [data-boot]').click(); document.querySelector('.pcard[data-id=seed] [data-dice]').click(); }); await wait(800);
  ok('max_features, bootstrap y semilla');
  // 4. Barridos + insight
  for (const id of ['nEstimators', 'maxSamples', 'contamination', 'maxFeatures', 'bootstrap', 'seed']) {
    await p.evaluate((i) => document.querySelector(`.pcard[data-id=${i}] .p-head`).click(), id); await wait(3800);
    const circles = await p.$$eval('#sweepChart svg circle', (c) => c.length), run = await p.$eval('#sweepChart', (e) => /calculando/.test(e.textContent)), ins = await p.$eval('#sweepInsight', (e) => e.textContent.length);
    assert(circles > 5 && !run && ins > 20, `sweep ${id}: ${circles} ${run} ${ins}`);
  }
  ok('6 barridos con lectura automática');
  // 5. Escenarios (incluye fraude con n=1000, 1 %)
  for (const id of ['trees', 'contam', 'masking', 'ring', 'seed', 'fraud', 'tree']) {
    await p.evaluate((i) => document.querySelector(`#lessonChips button[data-id=${i}]`).click(), id); await wait(3800);
    const mm = await M(); assert(Number.isFinite(mm.auc)); console.log(`    escenario ${id}: AUC=${mm.auc.toFixed(3)} AP=${mm.ap.toFixed(3)} Jaccard=${mm.jac && mm.jac.toFixed(2)}`);
    if (id === 'fraud') { assert.strictEqual(await p.evaluate(() => window.MLLab.lab.data.n), 1000); ok('escenario fraude: 1000 puntos, AUC ' + mm.auc.toFixed(3) + ' vs AP ' + mm.ap.toFixed(3)); }
  }
  // 6. Carrera de aislamiento
  await p.evaluate(() => document.querySelector('#race').scrollIntoView());
  await p.evaluate(() => document.querySelector('#raceTarget [data-t=a]').click());
  for (let i = 0; i < 6; i++) await p.evaluate(() => document.querySelector('#raceCut').click());
  assert(+(await p.$eval('#raceCuts', (e) => e.textContent)) >= 1); ok('carrera: cortes');
  await p.evaluate(() => document.querySelector('#raceMany').click()); await wait(600);
  const means = await p.$eval('#raceMeans', (e) => e.textContent); assert(/anómalo/.test(means)); ok('carrera ×300: ' + means.slice(0, 60));
  // 7. Arena de optimización
  await wait(1500);
  const rows = await p.$$eval('#taTable tbody tr', (r) => r.length); assert(rows >= 5); ok('arena: tabla con ' + rows + ' filas');
  await p.evaluate(() => document.querySelector('#taObj [data-o=mv]').click()); await wait(2500);
  assert(/Ojo: objetivo sin etiquetas/i.test(await p.$eval('#taRead', (e) => e.textContent))); ok('arena: objetivo sin etiquetas advierte');
  await p.evaluate(() => { const s = document.querySelector('#pxDs'); s.value = 'masking'; s.dispatchEvent(new Event('change')); }); await wait(600);
  assert(/eligen? mal/.test(await p.$eval('#pxRead', (e) => e.textContent))); ok('proxies: fallo detectado en masking');
  // 8. Casos reales → laboratorio
  await p.evaluate(() => document.querySelector('#casesTabs button[data-id=ciber]').click());
  await p.evaluate(() => document.querySelector('#caseBody [data-sim]').click()); await wait(3500);
  assert.strictEqual(await p.evaluate(() => window.MLLab.lab.state.dataset), 'masking'); ok('caso → escenario masking');
  // 9. Términos y preguntas
  const tip = await p.$eval('[data-term=jaccard]', (e) => e.getAttribute('data-tip')); assert(/Jaccard/.test(tip)); ok('términos del glosario con tooltip');
  await p.evaluate(() => document.querySelector('.checks details summary').click());
  assert(await p.evaluate(() => document.querySelector('.checks details').open)); ok('preguntas de comprobación se despliegan');
  // 10. Hash compartible
  assert(/ds=masking/.test(await p.evaluate(() => location.hash))); ok('configuración en la URL');

  console.log(errs.length ? 'ERRORES DE CONSOLA:\n' + errs.join('\n') : '  ✔ sin errores de consola');
  await b.close();
  process.exit(errs.length ? 1 : 0);
})().catch((e) => { console.error('FALLÓ:', e.message); process.exit(1); });
