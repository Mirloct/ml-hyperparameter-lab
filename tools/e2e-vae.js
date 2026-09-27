/*
 * Prueba end-to-end del laboratorio de VAE.
 *   node tools/serve.js &  ;  NODE_PATH=<ruta a node_modules> node tools/e2e-vae.js [url]
 */
const puppeteer = require('puppeteer-core');
const assert = require('assert');
const url = process.argv[2] || 'http://localhost:8080/algorithms/vae/';
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
  const S = () => p.evaluate(() => { const L = window.MLLab.vaeLab, e = L.ev; return { ep: L.model.epoch, rec: e.rec, kl: e.kl, active: e.active, L: L.state.latent, beta: L.state.beta, n: L.data.n }; });
  const settle = async (ms = 9000) => { const t0 = Date.now(); let prev = -1; while (Date.now() - t0 < ms) { await wait(600); const s = await S(); if (s.ep === prev) return s; prev = s.ep; } return S(); };

  // 1. Entrenamiento automático inicial
  let s = await settle();
  assert.ok(s.ep >= 40, 'épocas iniciales ' + s.ep);
  assert.ok(s.rec < 20 && s.kl > 0.3, `rec=${s.rec} kl=${s.kl}`);
  ok(`entrena al cargar: ${s.ep} épocas, reconstrucción ${s.rec.toFixed(2)}, KL ${s.kl.toFixed(2)}, ${s.active}/${s.L} activas`);

  // 2. Las gráficas se dibujan
  for (const [sel, what] of [['#curveChart svg', 'curva'], ['#klChart svg', 'KL por dimensión'], ['#latentChart svg', 'espacio latente'], ['#factTable table', 'tabla de factores']]) {
    assert.ok(await p.$(sel), 'falta ' + what);
  }
  const genPx = await p.evaluate(() => { const c = document.querySelector('#genGrid'); return c.width * c.height; });
  assert.ok(genPx > 10000, 'mapa generativo vacío');
  ok('todas las gráficas dibujadas (mapa generativo ' + genPx + ' px)');

  // 3. Entrenar más mejora
  const before = s.rec;
  await p.evaluate(() => document.querySelector('[data-ep="50"]').click());
  s = await settle(15000);
  assert.ok(s.ep >= 90 && s.rec <= before, `${before} → ${s.rec}`);
  ok(`+50 épocas: ${s.ep} totales, reconstrucción ${s.rec.toFixed(2)}`);

  // 4. Colapso posterior con β alto
  await p.evaluate(() => {
    const r = document.querySelector('.pcard[data-id=beta] input[type=range]');
    r.value = 9; r.dispatchEvent(new Event('input', { bubbles: true }));       // β = 8
  });
  await wait(400);
  await p.evaluate(() => document.querySelector('[data-ep="100"]').click());
  s = await settle(25000);
  assert.ok(s.beta >= 6, 'beta=' + s.beta);
  assert.ok(s.kl < 0.1 && s.active === 0, `no colapsó: kl=${s.kl} activas=${s.active}`);
  assert.ok(/[Cc]olapso/.test(await p.$eval('#reading', (e) => e.textContent)), 'la lectura no avisa del colapso');
  ok(`β=${s.beta} → colapso posterior (KL ${s.kl.toFixed(3)}, ${s.active} activas) y la lectura lo avisa`);

  // 5. Escenarios
  for (const id of ['first', 'beta-collapse', 'rescue', 'beta0', 'toosmall', 'lr', 'anom']) {
    await p.evaluate((i) => document.querySelector(`#lessonChips button[data-id=${i}]`).click(), id);
    await wait(700);
    s = await settle(25000);
    console.log(`    escenario ${id}: L=${s.L} β=${s.beta} · ${s.ep} ép · rec ${s.rec.toFixed(2)} · KL ${s.kl.toFixed(2)} · ${s.active} activas`);
    assert.ok(Number.isFinite(s.rec), 'escenario ' + id);
  }
  ok('7 escenarios ejecutados');

  // 6. El escenario de anomalías muestra su tarjeta y métricas
  assert.ok(!(await p.$eval('#card-anom', (e) => e.hidden)), 'la tarjeta de anomalías sigue oculta');
  const anomTxt = await p.$eval('#anomRead', (e) => e.textContent);
  assert.ok(/ROC-AUC/.test(anomTxt), anomTxt);
  ok('anomalías: ' + anomTxt.slice(0, 80).trim());

  // 7. Barrido
  await p.evaluate(() => document.querySelector('.pcard[data-id=beta] .p-head').click());
  await wait(1000);
  for (let i = 0; i < 40; i++) { await wait(700); if (!/calculando/.test(await p.$eval('#sweepChart', (e) => e.textContent))) break; }
  const circles = await p.$$eval('#sweepChart svg circle', (c) => c.length);
  const ins = await p.$eval('#sweepInsight', (e) => e.textContent);
  assert.ok(circles > 4, 'barrido con ' + circles + ' puntos');
  assert.ok(ins.length > 30, 'sin lectura del barrido');
  ok('barrido de β: ' + circles + ' puntos · ' + ins.slice(0, 70).trim() + '…');

  // 8. Cuadrícula β × latent_dim
  await p.evaluate(() => { document.querySelector('#hmEp').value = '25'; document.querySelector('#hmEp').dispatchEvent(new Event('change')); });
  await p.evaluate(() => document.querySelector('#hmRun').click());
  for (let i = 0; i < 90; i++) { await wait(1000); if (!(await p.$eval('#hmRun', (e) => e.disabled))) break; }
  const rects = await p.$$eval('#hmChart svg rect', (r) => r.length);
  const hmRead = await p.$eval('#hmRead', (e) => e.textContent);
  assert.ok(rects >= 30, 'cuadrícula incompleta: ' + rects);
  assert.ok(/interacci/i.test(hmRead), hmRead);
  ok('cuadrícula 5×6 entrenada · ' + hmRead.slice(0, 90).trim() + '…');

  // 9. Estándar "qué no optimizar" y casos
  assert.equal(await p.$$eval('[data-not-tune] .nt-card', (c) => c.length), 4);
  await p.evaluate(() => document.querySelector('#casesTabs button[data-id=datos]').click());
  await wait(300);
  assert.ok(/borros/.test(await p.$eval('#caseBody', (e) => e.textContent)));
  ok('bloque "qué no optimizar" (4 tarjetas) y casos reales');

  // 10. Glosario y URL
  assert.ok(/colapso/i.test(await p.$eval('[data-term=colapso]', (e) => e.getAttribute('data-tip'))));
  assert.ok(/ds=/.test(await p.evaluate(() => location.hash)));
  ok('glosario y configuración en la URL');

  console.log(errs.length ? 'ERRORES DE CONSOLA:\n' + errs.join('\n') : '  ✔ sin errores de consola');
  await b.close();
  process.exit(errs.length ? 1 : 0);
})().catch((e) => { console.error('FALLÓ:', e.message); process.exit(1); });
