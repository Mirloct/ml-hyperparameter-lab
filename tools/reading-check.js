/*
 * Verifica la secuencia de lectura del curso de Isolation Forest.
 *   1. Orden de capítulos: navegación = orden del documento; cada capítulo tiene objetivos, resumen y enlace "siguiente".
 *   2. Definición antes de uso: la primera aparición de cada concepto clave debe estar "presentada"
 *      (término del glosario con tooltip, encabezado, celda con definición, tile con tooltip, etc.).
 *   3. Integridad: citas → bibliografía, términos → glosario, jerarquía de encabezados sin saltos.
 *   4. Seguimiento: al hacer scroll a cada capítulo, la navegación lo marca como activo.
 * Requiere puppeteer-core y un navegador Chromium/Edge:
 *   node tools/serve.js &   ;   NODE_PATH=<ruta a node_modules> node tools/reading-check.js [url]
 */
const puppeteer = require('puppeteer-core');
const url = process.argv[2] || 'http://localhost:8080/algorithms/isolation-forest/';
const exe = process.env.BROWSER || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';

// Los conceptos los declara cada laboratorio en ML.READING_CONCEPTS = [nombre, subcadena, capítulo].
const fail = [], warn = [], ok = [];
const NO_CONCEPTS = 'la página no declara ML.READING_CONCEPTS: no se comprueba "definición antes de uso"';
const rec = (arr, m) => arr.push(m);

(async () => {
  const b = await puppeteer.launch({ executablePath: exe, headless: 'new' });
  const p = await b.newPage();
  await p.setViewport({ width: 1440, height: 900 });
  await p.goto(url, { waitUntil: 'networkidle0' });
  await p.addStyleTag({ content: 'html{scroll-behavior:auto!important}' });
  await new Promise((r) => setTimeout(r, 4500));

  /* 1. Orden de capítulos */
  const struct = await p.evaluate(() => {
    const secs = [...document.querySelectorAll('section.chapter')].map((s) => ({ id: s.id, top: s.getBoundingClientRect().top + scrollY, goals: !!s.querySelector('.ch-goals'), take: !!s.querySelector('.takeaways li'), checks: !!s.querySelector('.checks details'), next: !!s.querySelector('.ch-next a'), h2: (s.querySelector('h2') || {}).textContent }));
    const nav = [...document.querySelectorAll('#chapList a')].map((a) => a.dataset.ch);
    return { secs, nav };
  });
  const domOrder = struct.secs.map((s) => s.id);
  JSON.stringify(domOrder) === JSON.stringify(struct.nav) ? rec(ok, `Orden de capítulos: navegación = documento (${domOrder.join(' → ')})`) : rec(fail, `La navegación ${struct.nav} no coincide con el documento ${domOrder}`);
  struct.secs.forEach((s) => {
    if (s.id === 'refs') return;
    if (!s.goals) rec(fail, `${s.id}: falta el cuadro de objetivos de aprendizaje`);
    if (s.id !== 'ch6' && !s.take) rec(fail, `${s.id}: falta "En resumen"`);
    if (!s.checks) rec(fail, `${s.id}: faltan preguntas de comprobación`);
    if (!s.next) rec(fail, `${s.id}: falta enlace a siguiente/anterior`);
  });

  /* 2. Definición antes de uso */
  const CONCEPTS = await p.evaluate(() => window.MLLab.READING_CONCEPTS || []);
  if (!CONCEPTS.length) rec(warn, NO_CONCEPTS);
  const report = await p.evaluate((CONCEPTS0) => {
    // Coincidencia por palabra: la raíz declarada admite sufijos de flexión (plural, género),
    // salvo en siglas cortas como «AP», donde el límite es estricto para no casar con «aprende».
    const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const W = '[\\p{L}\\p{N}_]';
    const CONCEPTS = CONCEPTS0.map(([name, sub, byCh]) => {
      const s = String(sub), tail = s.length <= 3 ? '(?!' + W + ')' : W + '{0,3}(?!' + W + ')';
      return [name, new RegExp('(^|(?!' + W + ').)' + esc(s) + tail, 'iu'), byCh];
    });
    const main = document.querySelector('main');
    const walker = document.createTreeWalker(main, NodeFilter.SHOW_TEXT);
    const nodes = []; let n;
    while ((n = walker.nextNode())) { if (n.nodeValue.trim().length) nodes.push(n); }
    const chOf = (el) => { const s = el.closest('section.chapter, section.hero'); return s ? s.id : ''; };
    const introduced = (el) => !!el.closest('[data-term],[data-tip],h1,h2,h3,h4,th,summary,.route,.tg-title,.tile,.ind-table td:first-child,.methods h4,.glossary,.cite,.refs-list,.lesson-chips,.case-tabs,.p-head,.ds-list,.ch-goals li,.effects b,.takeaways h4,code');
    const res = [];
    for (const [name, rx, byCh] of CONCEPTS) {
      const defined = new Set();
      let first = null, firstIntro = null;
      for (const t of nodes) {
        if (!rx.test(' ' + t.nodeValue)) continue;
        const el = t.parentElement;
        if (el.closest('.refs-list, svg, [aria-hidden="true"], .chapnav, .route')) continue;   // bibliografía, ilustraciones y navegación no cuentan
        const intro = introduced(el);
        if (!first) first = { ch: chOf(el), intro, text: t.nodeValue.trim().slice(0, 70), tag: el.tagName };
        if (intro && !firstIntro) firstIntro = { ch: chOf(el) };
      }
      res.push({ name, byCh, first, firstIntro });
    }
    return res;
  }, CONCEPTS);
  const chIdx = (id) => ['top', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'refs'].indexOf(id);
  report.forEach((r) => {
    if (!r.first) { rec(warn, `«${r.name}»: no aparece en el texto`); return; }
    if (!r.first.intro) rec(fail, `«${r.name}»: primera aparición SIN presentar en ${r.first.ch} → "${r.first.text}" (<${r.first.tag}>) — añade data-term o definición`);
    else if (chIdx(r.first.ch) > chIdx(r.byCh)) rec(warn, `«${r.name}»: se presenta en ${r.first.ch}, más tarde de lo esperado (${r.byCh})`);
    else rec(ok, `«${r.name}»: presentado en ${r.first.ch}`);
  });

  /* 3. Integridad */
  const integ = await p.evaluate(() => {
    const ids = new Set([...document.querySelectorAll('[id]')].map((e) => e.id));
    const badCites = [...document.querySelectorAll('a.cite')].map((a) => a.getAttribute('href').slice(1)).filter((id) => !ids.has(id));
    const unusedRefs = [...document.querySelectorAll('.refs-list li')].map((li) => li.id).filter((id) => !document.querySelector(`a.cite[href="#${id}"]`));
    const badTerms = [...document.querySelectorAll('[data-term]')].filter((e) => !e.dataset.tip && !e.getAttribute('data-tip')).map((e) => e.dataset.term);
    const heads = [...document.querySelectorAll('main h1, main h2, main h3, main h4')].map((h) => +h.tagName[1]);
    let jumps = 0; const at = []; const hs = [...document.querySelectorAll('main h1, main h2, main h3, main h4')]; for (let i = 1; i < heads.length; i++) if (heads[i] - heads[i - 1] > 1) { jumps++; at.push(hs[i - 1].textContent.trim().slice(0, 30) + ' → ' + hs[i].textContent.trim().slice(0, 30)); }
    const emptyCharts = [...document.querySelectorAll('.chart')].filter((c) => !c.innerHTML.trim() && !c.closest('[hidden]')).map((c) => c.id);
    return { badCites, unusedRefs, badTerms, jumps, at, emptyCharts, nCites: document.querySelectorAll('a.cite').length, nRefs: document.querySelectorAll('.refs-list li').length };
  });
  integ.badCites.length ? rec(fail, `Citas sin referencia: ${integ.badCites}`) : rec(ok, `Las ${integ.nCites} citas apuntan a la bibliografía`);
  integ.unusedRefs.length ? rec(warn, `Referencias sin citar en el texto: ${integ.unusedRefs.join(', ')}`) : rec(ok, `Las ${integ.nRefs} referencias están citadas`);
  integ.badTerms.length ? rec(fail, `Términos sin glosario: ${integ.badTerms}`) : rec(ok, 'Todos los términos del glosario tienen definición');
  integ.jumps ? rec(warn, `${integ.jumps} saltos de nivel de encabezado: ${integ.at.join(' | ')}`) : rec(ok, 'Jerarquía de encabezados sin saltos');
  integ.emptyCharts.length ? rec(fail, `Gráficas vacías: ${integ.emptyCharts}`) : rec(ok, 'Todas las gráficas se dibujaron');

  /* 4. Seguimiento de lectura */
  for (const s of struct.secs) {
    await p.evaluate((y) => window.scrollTo(0, y), s.top + 120);
    await new Promise((r) => setTimeout(r, 350));
    const on = await p.evaluate(() => { const a = document.querySelector('#chapList a.on'); return a ? a.dataset.ch : null; });
    on === s.id ? rec(ok, `Al leer ${s.id} la navegación lo marca activo`) : rec(fail, `Al leer ${s.id} la navegación marca «${on}»`);
  }

  console.log('\n✔ CORRECTO'); ok.forEach((m) => console.log('  ✔', m));
  if (warn.length) { console.log('\n⚠ AVISOS'); warn.forEach((m) => console.log('  ⚠', m)); }
  if (fail.length) { console.log('\n✖ PROBLEMAS'); fail.forEach((m) => console.log('  ✖', m)); }
  console.log(`\nResumen: ${ok.length} correctos · ${warn.length} avisos · ${fail.length} problemas`);
  await b.close();
  process.exit(fail.length ? 1 : 0);
})();
