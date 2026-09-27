/* Estructura de lectura COMPARTIDA por todos los laboratorios: navegación por capítulos, términos del
 * glosario, resúmenes, preguntas de comprobación y casos reales.
 * Cada laboratorio define ML.GLOSSARY, ML.CHECKS, ML.TAKEAWAYS y ML.CASES (o las variantes ML.IF_*). */
(function () {
  'use strict';
  const ML = window.MLLab;
  const { $, $$, esc } = ML;

  /* ---------- Términos del glosario: subrayado punteado + definición al señalarlos ---------- */
  function terms() {
    $$('[data-term]').forEach((el) => {
      const g = ML.GLOSSARY[el.dataset.term];
      if (!g) { console.warn('Término sin definición:', el.dataset.term); return; }
      el.classList.add('term');
      el.setAttribute('data-tip', `<span class="tip-h">${g[0]}</span>${g[1]}<span class="tip-sub">Definición del glosario</span>`);
      el.tabIndex = 0;
    });
  }

  /* ---------- Preguntas de comprobación ---------- */
  function checks() {
    $$('[data-checks]').forEach((host) => {
      const list = (ML.CHECKS || ML.IF_CHECKS || {})[host.dataset.checks] || [];
      if (!list.length) { host.remove(); return; }
      host.innerHTML = `<h3><span class="chk-i" aria-hidden="true">?</span> Comprueba lo aprendido</h3>` +
        list.map((c) => `<details class="check"><summary>${c.q}</summary><div class="answer">${c.a}</div></details>`).join('');
    });
  }

  /* ---------- Resumen de capítulo ---------- */
  function takeaways() {
    $$('[data-takeaways]').forEach((host) => {
      const list = (ML.TAKEAWAYS || ML.IF_TAKEAWAYS || {})[host.dataset.takeaways] || [];
      host.innerHTML = `<h3>En resumen</h3><ul>${list.map((t) => `<li>${t}</li>`).join('')}</ul>`;
    });
  }

  /* ---------- Casos reales ---------- */
  function cases() {
    const tabs = $('#casesTabs'), body = $('#caseBody');
    if (!tabs || !(ML.CASES || ML.IF_CASES)) return;
    tabs.innerHTML = (ML.CASES || ML.IF_CASES).map((c, i) => `<button type="button" role="tab" data-id="${c.id}" aria-selected="${i === 0}"><small>${c.kicker}</small>${c.name}</button>`).join('');
    const show = (id) => {
      const c = (ML.CASES || ML.IF_CASES).find((x) => x.id === id);
      $$('button', tabs).forEach((b) => b.setAttribute('aria-selected', String(b.dataset.id === id)));
      body.innerHTML = `<header><span class="chip">${c.kicker}</span><h3>${c.name}</h3></header>
        <div class="case-grid">
          <div><h5>Contexto</h5><p>${c.ctx}</p><h5>Qué buscamos</h5><p>${c.goal}</p><h5>Por qué Isolation Forest</h5><p>${c.why}</p></div>
          <div><h5>Hiperparámetros y decisiones</h5><ul class="keys">${c.keys.map(([k, t]) => `<li><code>${k}</code><span>${t}</span></li>`).join('')}</ul>
            <h5>Qué medir</h5><p>${c.measure}</p><h5>Trampa típica</h5><p class="trap">${c.trap}</p></div>
        </div>
        <footer><button class="btn primary" type="button" data-sim="${c.sim.lesson}">▶ ${c.sim.label} en el laboratorio</button><span class="fine">Carga un escenario del capítulo 2 con datos sintéticos análogos.</span></footer>`;
      $('[data-sim]', body).addEventListener('click', (e) => ML.applyLesson(e.currentTarget.dataset.sim));
    };
    $$('button', tabs).forEach((b) => b.addEventListener('click', () => show(b.dataset.id)));
    show((ML.CASES || ML.IF_CASES)[0].id);
  }

  /* ---------- Navegación por capítulos: capítulo activo y progreso de lectura ---------- */
  function nav() {
    const links = $$('#chapList a'), secs = $$('.chapter'), bar = $('#progBar');
    const setActive = (id) => links.forEach((a) => { const on = a.dataset.ch === id; a.classList.toggle('on', on); if (on) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current'); if (on) { const li = a.parentElement, ol = li.parentElement; if (ol.scrollWidth > ol.clientWidth) ol.scrollTo({ left: li.offsetLeft - 24, behavior: 'smooth' }); } });
    const io = new IntersectionObserver((es) => { es.forEach((e) => { if (e.isIntersecting) setActive(e.target.id); }); }, { rootMargin: '-30% 0px -60% 0px' });
    secs.forEach((s) => io.observe(s));
    const upd = () => { const h = document.documentElement, max = h.scrollHeight - innerHeight; bar.style.width = (max > 0 ? Math.min(100, (scrollY / max) * 100) : 0) + '%'; if (scrollY < 200) setActive(''); };
    addEventListener('scroll', upd, { passive: true }); upd();
  }

  /* ---------- Tabla de tunabilidad (capítulo 4) ---------- */
  function tunability() {  // solo existe en el laboratorio de Isolation Forest
    const host = $('#tuneTable'), LS = ML.landscapes, TU = ML.tuning;
    if (!host || !LS || !TU) return;
    const rows = Object.keys(LS).map((id) => { const r = TU.reference(LS[id]); return { id, def: r.defaultAP, best: r.oracleAP, gain: r.oracleAP - r.defaultAP, cfg: r.oracle }; }).sort((a, b) => b.gain - a.gain);
    const mx = Math.max(...rows.map((r) => r.gain), 0.05);
    host.innerHTML = `<table class="ta-table"><thead><tr><th>Datos</th><th>AP por defecto</th><th>AP mejor</th><th>Ganancia por tunear</th><th>Mejor configuración</th></tr></thead><tbody>` +
      rows.map((r) => `<tr><td>${esc(ML.datasets.DEFS[r.id].name)}</td><td>${r.def.toFixed(3)}</td><td>${r.best.toFixed(3)}</td>
        <td><div class="gainbar"><i style="width:${Math.max(2, (r.gain / mx) * 100).toFixed(0)}%"></i><b>+${r.gain.toFixed(3)}</b></div></td>
        <td>ψ=${r.cfg.psi} · max_features=${r.cfg.mf} · bootstrap=${r.cfg.bs ? 'True' : 'False'}</td></tr>`).join('') + `</tbody></table>
      <p class="fine">Donde la ganancia es casi cero, los valores por defecto ya son buenos y tunear <b>no compensa</b>. Donde es grande (p. ej. anomalías agrupadas), el óptimo suele estar en el <b>interior</b> del rango de ψ.</p>`;
  }

  function init() { if (ML.tuningDoc && ML.TUNING_DOC) ML.tuningDoc.render(ML.TUNING_DOC, esc); terms(); takeaways(); checks(); cases(); nav(); tunability(); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
