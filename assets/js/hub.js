/* Portada: catálogo agrupado por paradigma → familia, con filtro por tipo de supervisión. */
(function () {
  'use strict';
  const ML = window.MLLab;
  const { $, $$, esc } = ML;
  const C = ML.catalog;
  const state = { sup: 'all', paradigm: 'all' };

  const ART = {
    iforest: `<svg viewBox="0 0 240 120" preserveAspectRatio="xMidYMid slice" aria-hidden="true"><rect width="240" height="120" fill="var(--surface-2)"/>
      <g stroke="var(--axis)" stroke-width="1.4"><path d="M120 0v120M0 60h240M60 0v60M180 60v60M0 90h120M180 30h60"/></g>
      <g fill="var(--ink-2)"><circle cx="100" cy="52" r="3"/><circle cx="112" cy="66" r="3"/><circle cx="128" cy="50" r="3"/><circle cx="134" cy="70" r="3"/><circle cx="90" cy="70" r="3"/><circle cx="118" cy="58" r="3"/><circle cx="108" cy="44" r="3"/><circle cx="140" cy="58" r="3"/></g>
      <circle cx="205" cy="20" r="5" fill="var(--s2)"/><circle cx="30" cy="100" r="5" fill="var(--s2)"/></svg>`,
    vae: `<svg viewBox="0 0 240 120" preserveAspectRatio="xMidYMid slice" aria-hidden="true"><rect width="240" height="120" fill="var(--surface-2)"/>
      <g fill="none" stroke="var(--axis)" stroke-width="1.3">
        <path d="M32 24v72M56 36v48M88 52v16M152 52v16M184 36v48M208 24v72"/></g>
      <g stroke="var(--accent)" stroke-width="1.1" opacity=".5"><path d="M32 40L56 48M32 60L56 60M32 80L56 72M56 48L88 58M56 72L88 62M152 58L184 48M152 62L184 72M184 48L208 40M184 60L208 60M184 72L208 80"/></g>
      <rect x="104" y="46" width="32" height="28" rx="7" fill="var(--accent-wash)" stroke="var(--accent)" stroke-width="1.6"/>
      <g fill="var(--s2)"><circle cx="112" cy="60" r="2.6"/><circle cx="120" cy="56" r="2.6"/><circle cx="128" cy="63" r="2.6"/></g>
      <text x="120" y="92" font-size="9" fill="var(--ink-2)" text-anchor="middle" font-family="var(--font)">espacio latente z</text></svg>`,
    generic: `<svg viewBox="0 0 240 120" preserveAspectRatio="xMidYMid slice" aria-hidden="true"><rect width="240" height="120" fill="var(--surface-2)"/>
      <g fill="var(--border-strong)"><circle cx="70" cy="60" r="16"/><circle cx="120" cy="40" r="10"/><circle cx="120" cy="80" r="10"/><circle cx="170" cy="60" r="16"/></g></svg>`,
  };

  function card(a) {
    const sup = C.SUPERVISION[a.supervision];
    const chips = `<span class="chip sup-${a.supervision}" data-tip="${esc(sup.tip)}">${sup.name}</span><span class="chip">${esc(a.task)}</span>`;
    const hp = a.hp ? `<div class="hp-list">${a.hp.map((h) => `<code>${esc(h)}</code>`).join('')}</div>` : '';
    if (a.status === 'live') {
      return `<a class="card algo" href="${a.href}" data-sup="${a.supervision}" data-par="${a.paradigm}">
        <div class="thumb">${ART[a.art] || ART.generic}</div>
        <h4>${esc(a.name)} <span class="chip badge-live">Disponible</span></h4>
        <p>${a.blurb}</p>${hp}<div class="tags">${chips}</div><span class="go">Abrir laboratorio →</span></a>`;
    }
    return `<div class="card algo soon" data-sup="${a.supervision}" data-par="${a.paradigm}">
      <h4>${esc(a.name)} <span class="chip">Próximamente</span></h4>
      <p>${a.blurb}</p>${hp}<div class="tags">${chips}</div></div>`;
  }

  function render() {
    const host = $('#catalog');
    let html = '';
    for (const [pid, p] of Object.entries(C.PARADIGMS)) {
      const alg = C.ALGOS.filter((a) => a.paradigm === pid && (state.sup === 'all' || a.supervision === state.sup));
      if (!alg.length) continue;
      const fams = [...new Set(alg.map((a) => a.family))];
      html += `<section class="par-block" id="par-${pid}"><header class="par-head"><h3>${esc(p.name)}</h3><p>${esc(p.blurb)}</p>
        <span class="par-count">${alg.length} ${alg.length === 1 ? 'modelo' : 'modelos'}${state.sup !== 'all' ? ' · ' + C.SUPERVISION[state.sup].name.toLowerCase() : ''}</span></header>`;
      for (const f of fams) {
        const fam = C.FAMILIES[f], list = alg.filter((a) => a.family === f);
        html += `<div class="fam-block"><h4 class="fam-title" data-tip="${esc(fam.tip)}">${esc(fam.name)}<span class="fam-n">${list.length}</span></h4>
          <div class="algos">${list.map(card).join('')}</div></div>`;
      }
      html += '</section>';
    }
    host.innerHTML = html || '<p class="empty-cat">No hay modelos de ese tipo todavía.</p>';
  }

  function buildFilters() {
    const f = $('#supFilter');
    const opts = [['all', 'Todos', 'Muestra todos los modelos del catálogo.']].concat(Object.entries(C.SUPERVISION).map(([k, v]) => [k, v.name, v.tip]));
    f.innerHTML = opts.map(([k, n, tip]) => `<button type="button" data-sup="${k}" aria-pressed="${k === state.sup}" data-tip="${esc(tip)}">${n}</button>`).join('');
    $$('button', f).forEach((b) => b.addEventListener('click', () => {
      state.sup = b.dataset.sup;
      $$('button', f).forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      render();
    }));
  }

  function init() { ML.initTheme(); buildFilters(); render(); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
