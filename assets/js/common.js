/* Utilidades compartidas: tema claro/oscuro, tooltips y helpers DOM/formatos. */
(function () {
  'use strict';
  const ML = (window.MLLab = window.MLLab || {});

  ML.$ = (sel, root = document) => root.querySelector(sel);
  ML.$$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  ML.clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  ML.fmt = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '—');
  ML.pct = (v, d = 1) => (Number.isFinite(v) ? (v * 100).toFixed(d) + '%' : '—');
  ML.esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  ML.debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

  /** Lee los tokens de color actuales (para dibujar en <canvas>). */
  ML.colors = function () {
    const cs = getComputedStyle(document.documentElement);
    const g = (n) => cs.getPropertyValue(n).trim();
    return {
      bg: g('--bg'), surface: g('--surface'), ink: g('--ink'), ink2: g('--ink-2'), muted: g('--muted'),
      grid: g('--grid'), axis: g('--axis'), accent: g('--accent'),
      s1: g('--s1'), s2: g('--s2'), s3: g('--s3'), s4: g('--s4'),
      dark: (document.documentElement.dataset.theme === 'dark') ||
        (!document.documentElement.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches),
    };
  };

  /* ---------- Tema ---------- */
  function applyTheme(t) {
    if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
    window.dispatchEvent(new Event('themechange'));
  }
  ML.initTheme = function () {
    let saved = null;
    try { saved = localStorage.getItem('ml-theme'); } catch (e) { /* almacenamiento bloqueado */ }
    if (saved) document.documentElement.dataset.theme = saved;
    const btn = ML.$('#themeBtn');
    if (btn) btn.addEventListener('click', () => {
      const next = ML.colors().dark ? 'light' : 'dark';
      try { localStorage.setItem('ml-theme', next); } catch (e) { /* noop */ }
      applyTheme(next);
    });
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => window.dispatchEvent(new Event('themechange')));
  };

  /* ---------- Tooltip ---------- */
  const tipEl = document.createElement('div');
  tipEl.id = 'tip';
  tipEl.setAttribute('role', 'tooltip');
  document.addEventListener('DOMContentLoaded', () => document.body.appendChild(tipEl));

  function place(x, y) {
    const w = tipEl.offsetWidth, h = tipEl.offsetHeight, pad = 10;
    let left = x + 16, top = y + 16;
    if (left + w > innerWidth - pad) left = x - w - 16;
    if (left < pad) left = pad;
    if (top + h > innerHeight - pad) top = y - h - 16;
    if (top < pad) top = pad;
    tipEl.style.left = left + 'px';
    tipEl.style.top = top + 'px';
  }
  ML.tip = {
    show(html, x, y) { tipEl.innerHTML = html; tipEl.classList.add('on'); place(x, y); },
    hide() { tipEl.classList.remove('on'); },
  };

  // Delegación: cualquier elemento con data-tip (texto HTML) muestra su definición al señalarlo o enfocarlo.
  function fromTarget(t) { return t.closest && t.closest('[data-tip]'); }
  document.addEventListener('mousemove', (e) => {
    const el = fromTarget(e.target);
    if (el) ML.tip.show(el.getAttribute('data-tip'), e.clientX, e.clientY);
    else if (!e.target.closest || !e.target.closest('[data-chart-tip]')) ML.tip.hide();
  });
  document.addEventListener('focusin', (e) => {
    const el = fromTarget(e.target);
    if (el) { const r = el.getBoundingClientRect(); ML.tip.show(el.getAttribute('data-tip'), r.left, r.bottom - 12); }
  });
  document.addEventListener('focusout', () => ML.tip.hide());
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') ML.tip.hide(); });
  document.addEventListener('scroll', () => ML.tip.hide(), { passive: true });
})();
