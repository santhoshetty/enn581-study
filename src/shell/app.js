/* ENN581 study site — shell: storage, tabs, theme, maths, viewer, problems, marks, videos, downloads. */
(function () {
  'use strict';
  const PFX = 'enn581:';
  const store = {
    get(k, d) { try { const v = localStorage.getItem(PFX + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(PFX + k, JSON.stringify(v)); return true; } catch (e) { return false; } },
    del(k) { try { localStorage.removeItem(PFX + k); } catch (e) { /* ignore */ } },
  };
  window.ENN = { store };
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const DOCS = JSON.parse(($('#docs-data') || { textContent: '{}' }).textContent);
  ENN.DOCS = DOCS;

  function toast(msg, ms = 2600) {
    const t = document.createElement('div'); t.className = 'toast'; t.textContent = msg; t.setAttribute('role', 'status');
    document.body.appendChild(t); setTimeout(() => t.remove(), ms);
  }
  ENN.toast = toast;

  /* ---------- theme ---------- */
  const themeBtn = $('#theme-btn');
  function applyTheme(t) {
    if (t) document.documentElement.setAttribute('data-theme', t); else document.documentElement.removeAttribute('data-theme');
    if (themeBtn) themeBtn.textContent = t === 'dark' ? '☾ Dark' : t === 'light' ? '☀ Light' : '◐ Auto';
    window.dispatchEvent(new Event('enn-theme'));
  }
  applyTheme(store.get('theme', null));
  themeBtn && themeBtn.addEventListener('click', () => {
    const cur = store.get('theme', null); const next = cur === null ? 'dark' : cur === 'dark' ? 'light' : null;
    store.set('theme', next); applyTheme(next);
  });

  /* ---------- maths ---------- */
  const typeset = new WeakSet();
  ENN.typeset = function (el) {
    if (!el || typeset.has(el)) return;
    const go = () => { if (window.MathJax && MathJax.typesetPromise) { typeset.add(el); MathJax.typesetPromise([el]).catch(e => console.warn(e)); } };
    if (window.MathJax && MathJax.typesetPromise) go(); else { const iv = setInterval(() => { if (window.MathJax && MathJax.typesetPromise) { clearInterval(iv); go(); } }, 200); setTimeout(() => clearInterval(iv), 20000); }
  };

  /* ---------- tabs ---------- */
  const tabs = $$('.tabs button[data-tab]');
  const panels = $$('section[data-panel]');
  const valid = new Set(panels.map(p => p.dataset.panel));
  function show(id, push = true) {
    if (!valid.has(id)) id = 'plan';
    panels.forEach(p => { p.hidden = p.dataset.panel !== id; });
    tabs.forEach(b => b.setAttribute('aria-selected', String(b.dataset.tab === id)));
    const panel = panels.find(p => p.dataset.panel === id);
    ENN.typeset(panel);
    if (window.W3D) W3D.initIn(panel);
    store.set('tab', id);
    if (push && location.hash !== '#' + id) { try { history.replaceState(null, '', '#' + id); } catch (e) { location.hash = id; } }
    if (viewer.isOpen()) viewer.loadForPanel(panel, false);
    if (id === 'plan') refreshPlan();
    window.scrollTo({ top: 0 });
  }
  ENN.show = show;
  tabs.forEach(b => b.addEventListener('click', () => show(b.dataset.tab)));
  window.addEventListener('hashchange', () => show(location.hash.slice(1), false));
  document.addEventListener('click', e => {
    const a = e.target.closest('a[data-goto]');
    if (a) { e.preventDefault(); show(a.dataset.goto); const t = a.dataset.anchor && document.getElementById(a.dataset.anchor); if (t) setTimeout(() => t.scrollIntoView({ block: 'start' }), 60); }
  });

  /* ---------- downloads ---------- */
  async function saveFile(path, filename) {
    filename = filename || path.split('/').pop();
    try {
      const res = await fetch(path); if (!res.ok) throw new Error('HTTP ' + res.status);
      const blob = await res.blob();
      const dl = window.claude && window.claude.use ? await window.claude.use('downloads') : null;
      if (!dl) { toast('Downloads are not available in this view. Use "Open in new tab" instead.'); return; }
      await dl.save({ filename, data: blob });
      toast('Saved ' + filename);
    } catch (err) {
      if (err && err.code === 'declined') return;
      toast('Could not save the file' + (err && err.message ? ': ' + err.message : ''));
    }
  }
  ENN.saveFile = saveFile;
  document.addEventListener('click', e => {
    const b = e.target.closest('[data-dl]'); if (!b) return;
    e.preventDefault(); saveFile(b.dataset.dl, b.dataset.name);
  });

  /* ---------- PDF viewer ---------- */
  const viewer = (function () {
    const el = $('#viewer'), pagesEl = $('#viewer .vpages'), sel = $('#vdoc'), zv = $('#vzoom'), pginfo = $('#vpg');
    let pdf = null, docKey = null, base = { w: 960, h: 540 }, scale = 1, fit = store.get('viewer:fit', 'width'), io = null, token = 0, nPages = 0;
    const cache = new Map();
    const dpr = () => Math.min(window.devicePixelRatio || 1, 2);
    function isOpen() { return !el.hidden; }
    function setOpen(v) { el.hidden = !v; document.body.classList.toggle('viewer-open', v); store.set('viewer:open', v); }
    function fillSelect(panelDocs) {
      sel.innerHTML = '';
      const mk = (k) => { const o = document.createElement('option'); o.value = k; o.textContent = DOCS[k].short || DOCS[k].title; return o; };
      if (panelDocs.length) { const g = document.createElement('optgroup'); g.label = 'This topic'; panelDocs.forEach(k => DOCS[k] && g.appendChild(mk(k))); sel.appendChild(g); }
      const g2 = document.createElement('optgroup'); g2.label = 'All documents';
      Object.keys(DOCS).filter(k => !panelDocs.includes(k)).forEach(k => g2.appendChild(mk(k))); sel.appendChild(g2);
      if (docKey) sel.value = docKey;
    }
    function panelDocs(panel) { return (panel && panel.dataset.docs ? panel.dataset.docs.split(',') : []).filter(k => DOCS[k]); }
    async function loadForPanel(panel, force) {
      const docs = panelDocs(panel);
      fillSelect(docs);
      if (!docs.length) return;                      // tabs without notes keep the last document
      const want = store.get('viewer:last:' + panel.dataset.panel, docs[0]);
      if (force || !docKey || !docs.includes(docKey)) await load(docs.includes(want) ? want : docs[0]);
    }
    async function getPdf(key) {
      if (cache.has(key)) return cache.get(key);
      if (!window.pdfjsLib) throw new Error('PDF.js did not load');
      const p = pdfjsLib.getDocument({ url: DOCS[key].path }).promise;
      cache.set(key, p);
      if (cache.size > 2) { const [k0, p0] = cache.entries().next().value; if (k0 !== key) { cache.delete(k0); p0.then(d => d.destroy()).catch(() => {}); } }
      return p;
    }
    async function load(key, page) {
      if (!DOCS[key]) return;
      const myTok = ++token;
      docKey = key; sel.value = key;
      const panel = panels.find(p => !p.hidden); if (panel) store.set('viewer:last:' + panel.dataset.panel, key);
      pagesEl.innerHTML = '<div class="vmsg">Loading ' + DOCS[key].title + '…</div>';
      try {
        pdf = await getPdf(key);
        if (myTok !== token) return;
        nPages = pdf.numPages;
        const p1 = await pdf.getPage(1); const vp = p1.getViewport({ scale: 1 }); base = { w: vp.width, h: vp.height };
        build(); applyFit(); if (page) goto(page); else pagesEl.scrollTop = 0;
      } catch (err) {
        pagesEl.innerHTML = '<div class="vmsg">Could not open this PDF here (' + (err.message || err) + '). Use “Open in new tab”.</div>';
      }
    }
    function build() {
      if (io) io.disconnect();
      pagesEl.innerHTML = '';
      io = new IntersectionObserver(entries => {
        entries.forEach(en => {
          const d = en.target;
          if (en.isIntersecting) render(d); else release(d);
        });
        updateInfo();
      }, { root: pagesEl, rootMargin: '700px 0px' });
      for (let i = 1; i <= nPages; i++) {
        const d = document.createElement('div'); d.className = 'vpage'; d.dataset.n = i;
        const lab = document.createElement('span'); lab.className = 'pno'; lab.textContent = i; d.appendChild(lab);
        pagesEl.appendChild(d); io.observe(d);
      }
      size();
    }
    function size() {
      $$('.vpage', pagesEl).forEach(d => { d.style.width = Math.round(base.w * scale) + 'px'; d.style.height = Math.round(base.h * scale) + 'px'; d.dataset.rs = ''; });
      zv.textContent = Math.round(scale * 100 / 1.3333) + '%';
    }
    async function render(d) {
      const want = scale.toFixed(3);
      if (d.dataset.rs === want || d.dataset.busy === want) return;
      d.dataset.busy = want;
      const tok = token;
      try {
        const page = await pdf.getPage(+d.dataset.n);
        if (tok !== token) return;
        const vp = page.getViewport({ scale: scale * dpr() });
        const c = document.createElement('canvas'); c.width = Math.floor(vp.width); c.height = Math.floor(vp.height);
        await page.render({ canvasContext: c.getContext('2d'), viewport: vp }).promise;
        if (tok !== token || d.dataset.busy !== want) return;
        const old = d.querySelector('canvas'); if (old) old.remove();
        d.appendChild(c); d.dataset.rs = want;
      } catch (e) { /* page render cancelled or failed */ } finally { if (d.dataset.busy === want) d.dataset.busy = ''; }
    }
    function release(d) { const c = d.querySelector('canvas'); if (c) { c.width = 0; c.height = 0; c.remove(); } d.dataset.rs = ''; }
    function currentPage() {
      const top = pagesEl.scrollTop + 20; const h = base.h * scale + 10;
      return Math.min(nPages, Math.max(1, Math.floor(top / h) + 1));
    }
    function updateInfo() { pginfo.textContent = nPages ? 'p. ' + currentPage() + ' / ' + nPages : ''; }
    function setScale(s, keep = true) {
      s = Math.max(0.25 * 1.3333, Math.min(3 * 1.3333, s));
      const frac = keep && pagesEl.scrollHeight ? (pagesEl.scrollTop + pagesEl.clientHeight / 2) / pagesEl.scrollHeight : 0;
      scale = s; size();
      if (keep) pagesEl.scrollTop = frac * pagesEl.scrollHeight - pagesEl.clientHeight / 2;
      $$('.vpage', pagesEl).forEach(d => { const r = d.getBoundingClientRect(), R = pagesEl.getBoundingClientRect(); if (r.bottom > R.top - 700 && r.top < R.bottom + 700) render(d); });
    }
    function applyFit() {
      const W = pagesEl.clientWidth - 30, H = pagesEl.clientHeight - 24;
      if (fit === 'width') setScale(W / base.w, false);
      else if (fit === 'page') setScale(Math.min(W / base.w, H / base.h), false);
      else setScale(scale, false);
    }
    function goto(n) { const d = pagesEl.querySelector('.vpage[data-n="' + n + '"]'); if (d) { pagesEl.scrollTop = d.offsetTop - 8; updateInfo(); } }
    pagesEl.addEventListener('scroll', () => requestAnimationFrame(updateInfo), { passive: true });
    $('#vzin').addEventListener('click', () => { fit = 'manual'; store.set('viewer:fit', fit); setScale(scale * 1.15); });
    $('#vzout').addEventListener('click', () => { fit = 'manual'; store.set('viewer:fit', fit); setScale(scale / 1.15); });
    $('#vfitw').addEventListener('click', () => { fit = 'width'; store.set('viewer:fit', fit); const p = currentPage(); applyFit(); goto(p); });
    $('#vfitp').addEventListener('click', () => { fit = 'page'; store.set('viewer:fit', fit); const p = currentPage(); applyFit(); goto(p); });
    $('#vclose').addEventListener('click', () => setOpen(false));
    $('#vdl').addEventListener('click', () => docKey && saveFile(DOCS[docKey].path, DOCS[docKey].path.split('/').pop()));
    $('#vtab').addEventListener('click', () => { if (docKey) { const a = document.createElement('a'); a.href = DOCS[docKey].path; a.target = '_blank'; a.rel = 'noopener'; a.click(); } });
    $('#vfs').addEventListener('click', async () => {
      const p = currentPage();
      if (document.fullscreenElement) { await document.exitFullscreen().catch(() => {}); return; }
      if (el.classList.contains('fill')) { el.classList.remove('fill'); setTimeout(() => { applyFit(); goto(p); }, 50); return; }
      try { await el.requestFullscreen(); } catch (e) { el.classList.add('fill'); }
      setTimeout(() => { applyFit(); goto(p); }, 120);
    });
    document.addEventListener('fullscreenchange', () => setTimeout(() => { const p = currentPage(); applyFit(); goto(p); }, 120));
    sel.addEventListener('change', () => load(sel.value));
    pagesEl.addEventListener('wheel', e => { if (!(e.ctrlKey || e.metaKey)) return; e.preventDefault(); fit = 'manual'; store.set('viewer:fit', fit); setScale(scale * Math.exp(-e.deltaY * 0.0025)); }, { passive: false });
    let pinch = null;
    pagesEl.addEventListener('touchstart', e => { if (e.touches.length === 2) pinch = { d: Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY), s: scale }; }, { passive: true });
    pagesEl.addEventListener('touchmove', e => { if (pinch && e.touches.length === 2) { const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY); pagesEl.style.transformOrigin = 'center top'; pagesEl.dataset.pz = d / pinch.d; } }, { passive: true });
    pagesEl.addEventListener('touchend', () => { if (pinch && pagesEl.dataset.pz) { fit = 'manual'; store.set('viewer:fit', fit); setScale(pinch.s * (+pagesEl.dataset.pz)); pagesEl.dataset.pz = ''; } pinch = null; });
    let rz; window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { if (isOpen() && pdf) { const p = currentPage(); applyFit(); goto(p); } }, 150); });
    async function open(key, page) {
      const wasOpen = isOpen(); setOpen(true);
      fillSelect(panelDocs(panels.find(p => !p.hidden)));
      if (!wasOpen) await new Promise(r => requestAnimationFrame(r));
      if (key && key !== docKey) await load(key, page);
      else if (!docKey) { const panel = panels.find(p => !p.hidden); await loadForPanel(panel, true); if (page) goto(page); }
      else { if (!wasOpen) applyFit(); if (page) goto(page); }
    }
    ENN.viewer = { open, isOpen, goto: n => goto(n), state: () => ({ doc: docKey, scale, fit, nPages, page: currentPage(), pageWidth: base.w * scale }) };
    return { open, isOpen, setOpen, loadForPanel };
  })();
  document.addEventListener('click', e => {
    const a = e.target.closest('a.pg, [data-view]'); if (!a) return;
    e.preventDefault();
    const doc = a.dataset.doc || a.dataset.view; const page = a.dataset.page ? +a.dataset.page : undefined;
    viewer.open(doc, page);
  });

  /* ---------- numeric answer checks ---------- */
  function parseNum(s) {
    s = String(s).trim().replace(/−/g, '-').replace(/,/g, '.');
    if (!s) return NaN;
    if (/^[-+0-9.eE*/()\spi√sqrt]+$/.test(s)) {
      try { const v = Function('"use strict";const pi=Math.PI,sqrt=Math.sqrt;return (' + s.replace(/√\(?([0-9.]+)\)?/g, 'Math.sqrt($1)') + ')')(); return typeof v === 'number' ? v : NaN; } catch (e) { return NaN; }
    }
    return NaN;
  }
  document.addEventListener('click', e => {
    const b = e.target.closest('.ansrow button.check'); if (!b) return;
    const row = b.closest('.ansrow'); const fb = row.querySelector('.fb') || row.parentElement.querySelector('.fb');
    let allOk = true, msgs = [];
    row.querySelectorAll('input[data-ans]').forEach(inp => {
      const v = parseNum(inp.value), want = +inp.dataset.ans, tol = +(inp.dataset.tol || Math.max(1e-3, Math.abs(want) * 0.01));
      if (isNaN(v)) { allOk = false; msgs.push((inp.dataset.label || 'answer') + ': enter a number'); return; }
      if (Math.abs(v - want) <= tol) return;
      allOk = false;
      let hint = '';
      try { (JSON.parse(inp.dataset.wrong || '[]')).forEach(w => { if (Math.abs(v - w.v) <= (w.tol || tol)) hint = w.msg; }); } catch (err) { /* ignore */ }
      if (!hint && Math.abs(v * Math.PI / 180 - want) <= tol) hint = 'That looks like degrees; this answer is in radians.';
      if (!hint && Math.abs(v * 180 / Math.PI - want) <= tol) hint = 'That looks like radians; this answer is in degrees.';
      if (!hint && Math.abs(v + want) <= tol) hint = 'Right size, wrong sign: check your direction convention.';
      msgs.push((inp.dataset.label || 'answer') + ': not yet' + (hint ? ' — ' + hint : ''));
    });
    if (fb) { fb.className = 'fb ' + (allOk ? 'ok' : 'bad'); fb.textContent = allOk ? 'Correct ✓ (within tolerance). Now compare your working with the solution.' : msgs.join(' · '); }
  });

  /* ---------- self-marks ---------- */
  function markKey(id) { return 'mark:' + id; }
  function initMarks(root = document) {
    $$('.marks[data-id]', root).forEach(m => {
      if (m.dataset.init) return; m.dataset.init = 1;
      const v = store.get(markKey(m.dataset.id), null);
      m.innerHTML = '<button type="button" data-v="ok" title="I got it right">✓</button><button type="button" data-v="bad" title="I got it wrong">✗</button>';
      $$('button', m).forEach(b => b.classList.toggle('on', b.dataset.v === v));
    });
  }
  document.addEventListener('click', e => {
    const b = e.target.closest('.marks button'); if (!b) return;
    const m = b.closest('.marks'); const cur = store.get(markKey(m.dataset.id), null); const next = cur === b.dataset.v ? null : b.dataset.v;
    if (next === null) store.del(markKey(m.dataset.id)); else if (!store.set(markKey(m.dataset.id), next)) toast('Could not save your mark: browser storage is blocked.');
    $$('button', m).forEach(x => x.classList.toggle('on', x.dataset.v === next));
  });
  ENN.markCounts = function () {
    const out = {};
    $$('.marks[data-id]').forEach(m => { const t = m.dataset.topic || (m.closest('[data-topic]') || {}).dataset?.topic || 'other'; out[t] = out[t] || { ok: 0, bad: 0, n: 0 }; out[t].n++; const v = store.get(markKey(m.dataset.id), null); if (v) out[t][v]++; });
    return out;
  };

  /* ---------- study plan ---------- */
  function refreshPlan() {
    $$('input[type=checkbox][data-plan]').forEach(c => { c.checked = !!store.get('plan:' + c.dataset.plan, false); });
    const counts = ENN.markCounts();
    $$('[data-xcount]').forEach(el => { const c = counts[el.dataset.xcount]; el.textContent = c ? (c.bad + ' ✗ · ' + c.ok + ' ✓ of ' + c.n) : '—'; });
    const tot = $$('input[type=checkbox][data-plan]'); const done = tot.filter(c => c.checked).length;
    const pr = $('#plan-progress'); if (pr) pr.textContent = done + ' / ' + tot.length + ' plan items done';
  }
  document.addEventListener('change', e => {
    const c = e.target.closest('input[type=checkbox][data-plan]'); if (!c) return;
    if (!store.set('plan:' + c.dataset.plan, c.checked)) toast('Could not save: browser storage is blocked.');
    refreshPlan();
  });

  /* ---------- videos ---------- */
  const inArtifact = !!(window.claude && window.claude.use && !window.claude.__standin);
  function initVideos(root = document) {
    $$('.vid[data-yt]', root).forEach(v => {
      if (v.dataset.init) return; v.dataset.init = 1;
      const id = v.dataset.yt;
      const acts = document.createElement('div'); acts.className = 'acts';
      if (!inArtifact) {
        const play = document.createElement('button'); play.type = 'button'; play.className = 'btn small primary'; play.textContent = '▶ Play here';
        play.addEventListener('click', () => {
          const f = document.createElement('div'); f.className = 'frame';
          f.innerHTML = '<iframe src="https://www.youtube-nocookie.com/embed/' + id + '?autoplay=1" allow="autoplay; encrypted-media; picture-in-picture" allowfullscreen title="' + (v.querySelector('.t') || {}).textContent + '"></iframe>';
          v.insertBefore(f, acts); play.remove();
        });
        acts.appendChild(play);
      }
      const a = document.createElement('a'); a.className = 'btn small'; a.href = 'https://www.youtube.com/watch?v=' + id + (v.dataset.start ? '&t=' + v.dataset.start : ''); a.target = '_blank'; a.rel = 'noopener'; a.textContent = 'Open on YouTube ↗';
      acts.appendChild(a); v.appendChild(acts);
    });
  }

  /* ---------- countdown ---------- */
  const cd = $('#countdown');
  if (cd) {
    const now = new Date(); const study = new Date('2026-11-10T23:59:00+10:00'); const exam = new Date('2026-11-12T09:00:00+10:00');
    const days = (d) => Math.ceil((d - now) / 86400000);
    cd.textContent = now < study ? 'Study ends Tue 10 Nov (' + days(study) + ' d) · Exam Thu 12 Nov' : now < exam ? 'Exam Thu 12 Nov — review only' : 'Exam done';
  }

  /* ---------- boot ---------- */
  initMarks(); initVideos();
  const startTab = location.hash ? location.hash.slice(1) : store.get('tab', 'plan');
  show(startTab, true);
  if (store.get('viewer:open', false) && window.innerWidth >= 900) { viewer.setOpen(true); viewer.loadForPanel(panels.find(p => !p.hidden), true); }
  if (navigator.storage && navigator.storage.persist) { try { navigator.storage.persist(); } catch (e) { /* ignore */ } }
  window.__state = window.__state || {};
})();
