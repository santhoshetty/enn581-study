/* W3D — physically consistent 3D kinematics scenes.
   One rule: every drawn link, joint and frame comes from ONE transform chain (a list of elementary transforms),
   and the matrices printed beside the scene are read from that same chain. World is right-handed, z up.
   Matrices are row-major 4x4 arrays (16 numbers) acting on column vectors: p' = T p. */
(function () {
  'use strict';
  const D2R = Math.PI / 180;
  /* ---------- 4x4 maths (row-major) ---------- */
  const M = {
    I: () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    mul(a, b) { const o = new Array(16).fill(0); for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 4; k++) s += a[i * 4 + k] * b[k * 4 + j]; o[i * 4 + j] = s; } return o; },
    Rx(t) { const c = Math.cos(t), s = Math.sin(t); return [1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1]; },
    Ry(t) { const c = Math.cos(t), s = Math.sin(t); return [c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1]; },
    Rz(t) { const c = Math.cos(t), s = Math.sin(t); return [c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]; },
    T(x, y, z) { return [1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1]; },
    inv(a) { // rigid inverse [R^T, -R^T t]
      const R = [a[0], a[1], a[2], a[4], a[5], a[6], a[8], a[9], a[10]], t = [a[3], a[7], a[11]];
      const Rt = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]];
      const ti = [-(Rt[0] * t[0] + Rt[1] * t[1] + Rt[2] * t[2]), -(Rt[3] * t[0] + Rt[4] * t[1] + Rt[5] * t[2]), -(Rt[6] * t[0] + Rt[7] * t[1] + Rt[8] * t[2])];
      return [Rt[0], Rt[1], Rt[2], ti[0], Rt[3], Rt[4], Rt[5], ti[1], Rt[6], Rt[7], Rt[8], ti[2], 0, 0, 0, 1];
    },
    pos: a => [a[3], a[7], a[11]],
    col: (a, j) => [a[j], a[4 + j], a[8 + j]],
  };
  const ELEM = { Rx: M.Rx, Ry: M.Ry, Rz: M.Rz, tx: d => M.T(d, 0, 0), ty: d => M.T(0, d, 0), tz: d => M.T(0, 0, d) };

  /* ---------- chains ----------
     op: {op:'Rz', v:const} | {op:'Rz', q:i, off:c, sgn:±1} | {frame:'{1}'} ; angles in radians, lengths in metres */
  function evalChain(ops, q) {
    let T = M.I(); const steps = [], frames = [{ label: ops.base || '{0}', T }];
    for (const o of ops.list || ops) {
      if (o.frame) { frames.push({ label: o.frame, T, tool: !!o.tool }); continue; }
      const val = o.q !== undefined ? (o.sgn || 1) * q[o.q] + (o.off || 0) : o.v;
      const before = T; T = M.mul(T, ELEM[o.op](val));
      steps.push({ op: o.op, joint: o.q, before, after: T, val });
    }
    return { T, steps, frames };
  }
  function etsString(list, jointNames) {
    return list.filter(o => !o.frame).map(o => {
      if (o.q !== undefined) { const n = jointNames[o.q] || ('q' + (o.q + 1)); const s = (o.sgn === -1 ? '−' : '') + n + (o.off ? (o.off > 0 ? ' + ' : ' − ') + fmtConst(o.op, Math.abs(o.off)) : ''); return o.op + '(' + s + ')'; }
      return o.op + '(' + fmtConst(o.op, o.v) + ')';
    }).join(' ');
  }
  function fmtConst(op, v) { if (op[0] === 'R') { const d = v / D2R; return (Math.abs(d - Math.round(d)) < 1e-9 ? Math.round(d) : d.toFixed(1)) + '°'; } return (+v.toFixed(4)) + ''; }

  /* proximal DH row -> ops: Rx(alpha_{i-1}) tx(a_{i-1}) Rz(theta_i) tz(d_i) */
  function dhToOps(rows, toolOps) {
    const list = [];
    rows.forEach((r, i) => {
      if (r.alpha) list.push({ op: 'Rx', v: r.alpha * D2R });
      if (r.a) list.push({ op: 'tx', v: r.a });
      if (r.type === 'P') { if (r.theta) list.push({ op: 'Rz', v: r.theta * D2R }); list.push({ op: 'tz', q: i, off: r.d || 0 }); }
      else { list.push({ op: 'Rz', q: i, off: (r.theta || 0) * D2R }); if (r.d) list.push({ op: 'tz', v: r.d }); }
      list.push({ frame: '{' + (i + 1) + '}' });
    });
    (toolOps || []).forEach(o => list.push(o));
    return list;
  }

  /* ---------- presets ---------- */
  const P = {};
  P.planarRRR = { title: 'Planar RRR (slides pp. 6–9)', base: '{0}', joints: [{ n: 'q1', d: 30 }, { n: 'q2', d: 45 }, { n: 'q3', d: -60 }], scale: 0.5,
    list: [{ op: 'Rz', q: 0 }, { frame: '{1}' }, { op: 'tx', v: 0.5 }, { op: 'Rz', q: 1 }, { frame: '{2}' }, { op: 'tx', v: 0.4 }, { op: 'Rz', q: 2 }, { frame: '{3}' }, { op: 'tx', v: 0.2 }, { frame: '{E}', tool: true }] };
  P.furuta = { title: 'Furuta pendulum ETS (p. 19)', base: '{B}', joints: [{ n: 'θ1', d: 60 }, { n: 'θ2', d: 45 }], scale: 0.5,
    list: [{ op: 'tz', v: 0.5 }, { op: 'Rz', q: 0 }, { frame: '{1}' }, { op: 'tx', v: 0.4 }, { op: 'Rx', q: 1 }, { frame: '{2}' }, { op: 'tz', v: -0.3 }, { op: 'Ry', v: Math.PI }, { frame: '{E}', tool: true }] };
  P.furutaDH = { title: 'Furuta pendulum, proximal DH table (p. 34) + ³T_E = Rz(−90°)', base: '{0}', joints: [{ n: 'q1', d: 60 }, { n: 'q2', d: 45 }], scale: 0.5,
    dh: [{ alpha: 0, a: 0, d: 0.5, theta: 90 }, { alpha: 90, a: 0, d: 0.4, theta: 0 }, { alpha: 90, a: 0, d: 0.3, theta: 0, fixed: true }],
    tool: [{ op: 'Rz', v: -Math.PI / 2 }, { frame: '{E}', tool: true }] };
  P.panda = { title: 'Franka Panda, official proximal DH', base: '{0}', scale: 0.9,
    joints: [{ n: 'q1', d: 0 }, { n: 'q2', d: -20 }, { n: 'q3', d: 0 }, { n: 'q4', d: -110 }, { n: 'q5', d: 0 }, { n: 'q6', d: 90 }, { n: 'q7', d: 45 }],
    dh: [{ alpha: 0, a: 0, d: 0.333 }, { alpha: -90, a: 0, d: 0 }, { alpha: 90, a: 0, d: 0.316 }, { alpha: 90, a: 0.0825, d: 0 }, { alpha: -90, a: -0.0825, d: 0.384 }, { alpha: 90, a: 0, d: 0 }, { alpha: 90, a: 0.088, d: 0 }],
    tool: [{ op: 'tz', v: 0.107 }, { frame: '{F}', tool: true }] };
  P.puma = { title: 'PUMA-like arm, ETS (notebook Q5, own lengths)', base: '{0}', scale: 0.9,
    joints: [{ n: 'q1', d: 30 }, { n: 'q2', d: 30 }, { n: 'q3', d: 60 }, { n: 'q4', d: 0 }, { n: 'q5', d: 30 }, { n: 'q6', d: 0 }],
    list: [{ op: 'tz', v: 1.0 }, { op: 'Rz', q: 0 }, { frame: '{1}' }, { op: 'Ry', q: 1 }, { frame: '{2}' }, { op: 'tz', v: 0.5 }, { op: 'Ry', q: 2 }, { frame: '{3}' }, { op: 'tz', v: 0.3 }, { op: 'Rz', q: 3 }, { op: 'Ry', q: 4 }, { op: 'Rz', q: 5 }, { frame: '{W}', tool: true }] };
  P.w13rrr = { title: 'W13 Q1 RRR arm (lecturer’s ETS)', base: '{S}', scale: 0.7,
    joints: [{ n: 'θ1', d: 30 }, { n: 'θ2', d: 20 }, { n: 'θ3', d: 40 }],
    list: [{ op: 'Rz', q: 0 }, { frame: '{1}' }, { op: 'tz', v: 1.1 }, { op: 'Ry', q: 1, sgn: -1 }, { frame: '{2}' }, { op: 'tx', v: 0.4 }, { op: 'Ry', q: 2, sgn: -1 }, { frame: '{3}' }, { op: 'tx', v: 0.3 }, { op: 'Rx', v: Math.PI / 2 }, { frame: '{T}', tool: true }],
    L: [0.6, 0.5, 0.4, 0.3],
    ik(P_, L) { // all branches; returns [{q:[..], name}]
      const [l1, l2, l3, l4] = L, [x, y, z] = P_, out = [];
      for (const s1 of [1, -1]) {
        const th1 = Math.atan2(s1 * y, s1 * x), r = s1 * Math.hypot(x, y), zz = z - (l1 + l2);
        const c3 = (r * r + zz * zz - l3 * l3 - l4 * l4) / (2 * l3 * l4); if (Math.abs(c3) > 1) continue;
        for (const s3 of [1, -1]) {
          const th3 = Math.atan2(s3 * Math.sqrt(1 - c3 * c3), c3), th2 = Math.atan2(zz, r) - Math.atan2(l4 * Math.sin(th3), l3 + l4 * Math.cos(th3));
          out.push({ q: [th1, th2, th3], name: (s1 > 0 ? 'facing' : 'over-the-top') + ', θ3 ' + (s3 > 0 ? '> 0' : '< 0') });
        }
      }
      return out;
    } };
  function presetOps(p) { return p.dh ? dhToOps(p.dh.map(r => ({ ...r })).filter(() => true), p.tool) : p.list; }
  // fixed DH rows (no joint) — Furuta row 3 has theta fixed at 0
  function opsFor(p) {
    if (!p.dh) return p.list;
    const list = []; let j = 0;
    p.dh.forEach((r, i) => {
      if (r.alpha) list.push({ op: 'Rx', v: r.alpha * D2R });
      if (r.a) list.push({ op: 'tx', v: r.a });
      if (r.fixed) list.push({ op: 'Rz', v: (r.theta || 0) * D2R }); else list.push({ op: 'Rz', q: j++, off: (r.theta || 0) * D2R });
      if (r.d) list.push({ op: 'tz', v: r.d });
      list.push({ frame: '{' + (i + 1) + '}' });
    });
    (p.tool || []).forEach(o => list.push(o));
    return list;
  }

  /* ---------- rendering ---------- */
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim() || '#888';
  function webglOK() { try { const c = document.createElement('canvas'); return !!(window.WebGLRenderingContext && (c.getContext('webgl') || c.getContext('experimental-webgl'))); } catch (e) { return false; } }
  function labelSprite(text, color) {
    const c = document.createElement('canvas'); const g = c.getContext('2d'); const fs = 48;
    g.font = '600 ' + fs + 'px IBM Plex Mono, monospace'; const w = Math.ceil(g.measureText(text).width) + 16; c.width = w; c.height = fs + 16;
    g.font = '600 ' + fs + 'px IBM Plex Mono, monospace'; g.textBaseline = 'middle'; g.lineWidth = 8; g.strokeStyle = 'rgba(128,128,128,0.35)'; g.strokeText(text, 8, c.height / 2);
    g.fillStyle = color; g.fillText(text, 8, c.height / 2);
    const tex = new THREE.CanvasTexture(c); tex.minFilter = THREE.LinearFilter;
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
    s.userData.aspect = w / c.height; s.renderOrder = 10; return s;
  }
  function toM4(T) { const m = new THREE.Matrix4(); m.set(...T); return m; }

  class Scene3D {
    constructor(stage, opts) {
      this.stage = stage; this.opts = opts; this.ok = !!window.THREE && webglOK();
      if (!this.ok) { const f = document.createElement('div'); f.className = 'fallback'; f.textContent = '3D view needs WebGL, which this browser has turned off. The matrices and numbers on the right still update.'; stage.appendChild(f); return; }
      const R = this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
      R.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)); stage.appendChild(R.domElement);
      this.scene = new THREE.Scene();
      this.cam = new THREE.PerspectiveCamera(40, 1, 0.01, 100); this.cam.up.set(0, 0, 1);
      this.home = opts.home || [2.2, -2.6, 1.8]; this.target = opts.target || [0, 0, 0.5];
      this.scene.add(new THREE.HemisphereLight(0xffffff, 0x666666, 0.9)); const dl = new THREE.DirectionalLight(0xffffff, 0.6); dl.position.set(3, -2, 5); this.scene.add(dl);
      const grid = new THREE.GridHelper(opts.grid || 2, 20, 0x888888, 0x888888); grid.rotation.x = Math.PI / 2; grid.material.opacity = 0.25; grid.material.transparent = true; this.scene.add(grid); this.grid = grid; if (opts.gridZ !== undefined) grid.position.z = opts.gridZ;
      this.dyn = new THREE.Group(); this.scene.add(this.dyn);
      if (THREE.OrbitControls) { this.ctl = new THREE.OrbitControls(this.cam, R.domElement); this.ctl.addEventListener('change', () => this.render()); }
      this.resetView();
      this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(stage); this.resize();
      window.addEventListener('enn-theme', () => this.render());
    }
    resetView() {
      if (!this.ok) return;
      if (this.fitPts && this.fitPts.length) { // frame the robot's actual extent
        const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
        this.fitPts.forEach(p => p.forEach((v, i) => { lo[i] = Math.min(lo[i], v); hi[i] = Math.max(hi[i], v); }));
        const c = lo.map((v, i) => (v + hi[i]) / 2), rad = Math.max(0.25, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2);
        const h = this.home, n = Math.hypot(...h), k = rad * 3.1 / n;
        this.cam.position.set(c[0] + h[0] * k, c[1] + h[1] * k, c[2] + h[2] * k);
        if (this.ctl) { this.ctl.target.set(...c); this.ctl.update(); } else this.cam.lookAt(...c);
      } else {
        const s = this.opts.scale || 1; this.cam.position.set(this.home[0] * s, this.home[1] * s, this.home[2] * s);
        if (this.ctl) { this.ctl.target.set(...this.target.map(v => v * s)); this.ctl.update(); } else this.cam.lookAt(...this.target);
      }
      this.render();
    }
    fit(pts) { this.fitPts = pts; this.resetView(); }
    resize() { if (!this.ok) return; const w = this.stage.clientWidth, h = Math.max(this.stage.clientHeight, 300); this.renderer.setSize(w, h, false); this.renderer.domElement.style.height = h + 'px'; this.cam.aspect = w / h; this.cam.updateProjectionMatrix(); this.render(); }
    clear() { if (!this.ok) return; this.dyn.children.slice().forEach(c => { this.dyn.remove(c); c.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) { if (o.material.map) o.material.map.dispose(); o.material.dispose(); } }); }); }
    cyl(a, b, r, color, opacity = 1) {
      const A = new THREE.Vector3(...a), B = new THREE.Vector3(...b), L = A.distanceTo(B); if (L < 1e-6) return;
      const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, L, 16), new THREE.MeshStandardMaterial({ color, roughness: 0.6, transparent: opacity < 1, opacity }));
      m.position.copy(A).add(B).multiplyScalar(0.5); m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), B.clone().sub(A).normalize()); this.dyn.add(m);
    }
    arrow(o, dir, len, color, head = 0.18) { const a = new THREE.ArrowHelper(new THREE.Vector3(...dir).normalize(), new THREE.Vector3(...o), len, color, len * head, len * head * 0.55); this.dyn.add(a); }
    triad(T, s, label, emph) {
      const o = M.pos(T); const cols = [css('--axx'), css('--axy'), css('--axz')];
      [0, 1, 2].forEach(j => this.arrow(o, M.col(T, j), s, new THREE.Color(cols[j]).getHex(), 0.22));
      if (label) { const sp = labelSprite(label, css('--ink')); const h = s * (emph ? 0.42 : 0.34); sp.scale.set(h * sp.userData.aspect, h, 1); sp.position.set(o[0] - s * 0.15, o[1] - s * 0.15, o[2] + s * 0.25); this.dyn.add(sp); }
      ['x', 'y', 'z'].forEach((n, j) => { const c = M.col(T, j); const sp = labelSprite(n, cols[j]); const h = s * 0.26; sp.scale.set(h * sp.userData.aspect, h, 1); sp.position.set(o[0] + c[0] * s * 1.12, o[1] + c[1] * s * 1.12, o[2] + c[2] * s * 1.12); this.dyn.add(sp); });
    }
    points(pts, color, size = 0.012) { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pts.flat(), 3)); const p = new THREE.Points(g, new THREE.PointsMaterial({ color, size, transparent: true, opacity: 0.55 })); this.dyn.add(p); }
    render() { if (this.ok) this.renderer.render(this.scene, this.cam); }
  }

  /* draw a chain: links along translations, joint cylinders along rotation axes, triads at frames */
  function drawChain(sc, ops, q, o = {}) {
    const ev = evalChain(ops, q); if (!sc.ok) return ev;
    const s = o.scale || 0.5, linkR = s * 0.035, jointR = s * 0.07, jointL = s * 0.22;
    const linkCol = new THREE.Color(css('--muted')).getHex(), jc = new THREE.Color(css('--accent')).getHex();
    ev.steps.forEach(st => {
      const a = M.pos(st.before), b = M.pos(st.after);
      if (st.op[0] === 't') sc.cyl(a, b, st.joint !== undefined ? linkR * 1.6 : linkR, st.joint !== undefined ? jc : linkCol);
      else if (st.joint !== undefined) { const ax = M.col(st.before, { Rx: 0, Ry: 1, Rz: 2 }[st.op]); sc.cyl(a.map((v, i) => v - ax[i] * jointL / 2), a.map((v, i) => v + ax[i] * jointL / 2), jointR, jc, 0.85); }
    });
    ev.frames.forEach((f, i) => { if (o.only && !o.only.includes(f.label)) return; sc.triad(f.T, s * (f.tool || i === 0 ? 0.42 : 0.3), f.label, f.tool || i === 0); });
    return ev;
  }

  /* points swept near the start pose (origins of every frame), used to frame the camera */
  function sweepPoints(ops, q0) {
    const pts = [[0, 0, 0]]; let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let k = 0; k < 24; k++) {
      const q = q0.map(v => k ? v + (rnd() - 0.5) * 1.6 : v);
      evalChain(ops, q).frames.forEach(f => pts.push(M.pos(f.T)));
    }
    return pts;
  }

  /* ---------- side-panel helpers ---------- */
  const fmt = v => (Math.abs(v) < 5e-5 ? 0 : v).toFixed(4).replace(/^-0\.0000$/, '0.0000');
  function matHTML(T, label) {
    const cells = []; for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) cells.push('<span>' + fmt(T[i * 4 + j]) + '</span>');
    return (label ? '<div class="eyebrow" style="margin-bottom:.2rem">' + label + '</div>' : '') + '<div class="mat" style="grid-template-columns:repeat(4,auto)">' + cells.join('') + '</div>';
  }
  function sliderRow(id, name, min, max, val, step, unit) {
    return '<div class="row"><label for="' + id + '">' + name + '</label><input type="range" id="' + id + '" min="' + min + '" max="' + max + '" step="' + step + '" value="' + val + '"><output for="' + id + '">' + val + unit + '</output></div>';
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

  /* ---------- widget kinds ---------- */
  const kinds = {};
  const instances = {};

  // Chain explorer: preset selector, joint sliders, choose which transform to print.
  kinds.chain = function (root) {
    const presets = (root.dataset.presets || 'planarRRR').split(',');
    const id = root.id || ('w' + Math.random().toString(36).slice(2, 7)); root.id = id;
    const stage = root.querySelector('.stage'), side = root.querySelector('.side');
    let pkey = presets[0], p = P[pkey], ops = opsFor(p), q = p.joints.map(j => j.d * D2R);
    const sc = new Scene3D(stage, { scale: p.scale, home: [2.2, -2.6, 1.9], target: [0, 0, 0.45] });
    sc.fitPts = sweepPoints(ops, q);
    root.querySelector('.reset') && root.querySelector('.reset').addEventListener('click', () => sc.resetView());
    function buildSide() {
      const ev = evalChain(ops, q);
      side.innerHTML = (presets.length > 1 ? '<label class="eyebrow" for="' + id + '-p">Robot</label><select id="' + id + '-p">' + presets.map(k => '<option value="' + k + '"' + (k === pkey ? ' selected' : '') + '>' + esc(P[k].title) + '</option>').join('') + '</select>' : '<div class="eyebrow">' + esc(p.title) + '</div>') +
        p.joints.map((j, i) => sliderRow(id + '-q' + i, j.n, -180, 180, Math.round(q[i] / D2R), 1, '°')).join('') +
        '<label class="eyebrow" for="' + id + '-f">Show transform</label><select id="' + id + '-f">' +
        ev.frames.map((f, i) => i ? '<option value="0-' + i + '"' + (f.tool ? ' selected' : '') + '>' + ev.frames[0].label + ' → ' + f.label + '</option>' : '').join('') +
        ev.frames.map((f, i) => i > 1 ? '<option value="' + (i - 1) + '-' + i + '">' + ev.frames[i - 1].label + ' → ' + f.label + ' (adjacent)</option>' : '').join('') + '</select>' +
        '<div class="out"></div><div class="eyebrow">Elementary transform sequence</div><div class="mono" style="font-size:.78rem;line-height:1.45;word-break:break-word">' + esc(etsString(ops, p.joints.map(j => j.n))) + '</div>';
      const ps = side.querySelector('#' + id + '-p');
      ps && ps.addEventListener('change', () => { pkey = ps.value; p = P[pkey]; ops = opsFor(p); q = p.joints.map(j => j.d * D2R); sc.opts.scale = p.scale; buildSide(); sc.fit(sweepPoints(ops, q)); update(); });
      p.joints.forEach((j, i) => { const s = side.querySelector('#' + id + '-q' + i); s.addEventListener('input', () => { q[i] = +s.value * D2R; s.nextElementSibling.textContent = s.value + '°'; update(); }); });
      side.querySelector('#' + id + '-f').addEventListener('change', update);
    }
    function update() {
      sc.clear(); const ev = drawChain(sc, ops, q, { scale: p.scale });
      const [a, b] = side.querySelector('#' + id + '-f').value.split('-').map(Number);
      const Ta = ev.frames[a].T, Tb = ev.frames[b].T, Tab = M.mul(M.inv(Ta), Tb);
      side.querySelector('.out').innerHTML = matHTML(Tab, '<sup>' + ev.frames[a].label.replace(/[{}]/g, '') + '</sup>T<sub>' + ev.frames[b].label.replace(/[{}]/g, '') + '</sub> (position in metres)');
      sc.render();
      window.__state[id] = { preset: pkey, q_deg: q.map(v => v / D2R), T_end: ev.frames.filter(f => f.tool).map(f => f.T)[0] || ev.T, frames: ev.frames.map(f => f.label) };
    }
    buildSide(); update();
    instances[id] = { setQ(qd) { q = qd.map(v => v * D2R); buildSide(); update(); return window.__state[id]; }, setPreset(k) { pkey = k; p = P[k]; ops = opsFor(p); q = p.joints.map(j => j.d * D2R); sc.opts.scale = p.scale; buildSide(); sc.fit(sweepPoints(ops, q)); update(); return window.__state[id]; } };
  };

  // DH table editor (proximal): edit alpha_{i-1}, a_{i-1}, d_i, theta_i offset; joint sliders.
  kinds.dh = function (root) {
    const id = root.id || ('w' + Math.random().toString(36).slice(2, 7)); root.id = id;
    const stage = root.querySelector('.stage'), side = root.querySelector('.side');
    const presets = { furutaDH: P.furutaDH, panda: P.panda, rrr: { title: 'Planar RRR as DH', joints: [{ n: 'q1', d: 30 }, { n: 'q2', d: 45 }, { n: 'q3', d: -60 }], scale: 0.5, dh: [{ alpha: 0, a: 0, d: 0 }, { alpha: 0, a: 0.5, d: 0 }, { alpha: 0, a: 0.4, d: 0 }], tool: [{ op: 'tx', v: 0.2 }, { frame: '{E}', tool: true }] } };
    let key = root.dataset.preset || 'furutaDH';
    let rows, tool, joints, q, scale;
    function loadPreset(k) { key = k; const p = presets[k]; rows = p.dh.map(r => ({ ...r })); tool = p.tool; scale = p.scale; joints = p.joints; q = joints.map(j => j.d * D2R); }
    loadPreset(key);
    const sc = new Scene3D(stage, { scale, home: [2.2, -2.6, 1.9], target: [0, 0, 0.45] });
    function ops() { return opsFor({ dh: rows, tool }); }
    sc.fit(sweepPoints(ops(), q));
    root.querySelector('.reset') && root.querySelector('.reset').addEventListener('click', () => sc.fit(sweepPoints(ops(), q)));
    function buildSide() {
      let j = 0;
      side.innerHTML = '<label class="eyebrow" for="' + id + '-p">Preset</label><select id="' + id + '-p">' + Object.keys(presets).map(k => '<option value="' + k + '"' + (k === key ? ' selected' : '') + '>' + esc(presets[k].title) + '</option>').join('') + '</select>' +
        '<div class="tbl"><table class="dh-table"><tr><th>i</th><th>α<sub>i−1</sub> (°)</th><th>a<sub>i−1</sub> (m)</th><th>d<sub>i</sub> (m)</th><th>θ<sub>i</sub></th></tr>' +
        rows.map((r, i) => '<tr><td>' + (i + 1) + '</td><td><input data-r="' + i + '" data-k="alpha" value="' + (r.alpha || 0) + '" aria-label="alpha ' + i + '"></td><td><input data-r="' + i + '" data-k="a" value="' + (r.a || 0) + '" aria-label="a ' + i + '"></td><td><input data-r="' + i + '" data-k="d" value="' + (r.d || 0) + '" aria-label="d ' + (i + 1) + '"></td><td class="mono" style="font-size:.78rem">' + (r.fixed ? (r.theta || 0) + '° (fixed)' : (joints[j++] || { n: 'q' }).n + (r.theta ? ' + ' + r.theta + '°' : '')) + '</td></tr>').join('') + '</table></div>' +
        joints.map((jj, i) => sliderRow(id + '-q' + i, jj.n, -180, 180, Math.round(q[i] / D2R), 1, '°')).join('') + '<div class="out"></div>';
      side.querySelector('#' + id + '-p').addEventListener('change', e => { loadPreset(e.target.value); sc.opts.scale = scale; buildSide(); sc.fit(sweepPoints(ops(), q)); update(); });
      side.querySelectorAll('input[data-r]').forEach(inp => inp.addEventListener('change', () => { const v = parseFloat(inp.value); if (!isNaN(v)) { rows[+inp.dataset.r][inp.dataset.k] = v; update(); } }));
      joints.forEach((jj, i) => { const s = side.querySelector('#' + id + '-q' + i); s.addEventListener('input', () => { q[i] = +s.value * D2R; s.nextElementSibling.textContent = s.value + '°'; update(); }); });
    }
    function update() {
      sc.clear(); const ev = drawChain(sc, ops(), q, { scale });
      const last = ev.frames[ev.frames.length - 1];
      side.querySelector('.out').innerHTML = matHTML(last.T, '<sup>0</sup>T<sub>' + last.label.replace(/[{}]/g, '') + '</sub> = ∏ Rx(α)·tx(a)·Rz(θ)·tz(d)');
      sc.render(); window.__state[id] = { preset: key, q_deg: q.map(v => v / D2R), T_end: last.T };
    }
    buildSide(); update();
    instances[id] = { setQ(qd) { q = qd.map(v => v * D2R); buildSide(); update(); return window.__state[id]; }, setPreset(k) { loadPreset(k); sc.opts.scale = scale; buildSide(); sc.fit(sweepPoints(ops(), q)); update(); return window.__state[id]; } };
  };

  // W13 arm: IK with branch picker + workspace shell
  kinds.ik13 = function (root) {
    const id = root.id || ('w' + Math.random().toString(36).slice(2, 7)); root.id = id;
    const stage = root.querySelector('.stage'), side = root.querySelector('.side'); const p = P.w13rrr, L = p.L;
    let tgt = [0.4554, 0.2629, 1.4966], branch = 0, showWs = root.dataset.ws === '1', cloud = null;
    const sc = new Scene3D(stage, { scale: 0.75, home: [2.4, -2.8, 2.2], target: [0, 0, 0.9] });
    root.querySelector('.reset') && root.querySelector('.reset').addEventListener('click', () => sc.resetView());
    side.innerHTML = '<div class="eyebrow">Target point P in {S} (m) · L1..L4 = 0.6, 0.5, 0.4, 0.3</div>' +
      ['x', 'y', 'z'].map((n, i) => sliderRow(id + '-t' + i, n, i === 2 ? 0.3 : -0.8, i === 2 ? 1.9 : 0.8, tgt[i], 0.005, '')).join('') +
      '<label class="eyebrow" for="' + id + '-b">IK branch</label><select id="' + id + '-b"></select>' +
      '<label style="font-size:.85rem"><input type="checkbox" id="' + id + '-ws"' + (showWs ? ' checked' : '') + '> show workspace (spherical shell, radii 0.1 m and 0.7 m about the shoulder)</label><div class="out"></div>';
    const bsel = side.querySelector('#' + id + '-b');
    [0, 1, 2].forEach(i => { const s = side.querySelector('#' + id + '-t' + i); s.addEventListener('input', () => { tgt[i] = +s.value; s.nextElementSibling.textContent = (+s.value).toFixed(3); update(); }); });
    bsel.addEventListener('change', () => { branch = +bsel.value; update(); });
    side.querySelector('#' + id + '-ws').addEventListener('change', e => { showWs = e.target.checked; update(); });
    function wsCloud() { if (cloud) return cloud; cloud = []; for (let k = 0; k < 2500; k++) { const th = [Math.random() * 2 * Math.PI - Math.PI, Math.random() * 2 * Math.PI - Math.PI, Math.random() * 2 * Math.PI - Math.PI]; cloud.push(M.pos(evalChain(p.list, th).T)); } return cloud; }
    function update() {
      const sols = p.ik(tgt, L);
      bsel.innerHTML = sols.map((s, i) => '<option value="' + i + '"' + (i === branch ? ' selected' : '') + '>' + (i + 1) + ': ' + s.name + '</option>').join('') || '<option>unreachable</option>';
      if (branch >= sols.length) branch = 0;
      sc.clear();
      if (sc.ok) { sc.points([tgt], new THREE.Color(css('--exam')).getHex(), 0.06); if (showWs) sc.points(wsCloud(), new THREE.Color(css('--accent')).getHex(), 0.012); }
      const out = side.querySelector('.out');
      if (!sols.length) { drawChain(sc, p.list, [0, 0, 0], { scale: 0.75 }); out.innerHTML = '<p class="fb bad">Out of reach: |P − shoulder| must lie between |L3 − L4| = 0.1 m and L3 + L4 = 0.7 m (cos θ3 would exceed 1).</p>'; sc.render(); window.__state[id] = { tgt, sols: [] }; return; }
      const qv = sols[branch].q; const ev = drawChain(sc, p.list, qv, { scale: 0.75 });
      const P_ = M.pos(ev.T); const err = Math.hypot(P_[0] - tgt[0], P_[1] - tgt[1], P_[2] - tgt[2]);
      out.innerHTML = '<div class="mono" style="font-size:.82rem">θ1 = ' + (qv[0] / D2R).toFixed(2) + '°<br>θ2 = ' + (qv[1] / D2R).toFixed(2) + '°<br>θ3 = ' + (qv[2] / D2R).toFixed(2) + '°<br>FK check |P − FK(θ)| = ' + err.toExponential(1) + ' m</div>';
      sc.render(); window.__state[id] = { tgt: tgt.slice(), sols: sols.map(s => s.q.map(v => v / D2R)), err };
    }
    update();
    instances[id] = { setTarget(t, b = 0) { tgt = t.slice(); branch = b; update(); return window.__state[id]; } };
  };

  /* ---------- lazy init ---------- */
  function initIn(panel) {
    if (!panel) return;
    panel.querySelectorAll('.widget[data-w3d]').forEach(root => {
      if (root.dataset.ready) return; root.dataset.ready = '1';
      const k = kinds[root.dataset.w3d]; if (!k) return;
      try { k(root); } catch (e) { console.error('widget', root.dataset.w3d, e); const st = root.querySelector('.stage'); if (st) st.insertAdjacentHTML('beforeend', '<div class="fallback">This widget failed to start: ' + esc(e.message) + '</div>'); }
    });
  }
  window.W3D = { M, evalChain, dhToOps, opsFor, presets: P, kinds, instances, initIn, Scene3D, drawChain, matHTML, sliderRow, labelSprite, css };
})();
