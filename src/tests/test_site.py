"""Headless Chromium tests for the built site (dist/gh-index.html served over http).
Run: python3 site/src/tests/test_site.py [--quick]"""
import json, os, sys, threading, http.server, functools, glob, time
import numpy as np
from playwright.sync_api import sync_playwright

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(os.path.dirname(SRC), 'dist')
fails, passes = [], []
def check(name, ok, detail=''):
    (passes if ok else fails).append(name)
    print(('[PASS] ' if ok else '[FAIL] ') + name + (('  — ' + str(detail)) if detail and not ok else ''))

class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=DIST))
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f'http://127.0.0.1:{srv.server_address[1]}/gh-index.html'

def chromium_path():
    c = sorted(glob.glob(os.path.expanduser('~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium'))
               + glob.glob(os.path.expanduser('~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing')))
    return c[-1] if c else None

with sync_playwright() as pw:
    args = ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist']
    try:
        br = pw.chromium.launch(args=args)
    except Exception:
        br = pw.chromium.launch(args=args, executable_path=chromium_path())
    ctx = br.new_context(viewport={'width': 1400, 'height': 900})
    page = ctx.new_page()
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.on('console', lambda m: errors.append('console: ' + m.text) if m.type == 'error' and 'favicon' not in m.text else None)
    page.goto(URL, wait_until='networkidle')
    page.wait_for_timeout(1500)

    # tabs: each shows only its panel
    tabs = page.eval_on_selector_all('.tabs button[data-tab]', 'els => els.map(e => e.dataset.tab)')
    ok_tabs = True
    for t in tabs:
        page.click(f'.tabs button[data-tab="{t}"]')
        vis = page.eval_on_selector_all('section[data-panel]', 'els => els.filter(e => !e.hidden && e.offsetParent !== null).map(e => e.dataset.panel)')
        if vis != [t]:
            ok_tabs = False; print('   tab', t, 'visible:', vis)
    check(f'every tab shows only its own panel ({len(tabs)} tabs)', ok_tabs)

    # W2 widgets vs python vectors
    page.click('.tabs button[data-tab="w02"]'); page.wait_for_timeout(1500)
    vec = json.load(open(os.path.join(SRC, 'checks', 'w02_vectors.json')))['vectors']
    webgl = page.evaluate("() => !!document.querySelector('#w2-chain canvas')")
    check('W2 3D scene created a WebGL canvas', webgl)
    worst = 0
    for v in vec:
        st = page.evaluate(f"() => {{ W3D.instances['w2-chain'].setPreset('{v['preset'] if v['preset'] != 'planarRRR' else 'planarRRR'}'); return W3D.instances['w2-chain'].setQ({json.dumps(v['q_deg'])}); }}")
        T = st['T_end']; want = [x for row in v['T'] for x in row]
        worst = max(worst, max(abs(a - b) for a, b in zip(T, want)))
    check(f'W2 chain explorer T matches Python for {len(vec)} test poses (max err {worst:.1e})', worst < 1e-5)
    st = page.evaluate("() => { W3D.instances['w2-dh'].setPreset('furutaDH'); return W3D.instances['w2-dh'].setQ([60,45]); }")
    want = [x for v in vec if v['preset'] == 'furuta' and v['q_deg'] == [60, 45] for row in v['T'] for x in row]
    err = max(abs(a - b) for a, b in zip(st['T_end'], want))
    check(f'W2 DH editor (slide-34 table + Rz(-90°)) equals the ETS at (60°,45°) (err {err:.1e})', err < 1e-5)
    st = page.evaluate("() => { W3D.instances['w2-dh'].setPreset('panda'); return W3D.instances['w2-dh'].setQ([0,0,0,0,0,0,0]); }")
    check('W2 DH editor Panda at q=0: flange (0.088, 0, 0.926)', abs(st['T_end'][3] - 0.088) < 1e-6 and abs(st['T_end'][11] - 0.926) < 1e-6, st['T_end'])

    # ---- other widgets vs their Python vectors ----
    V = lambda n: json.load(open(os.path.join(SRC, 'checks', n)))
    def tab(t): page.click(f'.tabs button[data-tab="{t}"]'); page.wait_for_timeout(900)
    tab('w03'); v3 = V('w03_vectors.json')['ik2']
    st = page.evaluate("() => W3D.instances['w3-ik2r'].setTarget([1.2, 0.8])")
    check('W3 2R IK widget: both branches match Python', np.allclose(st['sols'], v3['sols_deg'], atol=1e-3), st['sols'])
    st = page.evaluate("() => W3D.instances['w3-ik2r'].step(6)")
    check('W3 Newton-Raphson in the widget converges (|e| < 1e-8 after 6 steps)', st['err'] < 1e-8, st['err'])
    tab('w04'); v4 = V('w04_vectors.json')
    st1 = page.evaluate("() => W3D.instances['w4-fourbar'].set(70, 1)"); st2 = page.evaluate("() => W3D.instances['w4-fourbar'].set(70, -1)")
    check('W4 four-bar widget: both assembly modes match Python K', np.allclose(st1['K'], v4['fourbar']['modes']['1']['K'], atol=1e-4) and np.allclose(st2['K'], v4['fourbar']['modes']['-1']['K'], atol=1e-4))
    st3 = page.evaluate("() => W3D.instances['w4-fourbar'].set(90, 1)")
    check('W4 four-bar widget: θ1 = 90° cannot assemble (limit 81.52°)', st3['ok'] is False)
    std = page.evaluate("() => W3D.instances['w4-delta'].set([0.03, -0.02, -0.38])")
    check('W4 delta widget: motor angles match Python and every rod is 0.45 m', np.allclose(std['th'], v4['delta']['th_deg'], atol=1e-3) and np.allclose(std['lens'], 0.45, atol=1e-9), std)
    tab('w07'); v7 = V('w07_vectors.json')['swing']
    sw = page.evaluate("() => W3D.instances['w7-dyn'].simulate([30, 0], 2.0)")
    check(f"W7 2R simulator: state at 2 s matches Python ({sw['q'][0]:.3f}, {sw['q'][1]:.3f})°, energy drift {sw['dE']:.1e} J", np.allclose(sw['q'], v7['qT_deg'], atol=0.05) and abs(sw['dE']) < 1e-6)
    tab('w10'); v10 = V('w10_vectors.json')
    a_ = page.evaluate("() => W3D.instances['w10-wheels'].set('dd', [4, 6])")
    b_ = page.evaluate("() => W3D.instances['w10-wheels'].set('omni', [4, -1, -1])")
    c_ = page.evaluate("() => W3D.instances['w10-wheels'].set('tri', [5, 36.8699])")
    check('W10 wheels widget: diff-drive ICR 0.75 m, omni pose after 2 s, tricycle twist match Python',
          abs(a_['icr'][1] - v10['dd']['icr']) < 1e-6 and np.allclose(b_['pose2'], v10['omni']['pose2'], atol=1e-4) and np.allclose(c_['zeta'], [0.4, 0, 0.5], atol=1e-4), (a_['icr'], b_['pose2'], c_['zeta']))
    tab('sample'); v13 = V('w13_vectors.json')['q3']
    sq = page.evaluate(f"() => W3D.instances['sq-ik'].setTarget({json.dumps(v13['P'])})")
    got = sorted([tuple(np.round(s, 2)) for s in sq['sols']]); want = sorted([tuple(np.round(s, 2)) for s in v13['sols_deg']])
    check('Sample Q3 3D IK widget: four branches match Python', np.allclose(got, want, atol=0.02), got)

    # maths typeset
    tab('w02')
    page.wait_for_timeout(2500)
    nmj = page.evaluate("() => document.querySelectorAll('section[data-panel=w02] mjx-container').length")
    check(f'MathJax typeset the W2 panel ({nmj} formulas)', nmj > 30)

    # answer checker
    page.fill('[data-id="w02-p1"]', '') if False else None
    gp = page.locator('.gp').filter(has=page.locator('[data-id="w02-p1"]'))
    ins = gp.locator('input[data-ans]')
    ins.nth(0).fill('0.0163'); ins.nth(1).fill('0.4525'); ins.nth(2).fill('0.7121')
    gp.locator('button.check').click()
    fb = gp.locator('.fb').inner_text()
    check('answer checker flags the +L2 mistake with its specific hint', 'subtracting' in fb, fb)
    ins.nth(2).fill('0.2879'); gp.locator('button.check').click()
    check('answer checker accepts the right answer', 'Correct' in gp.locator('.fb').inner_text())

    # self marks persist across reload
    page.click('.marks[data-id="w02-q1"] button[data-v="bad"]')
    page.reload(wait_until='networkidle'); page.wait_for_timeout(800)
    on = page.evaluate("() => document.querySelector('.marks[data-id=\"w02-q1\"] button[data-v=bad]').classList.contains('on')")
    check('✓/✗ self-mark persists after reload', on)

    # viewer: opens, renders, zoom changes width, slide link jumps
    page.click('.tabs button[data-tab="w02"]')
    page.click('section[data-panel="w02"] [data-view="w02-lec"]')
    page.wait_for_function("() => document.querySelectorAll('#viewer .vpage canvas').length > 0", timeout=20000)
    s1 = page.evaluate('ENN.viewer.state()')
    check(f"viewer renders the W2 deck ({s1['nPages']} pages)", s1['nPages'] == 50)
    page.click('#vzin'); page.wait_for_timeout(300)
    s2 = page.evaluate('ENN.viewer.state()')
    check('zoom + increases page width', s2['pageWidth'] > s1['pageWidth'] * 1.1, (s1['pageWidth'], s2['pageWidth']))
    page.click('section[data-panel="w02"] a.pg[data-page="28"]'); page.wait_for_timeout(800)
    check('slide link jumps to p.28', page.evaluate('ENN.viewer.state().page') == 28, page.evaluate('ENN.viewer.state().page'))
    page.select_option('#vdoc', 'w13-lec'); page.wait_for_timeout(1500)
    check('document picker switches to the W13 deck', page.evaluate('ENN.viewer.state()')['nPages'] == 23)
    page.click('#vclose')

    # phone width: no horizontal scroll
    m = br.new_page(viewport={'width': 390, 'height': 844})
    m.goto(URL + '#w02', wait_until='networkidle'); m.wait_for_timeout(1500)
    sw = m.evaluate('() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]')
    check(f'390px: no horizontal page scroll ({sw[0]} vs {sw[1]})', sw[0] <= sw[1] + 1)
    m.close()

    # fresh context sees no marks
    c2 = br.new_context(); p2 = c2.new_page(); p2.goto(URL + '#w02', wait_until='networkidle')
    on2 = p2.evaluate("() => document.querySelector('.marks[data-id=\"w02-q1\"] button.on') !== null")
    check('a fresh browser profile starts with no marks', not on2); c2.close()

    errs = [e for e in errors if 'youtube' not in e.lower()]
    check('no page errors', not errs, errs[:5])
    page.screenshot(path=os.path.join(SRC, 'img', 'test-w02.png'), full_page=False)
    br.close()
srv.shutdown()
print('-' * 60); print(f'{len(passes)} passed, {len(fails)} failed'); sys.exit(1 if fails else 0)
