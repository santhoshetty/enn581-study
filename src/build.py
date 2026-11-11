"""Assemble the ENN581 study site from src/frag/*.html + src/shell/* into site/dist/.
Usage: python3 site/src/build.py [--pdfs]   (--pdfs re-copies the lecture PDFs with metadata stripped)
Writes dist/index.html (Artifact body: no doctype/html/head — the publisher adds the skeleton)
and   dist/gh-index.html (full document with a window.claude stand-in, for GitHub Pages)."""
import glob, json, os, re, sys, html

SRC = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(SRC)
ROOT = os.path.dirname(SITE)
DIST = os.path.join(SITE, 'dist')
os.makedirs(os.path.join(DIST, 'notes'), exist_ok=True)

TABS = [  # (id, week label, name, docs)
    ('plan', '', 'Study plan', []),
    ('w01', 'W1', 'Mechanisms', ['w01-lec']),
    ('w02', 'W2', 'Forward kinematics', ['w02-lec']),
    ('w03', 'W3', 'Inverse kinematics', ['w03-lec']),
    ('w04', 'W4', 'Parallel robots', ['w04-lec']),
    ('w05', 'W5', 'Trajectories', ['w05-lec']),
    ('w06', 'W6', 'Velocity & statics', ['w06-lec', 'w06-cross']),
    ('w07', 'W7', 'Dynamics', ['w07-lec', 'w08-prac']),
    ('w08', 'W8', 'Control 1', ['w08-lec', 'w08-prac']),
    ('w09', 'W9', 'Control 2', ['w09-lec', 'w09-prac']),
    ('w10', 'W10', 'Wheeled kinematics', ['w10-lec', 'w10-prac']),
    ('w11', 'W11', 'Wheeled dynamics', ['w11-lec']),
    ('w12', 'W12', 'Planning', ['w12-lec', 'w12-prac']),
    ('sample', 'W13', 'Sample Qs solved', ['w13-lec']),
    ('mock', '', 'Mock exams', []),
    ('qa', '', 'Practice Q&A', []),
    ('booklet', '', 'Booklet', ['booklet']),
]
DOC_SRC = {  # key: (relative source path, title, short)
    'w01-lec': ('Week-1-Mechanisms/Lecture/ENN581 - Week 1 - Lecture - Introduction.pdf', 'W1 Lecture: Introduction / mechanisms', 'W1 lecture'),
    'w02-lec': ('Week-2-FK-Robot-Arms/Lecture/ENN581 - Week 2 - Lecture - Forward Kinematics.pdf', 'W2 Lecture: Forward kinematics', 'W2 lecture'),
    'w03-lec': ('Week-3-IK/Lecture/ENN581 - Week 3 - Lecture - Inverse Kinematics.pdf', 'W3 Lecture: Inverse kinematics', 'W3 lecture'),
    'w04-lec': ('Week-4-Parallel-Manipulators/Lecture/ENN581 - Week 4 - Lecture - Parallel Manipulators.pdf', 'W4 Lecture: Parallel manipulators', 'W4 lecture'),
    'w05-lec': ('Week-5-Trajectories/Lecture/ENN581 - Week 5 - Lecture - Trajectories.pdf', 'W5 Lecture: Trajectories', 'W5 lecture'),
    'w06-lec': ('Week-6-Velocity-Kinematics/Lecture/ENN581 - Week 6 - Lecture - Velocity Kinematics.pdf', 'W6 Lecture: Velocity kinematics', 'W6 lecture'),
    'w06-cross': ('Week-6-Velocity-Kinematics/Lecture/Cross_Product_Clarification.pdf', 'W6 Cross-product clarification (1 p.)', 'W6 cross product'),
    'w07-lec': ('Week-7-Dynamics-of-Manipulators/Lecture/ENN581 - Week 7 - Lecture - Dynamics.pdf', 'W7 Lecture: Dynamics', 'W7 lecture'),
    'w08-lec': ('Week-8-Intro-to-Control/Lecture/ENN581 - Week 8 - Lecture - Control Part 1.pdf', 'W8 Lecture: Control part 1', 'W8 lecture'),
    'w08-prac': ('Week-8-Intro-to-Control/Lecture/ENN581 - Week 8 - Practicals - Dynamics and Control.pdf', 'W8 Practical sheet: Dynamics', 'W8 practical'),
    'w09-lec': ('Week-9-Control-2/Lecture/ENN581 - Week 9 - Lecture - Control Part 2.pdf', 'W9 Lecture: Control part 2', 'W9 lecture'),
    'w09-prac': ('Week-9-Control-2/Practical/ENN581 - Week 9 - Practicals - Control part 2.pdf', 'W9 Practical sheet: Control part 2', 'W9 practical'),
    'w10-lec': ('Week-10-Mobile-of-Kinematics/Lecture/ENN581 - Week 10 - Lecture - Wheeled Robots.pdf', 'W10 Lecture: Wheeled robots', 'W10 lecture'),
    'w10-prac': ('Week-10-Mobile-of-Kinematics/Practical/ENN581 - Week 10 - Practicals - Mobile Robot Kinematics.pdf', 'W10 Practical sheet: Mobile robot kinematics', 'W10 practical'),
    'w11-lec': ('Week-11-Wheeled-Robot-Dynamics/Lecture/ENN581 - Week 11 - Lecture - Wheeled Robots Dynamics.pdf', 'W11 Lecture: Wheeled robot dynamics', 'W11 lecture'),
    'w12-lec': ('Week-12-Planning/Lecture/ENN581 - Week 12 Inked - Lecture - Planning Algorithms.pdf', 'W12 Lecture: Planning algorithms', 'W12 lecture'),
    'w12-prac': ('Week-12-Planning/Practical/ENN581 - Week 12 - Practicals - Planning Algorithms.pdf', 'W12 Practical sheet: Planning', 'W12 practical'),
    'w13-lec': ('Week-13-Exam-Revision/ENN581 - Week 13 - Lecture - Revision.pdf', 'W13 Revision (sample questions)', 'W13 revision'),
}
OWN_DOCS = {'booklet': ('booklet.pdf', 'Revision booklet (print this)', 'Booklet')}


def copy_pdfs():
    import fitz
    for k, (rel, *_rest) in DOC_SRC.items():
        dst = os.path.join(DIST, 'notes', k + '.pdf')
        doc = fitz.open(os.path.join(ROOT, rel))
        doc.set_metadata({})
        try:
            doc.del_xml_metadata()
        except Exception:
            pass
        doc.save(dst, garbage=4, deflate=True)
        doc.close()
        print(f'  notes/{k}.pdf  {os.path.getsize(dst)/1e6:.2f} MB')


def docs_json():
    import fitz
    out = {}
    for k, (rel, title, short) in DOC_SRC.items():
        p = os.path.join(DIST, 'notes', k + '.pdf')
        n = fitz.open(p).page_count if os.path.exists(p) else 0
        out[k] = dict(path=f'notes/{k}.pdf', title=title, short=short, pages=n)
    for k, (fn, title, short) in OWN_DOCS.items():
        p = os.path.join(DIST, fn)
        out[k] = dict(path=fn, title=title, short=short, pages=fitz.open(p).page_count if os.path.exists(p) else 0)
    return out


def read(p):
    with open(p, encoding='utf-8') as f:
        return f.read()


def load_json(name):
    p = os.path.join(SRC, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def fmt_len(sec):
    sec = int(sec)
    return f'{sec // 60}:{sec % 60:02d}'


def videos_html(tid, vids):
    items = vids.get(tid, [])
    if not items:
        return ''
    out = ['<p class="eyebrow" style="margin-top:.8rem">Videos · verified · the player may be blocked inside some viewers, so use “Open on YouTube”</p><div class="vids">']
    for v in items:
        deep = v.get('kind') == 'deep'
        out.append(f'<div class="vid{" deep" if deep else ""}" data-yt="{html.escape(v["id"])}"><div class="t">{html.escape(v["title"])}</div>'
                   f'<div class="m">{html.escape(v["channel"])} · {fmt_len(v["seconds"])}{" · deep dive" if deep else ""}</div>'
                   f'<div class="why">{html.escape(v.get("why", ""))}</div></div>')
    out.append('</div>')
    return ''.join(out)


def reading_html(tid, rd):
    r = rd.get(tid)
    if not r:
        return ''
    def items(lst):
        lis = []
        for it in lst:
            ref = html.escape(it.get('ref') or it.get('title') or '')
            url = it.get('url')
            note = it.get('src') or it.get('why') or it.get('note') or ''
            lis.append('<li>' + (f'<a href="{html.escape(url)}" target="_blank" rel="noopener">{ref}</a>' if url else ref)
                       + (f' <span class="added">{html.escape(note)}</span>' if note in ('unit', 'suggested', 'from unit book.txt') else (f' — {html.escape(note)}' if note else '')) + '</li>')
        return '<ul>' + ''.join(lis) + '</ul>'
    parts = ['<span class="tag" style="color:var(--muted)">Reading map (books are for depth; the slides define the exam)</span>']
    for key, label in (('craig', 'Craig, Introduction to Robotics, 4th ed. (your copy)'), ('lynch_park', 'Lynch &amp; Park, Modern Robotics (free PDF)'),
                       ('corke', 'Corke, Robotics, Vision and Control (QUT library)'), ('other', 'Other'), ('free_notes', 'Free notes and lectures')):
        if r.get(key):
            parts.append(f'<h4>{label}</h4>' + items(r[key]))
    return ''.join(parts)


def build(gh=False):
    vids, rd = load_json('videos.json'), load_json('reading.json')
    frags = {os.path.basename(p)[3:-5] if re.match(r'\d\d-', os.path.basename(p)) else os.path.basename(p)[:-5]: p
             for p in sorted(glob.glob(os.path.join(SRC, 'frag', '*.html')))}
    after_js, body = [], []
    built = set()
    for tid, wk, name, docs in TABS:
        path = frags.get(tid)
        if path:
            s = read(path)
            s = re.sub(r'<!--#include ([\w/.-]+)-->', lambda m: read(os.path.join(SRC, 'frag', m.group(1))) if os.path.exists(os.path.join(SRC, 'frag', m.group(1))) else '<div class="banner-todo">This part is still being written.</div>', s)
            for m in re.finditer(r'<script data-after>(.*?)</script>', s, re.S):
                after_js.append(m.group(1))
            s = re.sub(r'<script data-after>.*?</script>', '', s, flags=re.S)
            s = re.sub(r'<div class="vids" data-vids="(\w+)"></div>', lambda m: videos_html(m.group(1), vids), s)
            s = re.sub(r'<div class="box" data-reading="(\w+)"></div>', lambda m: ('<div class="box">' + reading_html(m.group(1), rd) + '</div>') if rd.get(m.group(1)) else '', s)
            built.add(tid)
        else:
            s = (f'<h2>{html.escape((wk + " · ") if wk else "")}{html.escape(name)}</h2>'
                 '<div class="banner-todo"><b>Not built yet.</b> This tab is still being written and checked. '
                 'Use the lecture viewer meanwhile.</div>')
        body.append(f'<section data-panel="{tid}" data-docs="{",".join(docs)}" data-topic="{tid}" hidden>\n{s}\n</section>')
    todo = ' class="todo"'
    tabs = '\n'.join(
        f'<button type="button" data-tab="{tid}" aria-selected="false"{"" if tid in built else todo}>'
        + (f'<span class="wk">{wk}</span>' if wk else '') + html.escape(name) + '</button>'
        + ('<span class="sep" aria-hidden="true"></span>' if tid in ('plan', 'w12') else '')
        for tid, wk, name, _ in TABS)
    css = read(os.path.join(SRC, 'shell', 'style.css'))
    page = f'''<title>ENN581 Exam Workbook</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,500..800&family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400..700;1,8..60,400&display=swap">
<style>
{css}
</style>
<header class="topbar">
  <div class="brand">ENN581 Exam Workbook<small>Robot Motion, Control &amp; Planning · QUT 2026</small></div>
  <span class="spacer"></span>
  <span class="countdown" id="countdown"></span>
  <button type="button" class="btn small" id="theme-btn" aria-label="Switch colour theme">◐ Auto</button>
</header>
<nav class="tabs" role="tablist" aria-label="Sections">
{tabs}
</nav>
<main id="main">
{chr(10).join(body)}
</main>
<aside id="viewer" hidden aria-label="Lecture notes viewer">
  <div class="vbar">
    <select id="vdoc" aria-label="Document"></select>
    <button type="button" id="vzout" aria-label="Zoom out">−</button><span class="zv" id="vzoom">100%</span><button type="button" id="vzin" aria-label="Zoom in">+</button>
    <button type="button" id="vfitw" title="Fit width">↔ Width</button><button type="button" id="vfitp" title="Fit page">↕ Page</button>
    <button type="button" id="vfs" title="Full screen">⛶</button><button type="button" id="vdl" title="Download PDF">⬇</button><button type="button" id="vtab" title="Open in new tab">↗</button>
    <span class="pginfo" id="vpg"></span>
    <button type="button" id="vclose" aria-label="Close viewer">✕</button>
  </div>
  <div class="vpages" tabindex="0"></div>
</aside>
<script type="application/json" id="docs-data">{json.dumps(docs_json())}</script>
<script>
window.MathJax = {{ tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']], macros: {{ R: '\\\\mathbb{{R}}' }} }},
  svg: {{ fontCache: 'global' }}, startup: {{ typeset: false }}, options: {{ enableMenu: false }} }};
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.min.js" async></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
<script>if (window.pdfjsLib) pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>window.__state = window.__state || {{}};</script>
<script>
{read(os.path.join(SRC, 'shell', 'w3d.js'))}
</script>
<script>
{chr(10).join(after_js)}
</script>
<script>
{read(os.path.join(SRC, 'shell', 'app.js'))}
</script>
'''
    if gh:
        standin = '''<script>
/* GitHub Pages stand-in for the claude.ai runtime: downloads fall back to an <a download> link. */
window.claude = { __standin: true, use: async (name) => name === 'downloads' ? { save: async ({ filename, data }) => {
  const url = URL.createObjectURL(data instanceof Blob ? data : new Blob([data])); const a = document.createElement('a');
  a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 4000);
  return { status: 'saved' }; } } : null };
</script>'''
        page = ('<!doctype html><html lang="en-AU"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
                + standin + '</head><body>' + page + '</body></html>')
    out = os.path.join(DIST, 'gh-index.html' if gh else 'index.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(page)
    print(f'wrote {out} ({os.path.getsize(out)/1e3:.0f} kB); built tabs: {sorted(built)}')


if __name__ == '__main__':
    if '--pdfs' in sys.argv or not glob.glob(os.path.join(DIST, 'notes', '*.pdf')):
        copy_pdfs()
    build(False)
    build(True)
