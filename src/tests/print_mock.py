"""Render printable mock-exam PDFs (questions with working space; full solutions) from frag/14-mock.html.
Run: python3 site/src/tests/print_mock.py   -> site/dist/mock-{1,2}-{questions,solutions}.pdf"""
import os, re, threading, http.server, functools
from playwright.sync_api import sync_playwright
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DIST = os.path.join(os.path.dirname(SRC), 'dist')
frag = open(os.path.join(SRC, 'frag', '14-mock.html'), encoding='utf-8').read()
frag = re.sub(r'<script.*?</script>', '', frag, flags=re.S)
p1 = frag[frag.index('<h3 id="mock1">'):frag.index('<h3 id="mock2">')]
p2 = frag[frag.index('<h3 id="mock2">'):]
CSS = """@page{size:A4;margin:16mm 15mm 16mm 15mm}
body{font-family:'Source Serif 4',Georgia,serif;font-size:11pt;line-height:1.45;color:#000}
h1{font-family:Archivo,Arial,sans-serif;font-size:16pt;margin:0 0 4pt}
h3{font-family:Archivo,Arial,sans-serif;font-size:13pt}
.gp{border-top:1.2px solid #000;padding-top:6pt;margin-top:10pt;break-inside:auto}
.head b{font-family:Archivo,Arial,sans-serif;font-size:11.5pt}.chip{font:600 8pt Arial;border:1px solid #888;border-radius:3px;padding:1px 4px;margin-left:5px}
.marks,.chk{display:none}.added{font:8pt Arial;color:#444}
.work{height:%s;border:1px dashed #999;margin:6pt 0 2pt;border-radius:3px}
details>summary{display:none}details{display:block}
.note{font:9pt Arial;color:#333;border:1px solid #999;padding:6pt;margin:6pt 0}"""
def page(title, body, mode):
    if mode == 'q':
        body = re.sub(r'<details class="sol">.*?</details>', '<div class="work"></div>', body, flags=re.S)
        css = CSS % '62mm'
        note = '<p class="note">Indicative marks in [brackets] (the real paper’s marks are not published). 2 hours. Calculator and printed notes allowed. Write all working.</p>'
    else:
        body = body.replace('<details class="sol">', '<details class="sol" open>')
        css = CSS % '0'
        note = '<p class="note">Model answers with an indicative marking scheme. Every number was verified by script (site/src/checks/mock.py).</p>'
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>{css}</style>
<script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}},svg:{{fontCache:'global'}},startup:{{pageReady:()=>MathJax.startup.defaultPageReady().then(()=>{{document.body.dataset.ready='1'}})}}}};</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.min.js"></script></head>
<body><h1>ENN581 Robot Motion, Control and Planning · {title}</h1>{note}{body}</body></html>"""
os.makedirs(os.path.join(DIST, 'print'), exist_ok=True)
jobs = []
for n, part in ((1, p1), (2, p2)):
    for mode, name in (('q', 'questions'), ('s', 'solutions')):
        fn = f'mock-{n}-{name}'
        open(os.path.join(DIST, 'print', fn + '.html'), 'w', encoding='utf-8').write(page(f'Mock exam {n} — {name}', part, mode))
        jobs.append(fn)
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Q, directory=DIST)); threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as pw:
    br = pw.chromium.launch(); pg = br.new_page()
    for fn in jobs:
        pg.goto(f'http://127.0.0.1:{srv.server_address[1]}/print/{fn}.html', wait_until='networkidle')
        pg.wait_for_function("document.body.dataset.ready === '1'", timeout=60000)
        pg.pdf(path=os.path.join(DIST, fn + '.pdf'), format='A4', print_background=True,
               display_header_footer=True, header_template='<span></span>',
               footer_template='<div style="font:8pt Arial;width:100%;text-align:center">page <span class="pageNumber"></span> of <span class="totalPages"></span></div>')
        print('wrote', fn + '.pdf', os.path.getsize(os.path.join(DIST, fn + '.pdf')) // 1024, 'kB')
    br.close()
srv.shutdown()
