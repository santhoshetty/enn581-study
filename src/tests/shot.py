"""Screenshot helper: python3 shot.py <tab> <out.png> [selector] [dark|light] [width]"""
import sys, threading, http.server, functools, os
from playwright.sync_api import sync_playwright
D = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'dist')
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
s = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Q, directory=D)); threading.Thread(target=s.serve_forever, daemon=True).start()
tab, out = sys.argv[1], sys.argv[2]; sel = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != '-' else None
theme = sys.argv[4] if len(sys.argv) > 4 else 'light'; width = int(sys.argv[5]) if len(sys.argv) > 5 else 1280
with sync_playwright() as p:
    b = p.chromium.launch(args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
    pg = b.new_page(viewport={'width': width, 'height': 900}, color_scheme=theme)
    pg.goto(f'http://127.0.0.1:{s.server_address[1]}/gh-index.html#{tab}', wait_until='networkidle'); pg.wait_for_timeout(3000)
    if sel:
        el = pg.locator(sel).first; el.scroll_into_view_if_needed(); pg.wait_for_timeout(800); el.screenshot(path=out)
    else:
        pg.screenshot(path=out)
    b.close()
s.shutdown()
