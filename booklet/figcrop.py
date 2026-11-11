#!/usr/bin/env python3
"""Crop a figure from a lecture/practical PDF for the booklet.

usage: python3 figcrop.py <deck> <page> <x0> <y0> <x1> <y1> [width]
  deck   : key in ../dist/notes/ without .pdf (w02-lec, w08-prac, w13-lec, ...)
  page   : 1-based page number
  x0..y1 : crop box as FRACTIONS of the page, origin top-left (0 0 1 1 = whole page)
  width  : LaTeX width (default \\linewidth), e.g. 0.6\\linewidth

Prints the \\slidefig line to paste into the .tex file and writes a preview PNG
to /tmp/figcrop/<deck>-p<page>.png (Read it to check the crop).
Use `python3 figcrop.py <deck> <page> full` to preview the whole page with a 10% grid.
"""
import os, subprocess, sys
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = os.path.join(HERE, '..', 'dist', 'notes')
OUT = '/tmp/figcrop'


def size(pdf):
    o = subprocess.run(['pdfinfo', pdf], capture_output=True, text=True).stdout
    for line in o.splitlines():
        if line.startswith('Page size'):
            w, h = line.split(':', 1)[1].split('pts')[0].split('x')
            return float(w), float(h)
    raise SystemExit('no page size')


def render(pdf, page, dpi=110):
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, f'tmp{os.getpid()}')
    subprocess.run(['pdftoppm', '-r', str(dpi), '-f', str(page), '-l', str(page), '-png', '-singlefile', pdf, base], check=True)
    im = Image.open(base + '.png'); im.load(); os.remove(base + '.png')
    return im


def main(a):
    deck, page = a[0], int(a[1])
    pdf = os.path.join(NOTES, deck + '.pdf')
    if not os.path.exists(pdf):
        raise SystemExit(f'no such deck: {pdf}')
    W, H = size(pdf)
    if a[2] == 'full':
        im = render(pdf, page).convert('RGB')
        d = ImageDraw.Draw(im)
        for k in range(1, 10):
            x = im.width * k / 10; y = im.height * k / 10
            d.line([(x, 0), (x, im.height)], fill=(255, 0, 0), width=1)
            d.line([(0, y), (im.width, y)], fill=(255, 0, 0), width=1)
            d.text((x + 2, 2), f'.{k}', fill=(255, 0, 0)); d.text((2, y + 2), f'.{k}', fill=(255, 0, 0))
        p = os.path.join(OUT, f'{deck}-p{page}-grid.png'); im.save(p); print(p); return
    x0, y0, x1, y1 = map(float, a[2:6])
    width = a[6] if len(a) > 6 else r'\linewidth'
    l, r = x0 * W, (1 - x1) * W
    t, b = y0 * H, (1 - y1) * H
    im = render(pdf, page, 150)
    c = im.crop((int(x0 * im.width), int(y0 * im.height), int(x1 * im.width), int(y1 * im.height)))
    p = os.path.join(OUT, f'{deck}-p{page}.png'); c.save(p)
    print(p)
    print(f'\\slidefig[{width}]{{{deck}}}{{{page}}}{{{l:.0f} {b:.0f} {r:.0f} {t:.0f}}}{{CAPTION}}')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(1)
    main(sys.argv[1:])
