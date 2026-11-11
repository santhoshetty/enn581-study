import json, glob, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(os.path.dirname(__file__), 'nb')
for f in sorted(glob.glob(os.path.join(ROOT, 'Week-*', 'Lecture', '**', '*.ipynb'), recursive=True)):
    w = int(os.path.relpath(f, ROOT).split('-')[1])
    nb = json.load(open(f))
    out = []
    for i, c in enumerate(nb['cells']):
        src = ''.join(c.get('source', ''))
        src = '\n'.join(l for l in src.split('\n') if 'data:image' not in l)
        if c['cell_type'] == 'markdown': out.append(f'<!-- cell {i} md -->\n{src}')
        elif c['cell_type'] == 'code':
            o = ''
            for op in c.get('outputs', []):
                t = ''.join(op.get('text', '')) or ''.join(op.get('data', {}).get('text/plain', ''))
                o += t
            out.append(f'<!-- cell {i} code -->\n```python\n{src}\n```' + (f'\n```text\n{o[:300]}\n```' if o.strip() else ''))
    p = os.path.join(OUT, f'w{w:02d}.md'); open(p, 'w').write('\n\n'.join(out))
    print(p.split('/')[-1], len(nb['cells']), 'cells', os.path.getsize(p), 'bytes')
