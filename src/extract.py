import subprocess, json, glob, os, re, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(os.path.dirname(__file__), 'extract')
docs = {}
for w in range(1, 14):
    for f in sorted(glob.glob(os.path.join(ROOT, f'Week-{w}-*', '**', '*.pdf'), recursive=True)):
        rel = os.path.relpath(f, ROOT)
        if 'week 10 practicals/' in rel or 'Cross_Product' in rel and False: pass
        if 'w10_solution' in rel or 'Comparison_of_two' in rel or ('week 10 practicals' in rel): continue
        kind = 'lec' if 'Lecture -' in rel else ('prac' if 'Practicals' in rel else 'other')
        key = f'w{w:02d}-{kind}' if kind != 'other' else f'w{w:02d}-' + re.sub(r'\W+', '_', os.path.basename(f)[:20]).lower()
        docs[key] = rel
table = []
for key, rel in docs.items():
    path = os.path.join(ROOT, rel)
    n = int(re.search(r'Pages:\s+(\d+)', subprocess.run(['pdfinfo', path], capture_output=True, text=True).stdout).group(1))
    d = os.path.join(OUT, key); os.makedirs(d, exist_ok=True)
    counts = []
    for p in range(1, n + 1):
        t = subprocess.run(['pdftotext', '-layout', '-f', str(p), '-l', str(p), path, '-'], capture_output=True, text=True).stdout
        t = re.sub(r'TEQSA Provider ID.*\n?', '', t); t = re.sub(r'ENN581 Robot Motion, Control and Planning\s*\n\s*Dr\..*\n?', '', t)
        t = re.sub(r'[ \t]+', ' ', t); t = re.sub(r'\n\s*\n+', '\n', t).strip()
        open(os.path.join(d, f'p{p:03d}.txt'), 'w').write(t)
        counts.append(len(t))
    table.append((key, rel, n, counts))
with open(os.path.join(OUT, 'index.md'), 'w') as fh:
    for key, rel, n, counts in table:
        fh.write(f'## {key} ({n} pp) — {rel}\n')
        fh.write(' '.join(f'{i+1}:{c}' for i, c in enumerate(counts)) + '\n\n')
print('\n'.join(f'{k}\t{n}pp\t{r}' for k, r, n, _ in table))
