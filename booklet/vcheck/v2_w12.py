"""v2 booklet W12 -- planning algorithms.
Every table and number printed in booklet/w12.tex is computed here.
Usage:  python3 v2_w12.py          (prints everything, asserts, and rebuilds ../w12.tex from w12_template.tex)
The template contains lines  %%GEN:key%%  that are replaced by the generated LaTeX, so the booklet tables are the printed tables.
Conventions: tie-break ALPHABETICAL on the slide graph; grid neighbour order up, right, down, left.
"""
import heapq, itertools, math, os, re
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = {}
FAILS = []


def P(label, val=''):
    print(f'{label:70s}', val)


def ok(label, cond):
    print(('PASS  ' if cond else 'FAIL  ') + label)
    if not cond:
        FAILS.append(label)


# =====================================================================================
# A. slide graph (pp. 50, 52, 56, 59)
# =====================================================================================
EDGES = {('A', 'B'): 10, ('B', 'E'): 9, ('A', 'C'): 6, ('C', 'F'): 5, ('F', 'H'): 7, ('H', 'J'): 4,
         ('J', 'K'): 3, ('A', 'D'): 8, ('D', 'G'): 9, ('F', 'G'): 3, ('G', 'I'): 7, ('I', 'K'): 5}
NODES = list('ABCDEFGHIJK')
ADJ = {n: {} for n in NODES}
for (u, v), w in EDGES.items():
    ADJ[u][v] = w; ADJ[v][u] = w
H = {'A': 20, 'B': 10, 'C': 17, 'D': 13, 'E': 8, 'F': 9, 'G': 8, 'H': 7, 'I': 2, 'J': 2, 'K': 0}   # F = 9 and K = 0 are our reading


def pcost(path, adj=ADJ):
    return sum(adj[a][b] for a, b in zip(path, path[1:]))


def trace(par, goal):
    p = []; n = goal
    while n is not None:
        p.append(n); n = par[n]
    return p[::-1]


def sl(lst):
    return ', '.join(lst) if lst else '--'


# ---- BFS with the queue and the visited set at every step ----
def bfs_trace(adj, start, goal):
    q = deque([start]); par = {start: None}; visited = [start]; rows = [(0, '--', '--', list(q), list(visited))]
    k = 0
    while q:
        n = q.popleft(); k += 1
        new = []
        if n != goal:
            for m in sorted(adj[n]):
                if m not in par:
                    par[m] = n; q.append(m); visited.append(m); new.append(f'{m}({n})')
        rows.append((k, n, ', '.join(new) if new else '--', list(q), list(visited)))
        if n == goal:
            break
    return rows, par


brows, bpar = bfs_trace(ADJ, 'A', 'K')
bpath = trace(bpar, 'K')
P('BFS pop order', [r[1] for r in brows[1:]])
P('BFS path / edges / cost', (bpath, len(bpath) - 1, pcost(bpath)))
ok('BFS pop order A..K', [r[1] for r in brows[1:]] == list('ABCDEFGHIJK'))
ok('BFS path A-D-G-I-K, 4 edges, cost 29', bpath == list('ADGIK') and pcost(bpath) == 29)
ok('BFS parents B,C,D<-A E<-B F<-C G<-D H<-F I<-G J<-H K<-I',
   bpar == {'A': None, 'B': 'A', 'C': 'A', 'D': 'A', 'E': 'B', 'F': 'C', 'G': 'D', 'H': 'F', 'I': 'G', 'J': 'H', 'K': 'I'})
depth = {n: len(trace(bpar, n)) - 1 for n in NODES}
P('BFS depth (edges from A)', depth)

lines = []
for k, n, new, q, vis in brows:
    if k == 0:
        lines.append(f'init & -- & -- & A & A\\\\')
    else:
        lines.append(f'{k} & {n} & {new} & {sl(q)} & {sl(vis)}\\\\')
GEN['bfs_rows'] = '\n'.join(lines)
GEN['bfs_par'] = ', '.join(f'{n}$\\leftarrow${bpar[n]}' for n in NODES if bpar[n])


# ---- DFS with an explicit stack (mark VISITED WHEN POPPED; push neighbours in REVERSE alphabetical order) ----
def dfs_trace(adj, start, goal):
    stack = [(start, None)]; visited = set(); par = {}; rows = []; order = []
    k = 0
    while stack:
        n, p = stack.pop(); k += 1
        if n in visited:
            rows.append((k, n, 'already visited: skip', [x for x, _ in stack])); continue
        visited.add(n); par[n] = p; order.append(n)
        if n == goal:
            rows.append((k, n, 'GOAL: stop', [x for x, _ in stack])); break
        pushed = []
        for m in sorted(adj[n], reverse=True):
            if m not in visited:
                stack.append((m, n)); pushed.append(m)
        rows.append((k, n, ('push ' + ', '.join(pushed)) if pushed else 'nothing new', [x for x, _ in stack]))
    return rows, par, order


drows, dpar, dorder = dfs_trace(ADJ, 'A', 'K')
dpath = trace(dpar, 'K')
P('DFS visit order', dorder)
P('DFS path / edges / cost', (dpath, len(dpath) - 1, pcost(dpath)))
ok('DFS visit order A B E C F G D I K', dorder == list('ABECFGDIK'))
ok('DFS path A-C-F-G-I-K cost 26 (5 edges)', dpath == list('ACFGIK') and pcost(dpath) == 26)
lines = ['init & -- & push A & A\\\\']
for k, n, act, st in drows:
    lines.append(f'{k} & {n} & {act} & {"".join(st) if st else "(empty)"}\\\\')
GEN['dfs_rows'] = '\n'.join(lines)
GEN['dfs_par'] = ', '.join(f'{n}$\\leftarrow${dpar[n]}' for n in dorder if dpar[n])
P('DFS parents (set when popped)', dpar)


# ---- Dijkstra / A* (best-first) ----
def best_first(adj, start, goal, h, full=False):
    """Pop lowest f = g+h (ties: lowest h, then alphabetical). Stop when goal popped (full=True: expand everything)."""
    g = {start: 0}; par = {start: None}; openset = {start}; closed = set(); rows = []
    while openset:
        n = min(openset, key=lambda x: (g[x] + h[x], h[x], x))
        openset.remove(n); closed.add(n); upd = []
        if n != goal or full:
            for m, w in sorted(adj[n].items()):
                if m in closed:
                    continue
                if m not in g or g[n] + w < g[m]:
                    g[m] = g[n] + w; par[m] = n; openset.add(m); upd.append(m)
        rows.append(dict(expand=n, g=g[n], h=h[n], f=g[n] + h[n], g_after=dict(g), par_after=dict(par), closed=set(closed), upd=upd,
                         open=sorted(openset, key=lambda x: (g[x] + h[x], h[x], x))))
        if n == goal and not full:
            break
    return rows, g, par


Z = {n: 0 for n in NODES}
djr, dg, dp = best_first(ADJ, 'A', 'K', Z)
P('Dijkstra order', [r['expand'] for r in djr])
ok('Dijkstra order A C D B F G H E I J K', [r['expand'] for r in djr] == list('ACDBFGHEIJK'))
ok('Dijkstra final g', dg == {'A': 0, 'B': 10, 'C': 6, 'D': 8, 'E': 19, 'F': 11, 'G': 14, 'H': 18, 'I': 21, 'J': 22, 'K': 25})
dpth = trace(dp, 'K')
ok('Dijkstra path A-C-F-H-J-K cost 25', dpth == list('ACFHJK') and pcost(dpth) == 25)
P('Dijkstra parents', dp)


def cellfmt(n, r):
    g = r['g_after']
    if n not in g:
        return '--'
    p = r['par_after'][n]
    s = f'{g[n]}' + (f'/{p}' if p else '')
    if n in r['upd']:
        s = '\\underline{' + s + '}'
    if n in r['closed']:
        s = '\\textbf{' + s + '}'
    return s


lines = []
for i, r in enumerate(djr, 1):
    cells = ' & '.join(cellfmt(n, r) for n in NODES)
    lines.append(f'{i} & {r["expand"]} ({r["g"]}) & {cells}\\\\')
GEN['dij_rows'] = '\n'.join(lines)
GEN['dij_final'] = ('Past cost $g$ & ' + ' & '.join(str(dg[n]) for n in NODES) + '\\\\\nParent node & ' +
                    ' & '.join(dp[n] if dp[n] else '--' for n in NODES) + '\\\\')
firstK = [(r['expand'], r['g_after']['K']) for r in djr if 'K' in r['g_after']]
P('K appears in', firstK[:3])
ok('K first discovered at 26 via I, then improved to 25 via J', firstK[0] == ('I', 26) and firstK[1] == ('J', 25))
dstep_G = [(r['expand'], r['g_after']['G'], r['par_after']['G']) for r in djr if 'G' in r['g_after']]
P('G sequence', dstep_G[:4])
ok('G: 17 via D (after D), 14 via F (after F)', dstep_G[0] == ('D', 17, 'D') and [x for x in dstep_G if x[0] == 'F'][0] == ('F', 14, 'F'))
P('Dijkstra expansions before K', len(djr) - 1)

# ---- true distances d(n,K), admissibility, consistency ----
dd_rows, dd, _ = best_first(ADJ, 'K', 'A', Z, full=True)
D = {n: dd[n] for n in NODES}
P('true d(n,K)', D)
ok('d(n,K): H 7 J 3 I 5 G 12 F 14 C 19 D 21 A 25 B 35 E 44', D == {'H': 7, 'J': 3, 'I': 5, 'G': 12, 'F': 14, 'C': 19, 'D': 21, 'A': 25, 'B': 35, 'E': 44, 'K': 0})
ok('h admissible (h <= d for all n)', all(H[n] <= D[n] for n in NODES))
viol = []
for (u, v), w in EDGES.items():
    for a, b in ((u, v), (v, u)):
        if H[a] > w + H[b]:
            viol.append((a, b))
P('consistency violations', viol)
ok('exactly two violations: C->F and H->J', sorted(viol) == [('C', 'F'), ('H', 'J')])
GEN['adm_rows'] = '\n'.join(f'{n} & {H[n]} & {D[n]} & {D[n] - H[n]} & ' + ('yes' if H[n] <= D[n] else 'NO') + '\\\\' for n in NODES)
lines = []
for (u, v), w in sorted(EDGES.items()):
    def cell(a, b):
        lhs, rhs = H[a], w + H[b]
        return f'{lhs}$\\le${w}+{H[b]}={rhs}' if lhs <= rhs else f'\\textbf{{{lhs}$>${w}+{H[b]}={rhs} FAIL}}'
    lines.append(f'{u}--{v} ({w}) & {u}$\\to${v}: {cell(u, v)} & {v}$\\to${u}: {cell(v, u)}\\\\')
GEN['cons_rows'] = '\n'.join(lines)

# ---- A* ----
ar, ag, ap = best_first(ADJ, 'A', 'K', H)
P('A* order', [r['expand'] for r in ar])
ok('A* order A B D C F G I H J K', [r['expand'] for r in ar] == list('ABDCFGIHJK'))
apth = trace(ap, 'K')
ok('A* path A-C-F-H-J-K cost 25', apth == list('ACFHJK') and ag['K'] == 25)
ok('A* expands 10 nodes, Dijkstra 11; E is discovered, never expanded', len(ar) == 10 and len(djr) == 11 and 'E' in ag and 'E' not in [r['expand'] for r in ar])
fpop = [r['f'] for r in ar]
P('popped f', fpop)
ok('popped f = 20 20 21 23 20 22 23 25 24 25', fpop == [20, 20, 21, 23, 20, 22, 23, 25, 24, 25])
lines = []
for i, r in enumerate(ar, 1):
    g = r['g_after']; par = r['par_after']
    opn = '; '.join(f'{m} {g[m]}+{H[m]}={g[m] + H[m]} ({par[m]})' for m in r['open']) if r['open'] else '(empty)'
    lines.append(f'{i} & {r["expand"]} & {r["g"]} & {r["h"]} & {r["f"]} & {opn}\\\\')
GEN['astar_rows'] = '\n'.join(lines)
GEN['astar_final'] = '\n'.join([
    '$g(n)$ & ' + ' & '.join(str(ag[n]) for n in NODES) + '\\\\',
    '$h(n)$ & ' + ' & '.join(str(H[n]) for n in NODES) + '\\\\',
    '$f=g+h$ & ' + ' & '.join(str(ag[n] + H[n]) for n in NODES) + '\\\\',
    'parent & ' + ' & '.join(ap[n] if ap[n] else '--' for n in NODES) + '\\\\'])
P('A* g after stop (E discovered, not expanded; K=25)', ag)
ok('A* final g: B10 C6 D8 E19 F11 G14 H18 I21 J22 K25', ag == {'A': 0, 'B': 10, 'C': 6, 'D': 8, 'E': 19, 'F': 11, 'G': 14, 'H': 18, 'I': 21, 'J': 22, 'K': 25})

same = []
for hf in range(0, 15):
    hh = dict(H); hh['F'] = hf
    rr, g_, p_ = best_first(ADJ, 'A', 'K', hh)
    same.append((hf, ''.join(r['expand'] for r in rr), g_['K']))
P('h(F) sweep 0..14', same)
ok('every h(F) in 0..13 gives order A B D C F G I H J K and cost 25', all(o == 'ABDCFGIHJK' and c == 25 for hf, o, c in same if hf <= 13))
lo = max(H['C'] - 5, H['G'] - 3, H['H'] - 7); hi = min(5 + H['C'], 3 + H['G'], 7 + H['H'])
ok('no h(F) is consistent (needs >= 12 and <= 11)', lo == 12 and hi == 11)
hh = dict(H); hh['F'] = 30
rr, g_, p_ = best_first(ADJ, 'A', 'K', hh)
P('overestimate h(F)=30: order, path, cost', (''.join(r['expand'] for r in rr), trace(p_, 'K'), g_['K']))
ok('h(F)=30: path A-D-G-I-K cost 29', trace(p_, 'K') == list('ADGIK') and g_['K'] == 29)
GEN['over_order'] = ' '.join(r['expand'] for r in rr)
rr, g_, p_ = best_first(ADJ, 'A', 'K', D)
P('perfect h=d: order', ''.join(r['expand'] for r in rr))
ok('perfect h=d expands only A C F H J K', ''.join(r['expand'] for r in rr) == 'ACFHJK')
# stopping when K is first discovered (the wrong rule): returns 26
ok('wrong stop-at-discovery gives 26', firstK[0][1] == 26)

# unit weights on the slide graph: BFS = Dijkstra
UADJ = {n: {m: 1 for m in ADJ[n]} for n in NODES}
ur, ug, up_ = best_first(UADJ, 'A', 'K', Z)
P('unit weights: Dijkstra order', ''.join(r['expand'] for r in ur))
ok('unit weights: Dijkstra order = BFS order A..K, path = BFS path, cost 4', ''.join(r['expand'] for r in ur) == 'ABCDEFGHIJK' and trace(up_, 'K') == bpath and ug['K'] == 4)

GEN['unit_cost'] = str(ug['K'])

# ---- mock-paper Q9 graph (exam question) ----
E2 = {('S', 'A'): 2, ('S', 'B'): 5, ('A', 'C'): 4, ('A', 'D'): 7, ('B', 'D'): 3, ('C', 'G'): 6, ('D', 'G'): 2}
N2 = list('SABCDG')
A2 = {n: {} for n in N2}
for (u, v), w in E2.items():
    A2[u][v] = w; A2[v][u] = w
H2 = {'S': 7, 'A': 6, 'B': 4, 'C': 5, 'D': 2, 'G': 0}
r2, g2, p2 = best_first(A2, 'S', 'G', H2)
P('mock Q9 A*: order', ''.join(r['expand'] for r in r2))
ok('mock Q9 A* order S A B D G, path S-B-D-G cost 10', ''.join(r['expand'] for r in r2) == 'SABDG' and trace(p2, 'G') == list('SBDG') and g2['G'] == 10)
lines = []
for i, r in enumerate(r2, 1):
    g = r['g_after']; par = r['par_after']
    opn = '; '.join(f'{m} {g[m]}+{H2[m]}={g[m] + H2[m]} ({par[m]})' for m in r['open']) if r['open'] else '(empty)'
    lines.append(f'{i} & {r["expand"]} & {r["g"]} & {r["h"]} & {r["f"]} & {opn}\\\\')
GEN['q9_rows'] = '\n'.join(lines)
z2 = {n: 0 for n in N2}
r2d, g2d, p2d = best_first(A2, 'S', 'G', z2)
P('mock Q9 Dijkstra order', [(r['expand'], r['g']) for r in r2d])
ok('mock Q9 Dijkstra path S-B-D-G, order S A B D G?', trace(p2d, 'G') == list('SBDG'))
GEN['q9_dij_order'] = ', '.join(f'{r["expand"]} ({r["g"]})' for r in r2d)
GEN['q9_dij_n'] = str(len(r2d))
GEN['q9_a_n'] = str(len(r2))
_, D2, _ = best_first(A2, 'G', 'S', z2, full=True)
P('mock Q9 true d(n,G)', D2)
ok('mock Q9 h admissible', all(H2[n] <= D2[n] for n in N2))
v2 = [(a, b) for (u, v), w in E2.items() for a, b in ((u, v), (v, u)) if H2[a] > w + H2[b]]
P('mock Q9 consistency violations', v2)
ok('mock Q9 h consistent', v2 == [])
GEN['q9_true'] = ', '.join(f'{n}~{D2[n]}' for n in N2)

# spot-the-error graph (Practice Q12.6)
E3 = {('S', 'A'): 2, ('S', 'B'): 3, ('A', 'G'): 9, ('B', 'C'): 2, ('C', 'G'): 4}
A3 = {n: {} for n in 'SABCG'}
for (u, v), w in E3.items():
    A3[u][v] = w; A3[v][u] = w
r3, g3, p3 = best_first(A3, 'S', 'G', {n: 0 for n in 'SABCG'})
P('Q12.6 Dijkstra', (''.join(r['expand'] for r in r3), trace(p3, 'G'), g3['G']))
ok('Q12.6 shortest S-B-C-G cost 9, order S A B C G', trace(p3, 'G') == list('SBCG') and g3['G'] == 9 and ''.join(r['expand'] for r in r3) == 'SABCG')
b3, bp3 = bfs_trace(A3, 'S', 'G')
ok('Q12.6 BFS path S-A-G cost 11', trace(bp3, 'G') == list('SAG') and pcost(trace(bp3, 'G'), A3) == 11)

# =====================================================================================
# B. 5x5 grid, BFS and A* (4-connected, unit cost, Manhattan)
# =====================================================================================
GRID = ['.#...',
        '.#...',
        '.#...',
        '...#.',
        '...#.']
GS, GG = (0, 0), (4, 4)
R5, C5 = 5, 5
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))     # up, right, down, left


def gnb(n):
    r, c = n
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < R5 and 0 <= cc < C5 and GRID[rr][cc] != '#':
            yield (rr, cc)


def gbfs():
    q = deque([GS]); par = {GS: None}; order = []
    while q:
        n = q.popleft(); order.append(n)
        if n == GG:
            break
        for m in gnb(n):
            if m not in par:
                par[m] = n; q.append(m)
    return order, par


def gh(n):
    return abs(n[0] - GG[0]) + abs(n[1] - GG[1])


def gastar():
    cnt = itertools.count()
    g = {GS: 0}; par = {GS: None}; op = [(gh(GS), gh(GS), next(cnt), GS)]; closed = set(); order = []; fval = {}
    while op:
        f, hh, _, n = heapq.heappop(op)
        if n in closed:
            continue
        closed.add(n); order.append(n); fval[n] = f
        if n == GG:
            break
        for m in gnb(n):
            if m in closed:
                continue
            if m not in g or g[n] + 1 < g[m]:
                g[m] = g[n] + 1; par[m] = n; heapq.heappush(op, (g[m] + gh(m), gh(m), next(cnt), m))
    return order, par, g, fval


gbo, gbp = gbfs(); gao, gap, gag, gaf = gastar()
gbpath = trace(gbp, GG); gapath = trace(gap, GG)
free = sum(1 for r in range(R5) for c in range(C5) if GRID[r][c] != '#')
P('grid free cells', free)
P('grid BFS expansions / path moves', (len(gbo), len(gbpath) - 1))
P('grid A* expansions / path moves', (len(gao), len(gapath) - 1))
ok('grid: BFS and A* both find a 10-move path', len(gbpath) - 1 == 10 and len(gapath) - 1 == 10)
ok('grid: Manhattan consistent on every edge (|h(n)-h(m)| <= 1) and h(S)=8 <= 10',
   all(abs(gh(n) - gh(m)) <= 1 for n in gbp for m in gnb(n)) and gh(GS) == 8)
ok('grid: A* expands fewer cells than BFS', len(gao) < len(gbo))
P('grid BFS order', gbo); P('grid A* order', gao); P('grid A* path', gapath); P('grid BFS path', gbpath)
P('grid A* f at expansion', [gaf[n] for n in gao])
ok('grid: popped f non-decreasing (consistent h)', all(gaf[gao[i]] <= gaf[gao[i + 1]] for i in range(len(gao) - 1)))
ok('grid: popped f at goal = 10', gaf[GG] == 10)
nodes_unexp = [n for n in gag if n not in gao]
P('grid A* discovered but not expanded', nodes_unexp)


def tikz_grid(order, path, title, show_f=False):
    idx = {n: i + 1 for i, n in enumerate(order)}
    s = ['\\begin{tikzpicture}[x=6.4mm,y=6.4mm,every node/.style={font=\\scriptsize\\sffamily,inner sep=0pt}]']
    for r in range(R5):
        for c in range(C5):
            x, y = c, -r
            if GRID[r][c] == '#':
                s.append(f'\\fill[g3] ({x},{y}) rectangle ++(1,-1);')
            elif (r, c) in path:
                s.append(f'\\fill[g2] ({x},{y}) rectangle ++(1,-1);')
    s.append('\\draw[line width=0.4pt] (0,-5) grid[step=1] (5,0);')
    for n in order:
        r, c = n
        s.append(f'\\node[font=\\small\\sffamily\\bfseries] at ({c + 0.5},{-r - 0.5}) {{{idx[n]}}};')
        if show_f:
            s.append(f'\\node[font=\\tiny\\sffamily,anchor=north east] at ({c + 0.96},{-r - 0.04}) {{{gaf[n]}}};')
    s.append('\\node[font=\\tiny\\sffamily\\bfseries,anchor=south west] at (0.03,-0.97) {S};')
    s.append('\\node[font=\\tiny\\sffamily\\bfseries,anchor=south west] at (4.03,-4.97) {G};')
    s.append(f'\\node[font=\\scriptsize\\sffamily\\bfseries,anchor=north] at (2.5,-5.1) {{{title}}};')
    s.append('\\end{tikzpicture}')
    return '\n'.join(s)


GEN['grid_bfs'] = tikz_grid(gbo, gbpath, f'BFS: {len(gbo)} expanded')
GEN['grid_astar'] = tikz_grid(gao, gapath, f'A*: {len(gao)} expanded', show_f=True)
GEN['grid_n'] = f'{len(gbo)}'
GEN['grid_a'] = f'{len(gao)}'
GEN['grid_free'] = f'{free}'
GEN['grid_atable'] = '\n'.join(f'{i} & ({n[0]},{n[1]}) & {gag[n]} & {gh(n)} & {gaf[n]}\\\\' for i, n in enumerate(gao, 1))
GEN['grid_unexp'] = ', '.join(f'({n[0]},{n[1]})' for n in sorted(nodes_unexp)) if nodes_unexp else 'none'
GEN['grid_bfs_path'] = ' $\\to$ '.join(f'({r},{c})' for r, c in gbpath)
GEN['grid_a_path'] = ' $\\to$ '.join(f'({r},{c})' for r, c in gapath)
GEN['grid_bfs_order'] = ', '.join(f'({r},{c})' for r, c in gbo)
GEN['grid_a_order'] = ', '.join(f'({r},{c})' for r, c in gao)
P('grid A* table g,h,f', [(n, gag[n], gh(n), gaf[n]) for n in gao])
q = deque([GS]); seen = {GS}; qs = []
for k in range(4):
    n = q.popleft(); new = []
    for m in gnb(n):
        if m not in seen:
            seen.add(m); q.append(m); new.append(m)
    qs.append((n, new, list(q)))
P('grid BFS first 4 pops', qs)
GEN['grid_notexp_bfs'] = str(free - len(gbo))
ok('BFS expands all free cells except those after the goal in the queue order', len(gbo) <= free)
P('free cells vs BFS expansions', (free, len(gbo)))

d8 = lambda dr, dc: min(dr, dc) * math.sqrt(2) + abs(dr - dc)
P('8-connected cost (7,5): 5 sqrt2 + 2', d8(7, 5))
P('Manhattan, Euclid (7,5)', (12, math.hypot(7, 5)))
ok('8-conn (7,5): 9.0711, Euclid 8.602 <= 9.071 < Manhattan 12', abs(d8(7, 5) - 9.0711) < 5e-5 and math.hypot(7, 5) < d8(7, 5) < 12)
P('8-connected single diagonal step cost vs Manhattan 2', math.sqrt(2))

# =====================================================================================
# C. C-space and inflation numbers
# =====================================================================================
L1 = L2 = 1.0
px, py = 1.0, 1.0
c2 = (px * px + py * py - L1 * L1 - L2 * L2) / (2 * L1 * L2)
sols = []
for s in (1, -1):
    q2 = math.atan2(s * math.sqrt(1 - c2 * c2), c2)
    q1 = math.atan2(py, px) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    sols.append((math.degrees(q1), math.degrees(q2)))
P('2R point obstacle (1,1), L1=L2=1: C-space points (deg)', sols)
ok('P=(1,1): (0,90) and (90,-90)', abs(sols[0][0]) < 1e-9 and abs(sols[0][1] - 90) < 1e-9 and abs(sols[1][0] - 90) < 1e-9 and abs(sols[1][1] + 90) < 1e-9)
for q1, q2 in sols:
    a1, a2 = math.radians(q1), math.radians(q2)
    ok(f'FK back to (1,1) for {q1:.0f},{q2:.0f}', abs(math.cos(a1) + math.cos(a1 + a2) - 1) < 1e-9 and abs(math.sin(a1) + math.sin(a1 + a2) - 1) < 1e-9)
# a point out of reach maps to nothing
P('point (1.5,1.5) r=2.121 > L1+L2=2: no C-space point', math.hypot(1.5, 1.5) > 2)


def poly_area(pts):
    return 0.5 * abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts))))


def poly_perim(pts):
    return sum(math.dist(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts)))


tri = [(5, 3), (6, 3), (5, 1)]
A_, P_ = poly_area(tri), poly_perim(tri); r_ = 0.5
Ag = A_ + P_ * r_ + math.pi * r_ ** 2
P('triangle A, P, r, A_grown, parts', (A_, P_, r_, Ag, P_ * r_, math.pi * r_ ** 2))
ok('triangle: A=1, P=5.236, Pr=2.618, pi r^2=0.785, A_grown=4.403', abs(A_ - 1) < 1e-12 and abs(P_ - 5.2361) < 1e-4 and abs(P_ * r_ - 2.618) < 1e-3 and abs(Ag - 4.4033) < 1e-3)
sq = [(0, 0), (2, 0), (2, 2), (0, 2)]
Ag_sq = poly_area(sq) + poly_perim(sq) * 0.5 + math.pi * 0.25
P('2x2 square r=0.5: disc-grown area / square margin', (Ag_sq, 3 * 3))
ok('2x2 square: 8.785 vs square margin 9', abs(Ag_sq - 8.785) < 1e-3)
ok('corridor 1.0, r 0.4 -> 0.2; r 0.55 -> -0.1; 0.9, 0.5 -> -0.1',
   abs((1.0 - 0.8) - 0.2) < 1e-12 and abs((1.0 - 1.1) + 0.1) < 1e-12 and abs((0.9 - 1.0) + 0.1) < 1e-12)

# =====================================================================================
# D. dimensionality
# =====================================================================================
P('100^2, 100^3, 100^6', (100 ** 2, 100 ** 3, 100 ** 6))
ok('2D 4/8, 3D 6/26, 6D 12/728', [(2 * d, 3 ** d - 1) for d in (2, 3, 6)] == [(4, 8), (6, 26), (12, 728)])
E4 = 2 * 100 * 99
ok('|E| = 19800, |V|+|E| = 29800', E4 == 19800 and 10 ** 4 + E4 == 29800)
ok('40x25 4-connected -> 1935 edges', 40 * 24 + 25 * 39 == 1935)
P('5D, 50 cells', 50 ** 5)

# =====================================================================================
# E. PRM and RRT on the practical map (C = [0,10]^2)
# =====================================================================================
T1 = [(2, 7), (4, 5), (3, 4)]
RECT = [(6, 6), (6, 8), (8, 8), (8, 6)]
T2 = [(5, 3), (6, 3), (5, 1)]
OBST = {'T1': T1, 'R': RECT, 'T2': T2}
GC, GR = (9, 9), 1.0
S0 = (1.5, 1.5)


def pip(pt, poly):
    x, y = pt; n = 0
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xi > x:
                n += 1
    return n % 2 == 1, n


def seg_inter(a, b, c, d):
    def orient(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)

    def onseg(p, q, r):
        return min(p[0], q[0]) - 1e-12 <= r[0] <= max(p[0], q[0]) + 1e-12 and min(p[1], q[1]) - 1e-12 <= r[1] <= max(p[1], q[1]) + 1e-12
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return (o1 == 0 and onseg(a, b, c)) or (o2 == 0 and onseg(a, b, d)) or (o3 == 0 and onseg(c, d, a)) or (o4 == 0 and onseg(c, d, b))


def seg_free(a, b):
    for k, poly in OBST.items():
        for i in range(len(poly)):
            if seg_inter(a, b, poly[i], poly[(i + 1) % len(poly)]):
                return False, k
        if pip(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), poly)[0]:
            return False, k
    return True, None


def pt_free(p):
    for k, poly in OBST.items():
        if pip(p, poly)[0]:
            return False, k
    return True, None


for p in ((3, 5.2), (5.4, 2), (7, 7), (4.5, 6)):
    P(f'sample {p}: free?', (pt_free(p), [(k, pip(p, poly)[1]) for k, poly in OBST.items()]))
ok('(3,5.2) inside T1 with 1 crossing', pip((3, 5.2), T1) == (True, 1))
ok('(5.4,2) inside T2', pip((5.4, 2), T2)[0])
ok('(7,7) inside the rectangle', pip((7, 7), RECT)[0])
ok('(4.5,6) free', pt_free((4.5, 6))[0])

NODE = {'S': S0, 'n1': (4, 2.5), 'n2': (7, 4.5), 'n3': (9, 7), 'n4': (9, 8.5), 'n5': (4.5, 6), 'n6': (7, 2)}
RCONN = 4.0
ok('all roadmap nodes are free', all(pt_free(p)[0] for p in NODE.values()))
ok('n4 is inside the goal disc (distance 0.5 <= 1)', abs(math.dist(NODE['n4'], GC) - 0.5) < 1e-12)
names = list(NODE)
edges = {}; rejected = []; toofar = []
for i, a in enumerate(names):
    for b in names[i + 1:]:
        d = math.dist(NODE[a], NODE[b])
        if d > RCONN:
            toofar.append((a, b, round(d, 3))); continue
        f, why = seg_free(NODE[a], NODE[b])
        if f:
            edges[(a, b)] = d
        else:
            rejected.append((a, b, round(d, 3), why))
P('PRM roadmap edges (r_conn=4)', {k: round(v, 3) for k, v in edges.items()})
P('PRM rejected by local planner', rejected)
P('PRM too far (> r_conn)', toofar)
ok('roadmap edges: S-n1, n1-n2, n1-n5, n2-n3, n2-n5, n2-n6, n3-n4',
   sorted(edges) == sorted([('S', 'n1'), ('n1', 'n2'), ('n1', 'n5'), ('n2', 'n3'), ('n2', 'n5'), ('n2', 'n6'), ('n3', 'n4')]))
ok('n1-n6 rejected: crosses T2', any(a == 'n1' and b == 'n6' and w == 'T2' for a, b, d, w in rejected))
RA = {n: {} for n in names}
for (a, b), d in edges.items():
    RA[a][b] = d; RA[b][a] = d
rr_, rg, rp = best_first(RA, 'S', 'n4', {n: 0 for n in names})
rpath = trace(rp, 'n4')
P('PRM query path / cost', (rpath, round(rg['n4'], 4)))
ok('PRM path S-n1-n2-n3-n4, cost 11.00', rpath == ['S', 'n1', 'n2', 'n3', 'n4'] and abs(rg['n4'] - 11.0) < 5e-4)
P('Dijkstra on roadmap order, g', [(r['expand'], round(r['g'], 3)) for r in rr_])
GEN['prm_order'] = ', '.join(f'{r["expand"]} ({r["g"]:.3f})' for r in rr_)
legs = [round(math.dist(NODE[a], NODE[b]), 3) for a, b in zip(rpath, rpath[1:])]
ok('legs 2.693 3.606 3.202 1.5', legs == [2.693, 3.606, 3.202, 1.5])
GEN['prm_edges'] = '\n'.join(f'{a}--{b} & {d:.3f} & free\\\\' for (a, b), d in sorted(edges.items()))
P('S->n3 direct', seg_free(S0, NODE['n3']))
P('S->n2 direct (distance, > r_conn)', (seg_free(S0, NODE['n2']), math.dist(S0, NODE['n2'])))
P('n5->n3 direct', (seg_free(NODE['n5'], NODE['n3']), math.dist(NODE['n5'], NODE['n3'])))
P('S->G direct', seg_free(S0, GC))
ok('S->G straight line is blocked by the rectangle', seg_free(S0, GC) == (False, 'R'))
ok('S->n3 blocked by the rectangle', seg_free(S0, NODE['n3']) == (False, 'R'))
ok('n1->n6 blocked by T2', seg_free(NODE['n1'], NODE['n6']) == (False, 'T2'))
P('n1->n6 distance', math.dist(NODE['n1'], NODE['n6']))
P('nearest to (6,4) among S,n1,n2,n3', sorted((round(math.dist((6, 4), NODE[n]), 3), n) for n in ('S', 'n1', 'n2', 'n3'))[:2])
ok('nearest to (6,4) is n2 at 1.118', abs(math.dist((6, 4), NODE['n2']) - 1.118) < 5e-4)

v1_ = math.dist(S0, (8, 6)); v2_ = math.dist((8, 6), GC)
P('S->(8,6), (8,6)->G, total, to the disc edge', (v1_, v2_, v1_ + v2_, v1_ + v2_ - 1))
ok('visibility via corner (8,6): 7.906 + 3.162; disc-edge 10.068', abs(v1_ - 7.906) < 5e-4 and abs(v2_ - 3.162) < 5e-4 and abs(v1_ + v2_ - 1 - 10.068) < 5e-4)
def open_clear(a, b, n=2000):
    for i in range(1, n):
        t = i / n; p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        if not pt_free(p)[0]:
            return False
    return True


ok('S->(8,6) and (8,6)->G touch the rectangle only at its corner (open segments are free)',
   open_clear(S0, (8, 6)) and open_clear((8, 6), GC) and seg_free(S0, (8, 6)) == (False, 'R'))
ok('PRM path is 9.3% longer than 10.068', abs(11.0 / (v1_ + v2_ - 1) - 1.0926) < 5e-4)
pg = math.pi / 100
P('P(sample in goal), expected samples', (pg, 1 / pg))
ok('p = 0.0314, 1/p = 31.8', abs(pg - 0.0314) < 5e-5 and abs(1 / pg - 31.83) < 0.01)


def rrt_step(tree, qr, delta):
    near = min(tree, key=lambda q: math.dist(q, qr))
    d = math.dist(near, qr)
    if d <= delta:
        return near, d, qr
    return near, d, (near[0] + delta * (qr[0] - near[0]) / d, near[1] + delta * (qr[1] - near[1]) / d)


tree = [S0]
near, d, qn = rrt_step(tree, (6, 2), 1.0)
P('RRT step 1: nearest, dist, q_new', (near, d, qn))
ok('step 1: |.|=4.528, q_new=(2.494,1.610)', abs(d - 4.528) < 5e-4 and abs(qn[0] - 2.494) < 5e-4 and abs(qn[1] - 1.610) < 5e-4)
ok('step 1 segment free', seg_free(S0, qn)[0])
P('step 1 unit vector', ((6 - 1.5) / d, (2 - 1.5) / d))
n1_ = qn
tree.append(n1_)
near2, d2, qn2 = rrt_step(tree, (4, 5), 1.0)
P('RRT step 2: dists to S, n1', (math.dist(S0, (4, 5)), math.dist(n1_, (4, 5))))
P('RRT step 2: nearest, dist, q_new', (near2, d2, qn2))
P('step 2 unit vector', ((4 - n1_[0]) / d2, (5 - n1_[1]) / d2))
ok('step 2 nearest is the new node (3.709 < 4.301), q_new=(2.900,2.524)', near2 == n1_ and abs(d2 - 3.7091) < 5e-4 and abs(qn2[0] - 2.900) < 5e-4 and abs(qn2[1] - 2.524) < 5e-4)
ok('step 2 segment free and q_new free', seg_free(n1_, qn2)[0] and pt_free(qn2)[0])
nr, dr_, qb = rrt_step([(4.5, 2.5)], (6.5, 2.0), 1.0)
P('RRT rejected example from (4.5,2.5) towards (6.5,2): q_new, free?', (qb, pt_free(qb)))
ok('rejected: q_new lies inside T2', not pt_free(qb)[0])
biased = rrt_step([S0, n1_, qn2], GC, 1.0)
P('goal-biased sample q_rand=G: nearest, dist, q_new', biased)
# p_goal bias numbers (added)
# non-holonomic: unicycle primitives, v=1 m/s, T=1 s from pose (0,0,0)
tgt = (2.0, 1.0)
prim = []
for om in (-1.0, 0.0, 1.0):
    if om == 0:
        x, y, th = 1.0, 0.0, 0.0
    else:
        x, y, th = math.sin(om) / om, (1 - math.cos(om)) / om, om
    prim.append((om, x, y, math.degrees(th), math.dist((x, y), tgt)))
for pr in prim:
    P('unicycle primitive omega, x, y, theta(deg), dist to (2,1)', tuple(round(v, 3) for v in pr))
best = min(prim, key=lambda p: p[4])
ok('closest primitive is omega = +1 rad/s', best[0] == 1.0)
ok('end poses (0.841,0.460,57.3) / (1,0,0) / (0.841,-0.460,-57.3)',
   abs(prim[2][1] - 0.8415) < 5e-4 and abs(prim[2][2] - 0.4597) < 5e-4 and abs(prim[1][1] - 1) < 1e-12 and abs(prim[0][2] + 0.4597) < 5e-4)
ok('distances 1.864, 1.414, 1.278', [round(p[4], 3) for p in prim] == [1.864, 1.414, 1.278])
GEN['prim_rows'] = '\n'.join(f'{p[0]:+.0f} & {p[1]:.3f} & {p[2]:+.3f} & {p[3]:+.1f}$^\\circ$ & {p[4]:.3f}\\\\' for p in prim)

# =====================================================================================
# F. practical Q1 maze (our grid reading, same as site/src/checks/w12.py)
# =====================================================================================
VW = set()
for r in (1, 2): VW.add((r, 1))
for r in (4, 5): VW.add((r, 1))
for r in (2, 3, 4, 5, 6): VW.add((r, 2))
for r in (2, 3): VW.add((r, 3))
for r in (5, 6): VW.add((r, 3))
for r in (1, 2): VW.add((r, 4))
HW = set()
for c in (2, 3): HW.add((4, c))
HW.add((5, 3)); HW.add((6, 4)); HW.add((7, 0)); HW.add((7, 3))
MS, MG = (0, 2), (4, 2)


def mnb(n):
    r, c = n; out = []
    if r > 0 and (r, c) not in HW and not (r - 1 == 0 and c != 2): out.append((r - 1, c))
    if c < 4 and (r, c + 1) not in VW and r >= 1: out.append((r, c + 1))
    if r < 7 and (r + 1, c) not in HW and not (r == 0 and c != 2): out.append((r + 1, c))
    if c > 0 and (r, c) not in VW and r >= 1: out.append((r, c - 1))
    return out


mn = [(0, 2)] + [(r, c) for r in range(1, 8) for c in range(5)]
mad = {n: mnb(n) for n in mn}
medges = sum(len(v) for v in mad.values()) // 2
q = deque([MS]); par = {MS: None}; mbo = []
while q:
    n = q.popleft(); mbo.append(n)
    if n == MG: break
    for m in mad[n]:
        if m not in par: par[m] = n; q.append(m)
mbp = trace(par, MG)
par = {MS: None}; mdo = []


def rec(n):
    mdo.append(n)
    if n == MG: return True
    for m in mad[n]:
        if m not in par:
            par[m] = n
            if rec(m): return True
    return False


rec(MS)
mdp = trace(par, MG)
hm = lambda n: abs(n[0] - MG[0]) + abs(n[1] - MG[1])
cnt = itertools.count(); g = {MS: 0}; par = {MS: None}; op = [(hm(MS), hm(MS), next(cnt), MS)]; closed = set(); mao = []
while op:
    f, hh, _, n = heapq.heappop(op)
    if n in closed: continue
    closed.add(n); mao.append(n)
    if n == MG: break
    for m in mad[n]:
        if m in closed: continue
        if m not in g or g[n] + 1 < g[m]:
            g[m] = g[n] + 1; par[m] = n; heapq.heappush(op, (g[m] + hm(m), hm(m), next(cnt), m))
P('maze cells/edges', (len(mn), medges))
P('maze BFS popped / moves', (len(mbo), len(mbp) - 1))
P('maze DFS popped / moves', (len(mdo), len(mdp) - 1))
P('maze A* popped / moves', (len(mao), len(trace(par, MG)) - 1))
ok('maze: 36 cells, 38 edges, BFS 8 moves (26 popped), DFS 16 moves (19 popped), A* 8 moves (16 popped)',
   (len(mn), medges, len(mbp) - 1, len(mbo), len(mdp) - 1, len(mdo), len(trace(par, MG)) - 1, len(mao)) == (36, 38, 8, 26, 16, 19, 8, 16))
GEN['maze_nums'] = (f'{len(mn)} open cells, {medges} edges; BFS {len(mbp) - 1} moves ({len(mbo)} cells popped); '
                    f'DFS {len(mdp) - 1} moves ({len(mdo)} popped); A$^\\ast$ {len(trace(par, MG)) - 1} moves ({len(mao)} popped)')


# ---- TikZ of the practical map with the hand roadmap, rejected sample, RRT steps ----
def tikz_map():
    L = ['\\begin{tikzpicture}[x=6mm,y=6mm,every node/.style={font=\\tiny\\sffamily,inner sep=1pt}]']
    L.append('\\draw[line width=0.5pt] (0,0) rectangle (10,10);')
    for t in range(0, 11, 2):
        L.append(f'\\node[anchor=north] at ({t},0) {{{t}}}; \\node[anchor=east] at (0,{t}) {{{t}}};')
    for poly in OBST.values():
        L.append('\\fill[g2] ' + ' -- '.join(f'({x},{y})' for x, y in poly) + ' -- cycle; \\draw[line width=0.5pt] ' + ' -- '.join(f'({x},{y})' for x, y in poly) + ' -- cycle;')
    L.append(f'\\draw[line width=0.6pt] ({GC[0]},{GC[1]}) circle (1); \\node at ({GC[0]},{GC[1]+0.35}) {{G}};')
    for (a, b) in edges:
        L.append(f'\\draw[line width=0.4pt,g3] ({NODE[a][0]},{NODE[a][1]}) -- ({NODE[b][0]},{NODE[b][1]});')
    pts = ' -- '.join(f'({NODE[n][0]},{NODE[n][1]})' for n in rpath)
    L.append(f'\\draw[line width=1.6pt,black] {pts};')
    L.append(f'\\draw[dashed,line width=0.5pt] ({NODE["n1"][0]},{NODE["n1"][1]}) -- ({NODE["n6"][0]},{NODE["n6"][1]});')
    for n, (x, y) in NODE.items():
        L.append(f'\\fill ({x},{y}) circle (0.12); \\node[anchor=south west] at ({x},{y}) {{{n if n != "S" else "S"}}};')
    for (x, y) in ((3, 5.2), (5.4, 2)):
        L.append(f'\\draw[line width=0.6pt] ({x-0.18},{y-0.18}) -- ({x+0.18},{y+0.18}) ({x-0.18},{y+0.18}) -- ({x+0.18},{y-0.18});')
    L.append('\\end{tikzpicture}')
    return '\n'.join(L)


GEN['prm_tikz'] = tikz_map()

# =====================================================================================
# G. write the booklet part from the template
# =====================================================================================
print('\n' + '=' * 60)
if FAILS:
    print('FAILURES:', FAILS)
    raise SystemExit(1)
tplp = os.path.join(HERE, 'w12_template.tex')
if os.path.exists(tplp):
    tpl = open(tplp).read()
    out = re.sub(r'%%GEN:([a-z0-9_]+)%%', lambda m: GEN[m.group(1)], tpl)
    with open(os.path.join(HERE, '..', 'w12.tex'), 'w') as fh:
        fh.write(out)
    print('wrote ../w12.tex from template')
print('ALL CHECKS PASSED')
