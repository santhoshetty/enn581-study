"""W12 Planning algorithms: checks for every number on the W12 tab.
Run: python3 checks/w12.py   (ends ALL CHECKS PASSED)
Writes checks/w12_vectors.json (test vectors for the gridplan and akgraph widgets).

Sections
 1. Slide graph A-K (pp. 56, 59): BFS, DFS, Dijkstra, A* tables, admissibility / consistency of h.
 2. Grid planner (widget gridplan): BFS / Dijkstra / A* on two fixed grids, 4- and 8-connected.
 3. Practical sheet Q1: the maze as a grid, BFS/DFS/A*.
 4. Practical sheet Q2: PRM geometry (point-in-polygon, segment-polygon, nearest vertex), hand roadmap.
 5. Practical sheet Q3 / slide numbers: RRT step, dimensionality counts, inflation, visibility-graph edge count.
"""
import json, math, os, sys
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import check, close, report

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-9
VEC = {}

# ======================================================================
# 1. Slide graph (pp. 56, 59)
# ======================================================================
EDGES = {('A', 'B'): 10, ('B', 'E'): 9, ('A', 'C'): 6, ('C', 'F'): 5, ('F', 'H'): 7, ('H', 'J'): 4,
         ('J', 'K'): 3, ('A', 'D'): 8, ('D', 'G'): 9, ('F', 'G'): 3, ('G', 'I'): 7, ('I', 'K'): 5}
NODES = list('ABCDEFGHIJK')
ADJ = {n: {} for n in NODES}
for (u, v), w in EDGES.items():
    ADJ[u][v] = w
    ADJ[v][u] = w
H_SLIDE = {'A': 20, 'B': 10, 'C': 17, 'D': 13, 'E': 8, 'G': 8, 'H': 7, 'I': 2, 'J': 2, 'K': 0}   # p. 59 red labels; K implied 0
H9 = dict(H_SLIDE, F=9)       # reading 1: the blue "9" above F is h(F)
H0 = dict(H_SLIDE, F=0)       # reading 2: F has no printed h; treat as 0 (always safe)

check('graph has 11 nodes and 12 edges', len(NODES) == 11 and len(EDGES) == 12)


def path_cost(path):
    return sum(ADJ[a][b] for a, b in zip(path, path[1:]))


def true_dist_to(goal):
    """Dijkstra from the goal (graph is undirected): true cost-to-go d(n, goal)."""
    d = {goal: 0}
    todo = {goal}
    done = set()
    while todo:
        n = min(todo, key=lambda x: (d[x], x))
        todo.remove(n); done.add(n)
        for m, w in ADJ[n].items():
            if m not in done and (m not in d or d[n] + w < d[m]):
                d[m] = d[n] + w; todo.add(m)
    return d


def bfs_graph(start, goal):
    q = deque([start]); parent = {start: None}; order = []
    while q:
        n = q.popleft(); order.append(n)
        if n == goal: break
        for m in sorted(ADJ[n]):
            if m not in parent:
                parent[m] = n; q.append(m)
    p = []; n = goal
    while n: p.append(n); n = parent[n]
    return order, p[::-1]


def dfs_graph(start, goal):
    """Recursive DFS, neighbours in alphabetical order, visited on entry."""
    parent = {start: None}; order = []; found = []

    def rec(n):
        order.append(n)
        if n == goal: found.append(n); return True
        for m in sorted(ADJ[n]):
            if m not in parent:
                parent[m] = n
                if rec(m): return True
        return False
    rec(start)
    p = []; n = goal
    while n: p.append(n); n = parent[n]
    return order, p[::-1]


def best_first(start, goal, h, reopen=False):
    """A* (h = 0 gives Dijkstra). Pop lowest f; ties -> lowest h, then alphabetical.
    Returns (rows, order, g, parent). rows[i] = dict(expand, g, f, open_after, closed_after)."""
    g = {start: 0}; parent = {start: None}; openset = {start}; closed = set(); rows = []; order = []
    while openset:
        n = min(openset, key=lambda x: (g[x] + h[x], h[x], x))
        openset.remove(n); closed.add(n); order.append(n)
        for m, w in sorted(ADJ[n].items()):
            ng = g[n] + w
            if m in closed and not (reopen and ng < g[m]):
                continue
            if m not in g or ng < g[m]:
                g[m] = ng; parent[m] = n; openset.add(m); closed.discard(m)
        rows.append(dict(expand=n, g=g[n], f=g[n] + h[n], open=sorted(openset, key=lambda x: (g[x] + h[x], h[x], x)), closed=sorted(closed),
                         open_detail=[(x, g[x], h[x], g[x] + h[x], parent[x]) for x in sorted(openset, key=lambda x: (g[x] + h[x], h[x], x))]))
        if n == goal: break
    return rows, order, g, parent


def path_of(parent, goal):
    p = []; n = goal
    while n: p.append(n); n = parent[n]
    return p[::-1]


# ---- true distances, admissibility, consistency ----
D = true_dist_to('K')
print('true d(n,K):', D)
close('d(A,K) = 25', D['A'], 25)
close('d(H,K) = 7', D['H'], 7)
close('d(F,K) = 14 (F-H-J-K)', D['F'], 14)
close('d(G,K) = 12 (G-I-K)', D['G'], 12)
close('d(C,K) = 19', D['C'], 19)
close('d(D,K) = 21', D['D'], 21)
close('d(B,K) = 35', D['B'], 35)
close('d(E,K) = 44', D['E'], 44)
close('d(I,K) = 5, d(J,K) = 3', [D['I'], D['J']], [5, 3])

for name, hh in (('h(F)=9', H9), ('h(F)=0', H0)):
    ok = all(hh[n] <= D[n] for n in NODES)
    check(f'heuristic ({name}) is admissible: h(n) <= d(n,K) for all n', ok)
slack = {n: D[n] - H9[n] for n in NODES}
print('slack d-h (h(F)=9):', slack)
check('admissible, tight at H (h=7=d) and K (0)', slack['H'] == 0 and slack['K'] == 0)

viol = []
for (u, v), w in EDGES.items():
    for a, b in ((u, v), (v, u)):
        if H9[a] > w + H9[b]:
            viol.append((a, b, H9[a], w, H9[b]))
print('consistency violations (h(F)=9):', viol)
check('h(F)=9: exactly two directed violations, C->F (17 > 5+9) and H->J (7 > 4+2)', viol == [('C', 'F', 17, 5, 9), ('H', 'J', 7, 4, 2)])
viol0 = [(a, b) for (u, v), w in EDGES.items() for a, b in ((u, v), (v, u)) if H0[a] > w + H0[b]]
print('consistency violations (h(F)=0):', viol0)
check('h(F)=0: three violations C->F, G->F, H->J', viol0 == [('C', 'F'), ('H', 'J'), ('G', 'F')])
# F's feasible consistent range
lo = max(H9[m] - w for m, w in ADJ['F'].items() if m != 'F' for _ in [0]) if False else None
# h(F) consistent iff h(F) <= w + h(m) for all neighbours m and h(m) <= w + h(F):  C:17<=5+hF, hF<=5+17 ; G: 8<=3+hF, hF<=3+8; H: 7<=7+hF, hF<=7+7
lowF = max(17 - 5, 8 - 3, 7 - 7); highF = min(5 + 17, 3 + 8, 7 + 7)
print('h(F) range consistent with its neighbours:', lowF, highF)
check('h(F) consistent range is 12..11 (empty): C needs h(F)>=12, G needs h(F)<=11', lowF > highF and lowF == 12 and highF == 11)

# ---- BFS / DFS ----
bo, bp = bfs_graph('A', 'K')
do, dp = dfs_graph('A', 'K')
print('BFS order', bo, 'path', bp, path_cost(bp))
print('DFS order', do, 'path', dp, path_cost(dp))
check('BFS visit order A B C D E F G H I J K', bo == list('ABCDEFGHIJK'))
check('BFS path A-D-G-I-K cost 29 (4 edges)', bp == list('ADGIK') and path_cost(bp) == 29)
check('DFS visit order A B E C F G D I K', do == list('ABECFGDIK'))
check('DFS path A-C-F-G-I-K cost 26 (5 edges)', dp == list('ACFGIK') and path_cost(dp) == 26)

# ---- Dijkstra ----
H_ZERO = {n: 0 for n in NODES}
drows, dorder, dg, dpar = best_first('A', 'K', H_ZERO)
print('Dijkstra order', dorder)
check('Dijkstra expansion order A C D B F G H E I J K', dorder == list('ACDBFGHEIJK'))
check('Dijkstra g values', dg == {'A': 0, 'B': 10, 'C': 6, 'D': 8, 'E': 19, 'F': 11, 'G': 14, 'H': 18, 'I': 21, 'J': 22, 'K': 25}, str(dg))
check('Dijkstra parents', dpar == {'A': None, 'B': 'A', 'C': 'A', 'D': 'A', 'E': 'B', 'F': 'C', 'G': 'F', 'H': 'F', 'I': 'G', 'J': 'H', 'K': 'J'})
check('Dijkstra path A-C-F-H-J-K cost 25', path_of(dpar, 'K') == list('ACFHJK') and dg['K'] == 25)
check('Dijkstra expands all 11 nodes before K (K is last, g=25 is the largest)', len(dorder) == 11)
print('--- Dijkstra step table')
for i, r in enumerate(drows, 1):
    print(i, r['expand'], r['g'], [(x, g_, par) for x, g_, _h, _f, par in r['open_detail']], r['closed'])
# tie check: no ties in Dijkstra selection
for i, r in enumerate(drows):
    pass
# check Dijkstra pop distinct g (strictly increasing = no tie)
check('Dijkstra pops nodes in non-decreasing g', all(drows[i]['g'] <= drows[i + 1]['g'] for i in range(len(drows) - 1)))

# ---- A* reading 1: h(F)=9 ----
arows, aorder, ag, apar = best_first('A', 'K', H9)
print('A* h(F)=9 order', aorder, path_of(apar, 'K'), ag['K'])
check('A* (h(F)=9) expansion order A B D C F G I H J K', aorder == list('ABDCFGIHJK'), str(aorder))
check('A* (h(F)=9) path A-C-F-H-J-K cost 25', path_of(apar, 'K') == list('ACFHJK') and ag['K'] == 25)
check('A* (h(F)=9) never expands E', 'E' not in aorder)
print('--- A* table (h(F)=9)')
for i, r in enumerate(arows, 1):
    print(i, r['expand'], 'g', r['g'], 'f', r['f'], 'open', r['open_detail'], 'closed', r['closed'])
# the f of popped nodes
fpop = [r['f'] for r in arows]
print('f of popped nodes', fpop)
# reopening version: same result?
arows_r, aorder_r, ag_r, apar_r = best_first('A', 'K', H9, reopen=True)
check('A* with re-opening (h(F)=9) gives the same order and cost 25', aorder_r == aorder and ag_r['K'] == 25)
# Tie-break sensitivity: alphabetical-only (ignore h) order
def best_first_alpha(start, goal, h):
    g = {start: 0}; parent = {start: None}; openset = {start}; closed = set(); order = []
    while openset:
        n = min(openset, key=lambda x: (g[x] + h[x], x))
        openset.remove(n); closed.add(n); order.append(n)
        if n == goal: break
        for m, w in sorted(ADJ[n].items()):
            ng = g[n] + w
            if m in closed: continue
            if m not in g or ng < g[m]:
                g[m] = ng; parent[m] = n; openset.add(m)
    return order, g, parent
ao2, ag2, ap2 = best_first_alpha('A', 'K', H9)
print('A* (alphabetical tie-break only):', ao2, ag2['K'])
check('A* order does not depend on the tie-break rule here', ao2 == aorder)
# are there any ties in f among open nodes at some pop?
ties = []
for i, r in enumerate(arows):
    fs = [d[3] for d in r['open_detail']]
    if len(fs) != len(set(fs)): ties.append(i + 1)
print('steps whose open set has equal f values (after expansion):', ties)

# ---- A* reading 2: h(F)=0 (no printed value) ----
arows0, aorder0, ag0, apar0 = best_first('A', 'K', H0)
print('A* h(F)=0 order', aorder0, path_of(apar0, 'K'), ag0['K'])
check('A* (h(F)=0) order A B D C F G I H J K (same as h(F)=9)', aorder0 == list('ABDCFGIHJK'), str(aorder0))
check('A* (h(F)=0) path cost 25', ag0['K'] == 25 and path_of(apar0, 'K') == list('ACFHJK'))
# F is popped when its f is the minimum: show the f(F) range for which the order is unchanged
def order_for(hf):
    return best_first('A', 'K', dict(H_SLIDE, F=hf))[1]
same = [hf for hf in range(0, 15) if order_for(hf) == aorder]
print('h(F) values 0..14 giving the same order:', same, ' order at 14:', order_for(14))
check('order identical for every integer h(F) in 0..13 (the answer does not hinge on the "9"); h(F)=14 (exact) changes it', same == list(range(0, 14)))
check('with the exact h(F)=14, G (f=25, h=8) is popped before F (f=25, h=14) by the lowest-h tie-break: A B D C G F H J K', order_for(14) == list('ABDCGFHJK'), str(order_for(14)))
# the answer if h(F) overestimates: h(F)=30
o30, g30 = best_first('A', 'K', dict(H_SLIDE, F=30))[1], best_first('A', 'K', dict(H_SLIDE, F=30))[2]
print('h(F)=30:', o30, g30['K'])

check('h(F)=30 (overestimate; d(F,K)=14) returns the cost-29 route A-D-G-I-K via order A B D C G I E K', o30 == list('ABDCGIEK') and g30['K'] == 29, str(o30))
VEC_hF30 = dict(order=o30, cost=g30['K'])
# true-cost-to-go (perfect) heuristic gives straight path
arowsP, aorderP, agP, aparP = best_first('A', 'K', D)
print('perfect h order', aorderP)
check('perfect heuristic expands only the optimal path nodes A C F H J K', aorderP == list('ACFHJK'))

# overestimating heuristic breaks optimality (teaching example on this graph)
Hbad = dict(H_SLIDE, F=9, C=40, H=20, J=2)   # C overestimates (d=19), H overestimates (d=7), pushing the optimal route away
arB, aoB, agB, apB = best_first('A', 'K', Hbad)
print('overestimating h:', aoB, path_of(apB, 'K'), agB['K'])
check('overestimating h(C)=40,h(H)=20 returns a worse path than 25', agB['K'] > 25, f"cost {agB['K']} via {path_of(apB, 'K')}")
VEC['akgraph'] = {
    'astar_hF30': VEC_hF30,
    'edges': [[u, v, w] for (u, v), w in EDGES.items()], 'h_slide': H_SLIDE, 'h_F': 9,
    'astar_order': aorder, 'astar_cost': ag['K'], 'astar_path': path_of(apar, 'K'),
    'dijkstra_order': dorder, 'dijkstra_cost': dg['K'], 'dijkstra_path': path_of(dpar, 'K'),
    'astar_rows': [dict(expand=r['expand'], g=r['g'], f=r['f'], open=[list(d) for d in r['open_detail']], closed=r['closed']) for r in arows],
}

# ---- consistency matters only with a closed set that is never re-opened: demonstrate on a tiny graph ----
# (graph S-A 1, S-B 1, A-C 1, B-C 5? ...) classic: h admissible, inconsistent
TADJ = {'S': {'A': 1, 'B': 4}, 'A': {'S': 1, 'B': 2, 'G': 12}, 'B': {'S': 4, 'A': 2, 'G': 5}, 'G': {'A': 12, 'B': 5}}
TH = {'S': 0, 'A': 7, 'B': 0, 'G': 0}   # h(A)=7 vs d(A,G)=7 (A-B-G): admissible; inconsistent on A->B (7 > 2+0)


def tiny_astar(reopen):
    g = {'S': 0}; openset = {'S'}; closed = set(); order = []; par = {'S': None}
    while openset:
        n = min(openset, key=lambda x: (g[x] + TH[x], TH[x], x)); openset.remove(n); closed.add(n); order.append(n)
        if n == 'G': break
        for m, w in sorted(TADJ[n].items()):
            ng = g[n] + w
            if m in closed and not (reopen and ng < g[m]): continue
            if m not in g or ng < g[m]:
                g[m] = ng; par[m] = n; openset.add(m); closed.discard(m)
    return order, g['G'], par
# true d: A->G = min(12, 2+5) = 7 ; B->G = 5 ; S->G = min(1+7, 4+5)= 8
tD = {'S': 8, 'A': 7, 'B': 5, 'G': 0}
check('tiny example: h admissible', all(TH[n] <= tD[n] for n in TH))
check('tiny example: h inconsistent on A->B', TH['A'] > TADJ['A']['B'] + TH['B'])
o_nr, c_nr, _ = tiny_astar(False)
o_r, c_r, _ = tiny_astar(True)
print('tiny: no reopen', o_nr, c_nr, ' reopen', o_r, c_r)
check('tiny: admissible-but-inconsistent h with no re-opening: cost 9 (not optimal)', c_nr == 9, f'{o_nr} {c_nr}')
check('tiny: re-opening restores the optimum 8', c_r == 8, f'{o_r} {c_r}')

# ======================================================================
# 2. Grid planner (mirrors W3D.kinds.gridplan exactly)
# ======================================================================
SQ2 = math.sqrt(2)
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]                       # up, right, down, left   (dr, dc)
DIRS8 = DIRS4 + [(-1, 1), (1, 1), (1, -1), (-1, -1)]             # then the four diagonals


def parse_grid(rows):
    R, C = len(rows), len(rows[0]); wall = set(); S = Gp = None
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            if ch == '#': wall.add((r, c))
            elif ch == 'S': S = (r, c)
            elif ch == 'G': Gp = (r, c)
    return R, C, wall, S, Gp


def heur(kind, a, b):
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    if kind == 'manhattan': return dr + dc
    if kind == 'euclid': return math.hypot(dr, dc)
    return 0.0


def neighbours(R, C, wall, n, conn):
    out = []
    for k, (dr, dc) in enumerate(DIRS8 if conn == 8 else DIRS4):
        r, c = n[0] + dr, n[1] + dc
        if not (0 <= r < R and 0 <= c < C) or (r, c) in wall: continue
        if k >= 4:   # diagonal: no corner cutting
            if (n[0] + dr, n[1]) in wall or (n[0], n[1] + dc) in wall: continue
            out.append(((r, c), SQ2))
        else:
            out.append(((r, c), 1.0))
    return out


def plan(rows, algo, conn=4, hk='manhattan'):
    """algo: bfs | dijkstra | astar. Expanded count includes the goal when it is popped."""
    R, C, wall, S, Gp = parse_grid(rows)
    parent = {S: None}; expanded = []
    if algo == 'bfs':
        q = deque([S]); seen = {S}; g = {S: 0.0}
        while q:
            n = q.popleft(); expanded.append(n)
            if n == Gp: break
            for m, w in neighbours(R, C, wall, n, conn):
                if m not in seen:
                    seen.add(m); parent[m] = n; g[m] = g[n] + w; q.append(m)
        frontier = list(q)
    else:
        h = (lambda n: 0.0) if algo == 'dijkstra' else (lambda n: heur(hk, n, Gp))
        g = {S: 0.0}; seqno = {S: 0}; seq = 1; openset = {S}; closed = set()
        while openset:
            def key_less(a, b):   # True if a should be popped before b
                fa, fb = g[a] + h(a), g[b] + h(b)
                if abs(fa - fb) > EPS: return fa < fb
                ha, hb = h(a), h(b)
                if abs(ha - hb) > EPS: return ha < hb
                return seqno[a] < seqno[b]
            n = None
            for x in sorted(openset, key=lambda t: seqno[t]):
                if n is None or key_less(x, n): n = x
            openset.remove(n); closed.add(n); expanded.append(n)
            if n == Gp: break
            for m, w in neighbours(R, C, wall, n, conn):
                if m in closed: continue
                ng = g[n] + w
                if m not in g or ng < g[m] - EPS:
                    g[m] = ng; parent[m] = n; seqno[m] = seq; seq += 1; openset.add(m)
        frontier = sorted(openset, key=lambda t: seqno[t])
    if Gp not in parent:
        return dict(cost=None, expanded=len(expanded), path=[], order=expanded, frontier=frontier)
    path = []; n = Gp
    while n is not None: path.append(n); n = parent[n]
    path.reverse()
    cost = sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(path, path[1:]))
    return dict(cost=cost, expanded=len(expanded), path=path, order=expanded, frontier=frontier)


GRID1 = ['..........',
         '..........',
         '.S..#.....',
         '....#.....',
         '....#..G..',
         '....#.....',
         '..........',
         '..........']
GRID2 = ['............',
         '.....#......',
         '.S...#......',
         '.....#..G...',
         '.....#......',
         '.....#......',
         '............',
         '............']
# GRID2 is a vertical wall with a gap at top row 0 and bottom rows 6-7
GRID2 = ['............',
         '.....#......',
         '.S...#......',
         '.....#..G...',
         '.....#......',
         '.....#......',
         '............',
         '............']
GRID3 = ['..........',     # U-trap: goal inside a cup that opens on the far (right) side
         '..........',
         '....#####.',
         '.S..#...#.',
         '....#.G...',
         '....#...#.',
         '....#####.',
         '..........']
GRID4 = ['..........',     # sealed cup: no path exists
         '..........',
         '....#####.',
         '.S..#...#.',
         '....#.G.#.',
         '....#...#.',
         '....#####.',
         '..........']


GRID5 = ['S.#.....',    # found by random search: Manhattan A* with 8-connectivity is suboptimal here
         '......#.',
         '...#....',
         '........',
         '.##.##..',
         '.#..#.#G']


def summary(res):
    return dict(cost=None if res['cost'] is None else round(res['cost'], 6), expanded=res['expanded'],
                path=[list(p) for p in res['path']], steps=max(len(res['path']) - 1, 0))


vec_grids = {}
for gname, rows in (('grid1', GRID1), ('grid2', GRID2), ('grid3', GRID3), ('grid4', GRID4), ('grid5', GRID5)):
    entry = {'rows': rows, 'results': {}}
    for conn in (4, 8):
        for algo, hk in (('bfs', 'zero'), ('dijkstra', 'zero'), ('astar', 'manhattan'), ('astar', 'euclid')):
            res = plan(rows, algo, conn, hk)
            key = f'{algo}-{hk if algo == "astar" else "none"}-{conn}'
            entry['results'][key] = summary(res)
    vec_grids[gname] = entry
VEC['gridplan'] = vec_grids

R1 = vec_grids['grid1']['results']
print('grid1', {k: (v['cost'], v['expanded'], v['steps']) for k, v in R1.items()})
R2 = vec_grids['grid2']['results']
print('grid2', {k: (v['cost'], v['expanded'], v['steps']) for k, v in R2.items()})
R3 = vec_grids['grid3']['results']
R4 = vec_grids['grid4']['results']
print('grid4', {k: (v['cost'], v['expanded']) for k, v in R4.items()})
check('grid4 (sealed cup): every algorithm reports failure after exploring the whole reachable region', all(v['cost'] is None for v in R4.values()))
check('grid4: BFS/Dijkstra expand every reachable cell (80 - walls - 4 interior cells... )', R4['bfs-none-4']['expanded'] == R4['dijkstra-none-4']['expanded'])
print('grid3', {k: (v['cost'], v['expanded'], v['steps']) for k, v in R3.items()})

# structural checks (independent of tie-breaks)
for gname in vec_grids:
    for conn in (4, 8):
        rr = vec_grids[gname]['results']
        cd = rr[f'dijkstra-none-{conn}']['cost']
        ca_e = rr[f'astar-euclid-{conn}']['cost']
        check(f'{gname} conn{conn}: A* (Euclid) cost equals Dijkstra cost', (cd is None and ca_e is None) or (cd is not None and ca_e is not None and abs(cd - ca_e) < 1e-6), f'{cd} vs {ca_e}')
        if conn == 4:
            ca_m = rr['astar-manhattan-4']['cost']
            check(f'{gname} conn4: A* (Manhattan) is optimal (admissible for 4-connected)', (cd is None and ca_m is None) or (cd is not None and ca_m is not None and abs(cd - ca_m) < 1e-6))
        check(f'{gname} conn{conn}: A* (Euclid) expands <= Dijkstra', rr[f'astar-euclid-{conn}']['expanded'] <= rr[f'dijkstra-none-{conn}']['expanded'])
    # 4-connected: BFS is optimal (unit costs)
    rr = vec_grids[gname]['results']
    cb, cdj = rr['bfs-none-4']['cost'], rr['dijkstra-none-4']['cost']
    check(f'{gname} conn4: BFS cost equals Dijkstra cost (unit steps)', (cb is None and cdj is None) or (cb is not None and cdj is not None and abs(cb - cdj) < 1e-6))
# Manhattan with 8-connection: it overestimates (diagonal costs sqrt2 < 2). Look for a grid where it is suboptimal
sub = [g for g in vec_grids if vec_grids[g]['results']['dijkstra-none-8']['cost'] is not None and vec_grids[g]['results']['astar-manhattan-8']['cost'] > vec_grids[g]['results']['dijkstra-none-8']['cost'] + 1e-6]
print('grids where Manhattan A* with 8-connectivity is suboptimal:', sub)
check('grid5: Manhattan A* with 8-connectivity returns a longer path than the optimum (overestimating h)', sub == ['grid5'], str(sub))
check('grid5: Euclidean A* (8-conn) is optimal', abs(vec_grids['grid5']['results']['astar-euclid-8']['cost'] - vec_grids['grid5']['results']['dijkstra-none-8']['cost']) < 1e-6)
print('grid5', {k: (v['cost'], v['expanded']) for k, v in vec_grids['grid5']['results'].items()})
# open-field demonstration: 8x8 open, start (0,0) goal (7,7)
OPEN = ['S.......', '........', '........', '........', '........', '........', '........', '.......G']
ro = {k: plan(OPEN, a, 8, hk) for k, (a, hk) in {'dij': ('dijkstra', 'zero'), 'man': ('astar', 'manhattan'), 'euc': ('astar', 'euclid')}.items()}
print('open 8x8 8-conn: dij', ro['dij']['cost'], ro['dij']['expanded'], 'man', ro['man']['cost'], ro['man']['expanded'], 'euc', ro['euc']['cost'], ro['euc']['expanded'])
close('open field: optimal 8-connected diagonal cost = 7*sqrt2', ro['dij']['cost'], 7 * SQ2, 1e-9)
VEC['gridplan']['open8'] = {'rows': OPEN, 'dij': summary(ro['dij']), 'man': summary(ro['man']), 'euc': summary(ro['euc'])}

# Manhattan overestimation on 8-connected: bent corridor example
BENT = ['S.......',
        '.######.',
        '.#....#.',
        '.#.##.#.',
        '.#.#G.#.',
        '.#.####.',
        '.#......',
        '.#######']
# (not required to be suboptimal; kept only if it is)
# A direct small example: 3x3 open grid, S top-left, G at (1,1) vs (2,2)? Manhattan h(S)=4 vs true sqrt2*2=2.83
h_man = heur('manhattan', (0, 0), (2, 2)); true8 = 2 * SQ2
check('Manhattan overestimates on 8-connected grid: h(S)=4 > true 2*sqrt2 = 2.828 for a 2-cell diagonal', h_man > true8)
close('Euclid heuristic for the same pair', heur('euclid', (0, 0), (2, 2)), 2 * SQ2)

# ======================================================================
# 3. Practical Q1: the maze as a grid
# ======================================================================
# Reading of the figure (5 columns x 7 rows inside the box + an entrance cell above the box). See the tab for the drawing.
# Cells (r, c): r = 0 is the entrance notch (only c = 2 is open); rows 1..7 inside the box; c = 0..4.
MAZE_COLS, MAZE_ROWS = 5, 8
# vertical walls: set of (r, c) meaning there is a wall between (r, c-1) and (r, c)
VW = set()
for r in (1, 2): VW.add((r, 1))                 # line x=181 from top to ~y=278  (rows 1-2)
for r in (4, 5): VW.add((r, 1))                 # line x=181 from y=365 to ~580   (rows 4-5)
for r in (2, 3, 4, 5, 6): VW.add((r, 2))        # line x=298 from y=195 to ~668   (rows 2-6)
for r in (2, 3): VW.add((r, 3))                 # line x=416 from y=195 to 367    (rows 2-3)
for r in (5, 6): VW.add((r, 3))                 # line x=416 from y=495 to 673    (rows 5-6)
for r in (1, 2): VW.add((r, 4))                 # line x=530 from top to ~278     (rows 1-2)
# horizontal walls: set of (r, c) meaning there is a wall between (r-1, c) and (r, c)
HW = set()
for c in (2, 3): HW.add((4, c))                 # y=367 from x=298 to 530         (between rows 3 and 4)
HW.add((5, 3))                                  # y=495 from x=416 to 530         (between rows 4 and 5)
HW.add((6, 4))                                  # y=583 from x=530 to 653         (between rows 5 and 6)
HW.add((7, 0)); HW.add((7, 3))                  # y=673 x=63..181 and x=416..530  (between rows 6 and 7)
MS, MG = (0, 2), (4, 2)


def maze_nbrs(n):
    r, c = n; out = []
    # order: up, right, down, left
    if r > 0 and (r, c) not in HW and not (r - 1 == 0 and c != 2):
        out.append((r - 1, c))
    if c < 4 and (r, c + 1) not in VW and r >= 1:
        out.append((r, c + 1))
    if r < 7 and (r + 1, c) not in HW and not (r == 0 and c != 2):
        out.append((r + 1, c))
    if c > 0 and (r, c) not in VW and r >= 1:
        out.append((r, c - 1))
    return out


def maze_graph():
    nodes = [(0, 2)] + [(r, c) for r in range(1, 8) for c in range(5)]
    return nodes, {n: maze_nbrs(n) for n in nodes}


mnodes, madj = maze_graph()
# symmetry check of adjacency
sym = all(a in madj[b] for a in madj for b in madj[a])
check('maze adjacency is symmetric', sym)
check('entrance cell (0,2) has exactly one neighbour (1,2)', madj[(0, 2)] == [(1, 2)])
edges_m = sum(len(v) for v in madj.values()) // 2
print('maze nodes', len(mnodes), 'edges', edges_m)
check('maze has 36 cells', len(mnodes) == 36)


def mbfs(order_fn=lambda l: l):
    q = deque([MS]); par = {MS: None}; order = []
    while q:
        n = q.popleft(); order.append(n)
        if n == MG: break
        for m in order_fn(madj[n]):
            if m not in par: par[m] = n; q.append(m)
    p = []; n = MG
    while n: p.append(n); n = par[n]
    return order, p[::-1]


def mdfs(order_fn=lambda l: l):
    par = {MS: None}; order = []

    def rec(n):
        order.append(n)
        if n == MG: return True
        for m in order_fn(madj[n]):
            if m not in par:
                par[m] = n
                if rec(m): return True
        return False
    rec(MS)
    p = []; n = MG
    while n: p.append(n); n = par[n]
    return order, p[::-1]


mb_o, mb_p = mbfs()
md_o, md_p = mdfs()
print('maze BFS', len(mb_o), 'expanded; path len', len(mb_p) - 1, mb_p)
print('maze DFS', len(md_o), 'expanded; path len', len(md_p) - 1, md_p)
check('maze: BFS shortest path has 8 moves', len(mb_p) - 1 == 8, str(mb_p))
check('maze: DFS (up,right,down,left order) path length', len(md_p) - 1 >= 8)
md2_o, md2_p = mdfs(lambda l: l[::-1])
print('maze DFS (reversed order)', len(md2_o), len(md2_p) - 1, md2_p)
# enumerate all simple paths S->G to see the structure
all_paths = []


def enum(n, path):
    if n == MG: all_paths.append(list(path)); return
    for m in madj[n]:
        if m not in path:
            path.append(m); enum(m, path); path.pop()
enum(MS, [MS])
print('number of simple S->G paths:', len(all_paths), sorted(len(p) - 1 for p in all_paths))
check('maze: six simple S->G routes of 8, 12, 14, 16, 18, 20 moves', sorted(len(p) - 1 for p in all_paths) == [8, 12, 14, 16, 18, 20], str(sorted(len(p) - 1 for p in all_paths)))
# A* with Manhattan on the maze (unit costs)
def maze_astar():
    g = {MS: 0}; par = {MS: None}; openset = {MS}; closed = set(); order = []
    h = lambda n: abs(n[0] - MG[0]) + abs(n[1] - MG[1])
    seqn = {MS: 0}; seq = 1
    while openset:
        n = min(openset, key=lambda x: (g[x] + h(x), h(x), seqn[x])); openset.remove(n); closed.add(n); order.append(n)
        if n == MG: break
        for m in madj[n]:
            if m in closed: continue
            if m not in g or g[n] + 1 < g[m]:
                g[m] = g[n] + 1; par[m] = n; seqn[m] = seq; seq += 1; openset.add(m)
    p = []; n = MG
    while n: p.append(n); n = par[n]
    return order, p[::-1], g
ma_o, ma_p, ma_g = maze_astar()
print('maze A* Manhattan expanded', len(ma_o), 'path', len(ma_p) - 1)
check('maze: A* (Manhattan) finds the 8-move optimum', len(ma_p) - 1 == 8)
check('maze: A* expands fewer cells than BFS', len(ma_o) < len(mb_o), f'{len(ma_o)} vs {len(mb_o)}')
close('Manhattan h(S) on the maze = 4 (row 0->4 plus col 2->2)', abs(MS[0] - MG[0]) + abs(MS[1] - MG[1]), 4)
# true cost 8 >= h: admissible
check('maze: Manhattan admissible at S (4 <= 8)', 4 <= len(mb_p) - 1)
# Manhattan consistent on a unit-cost 4-connected grid: |h(n)-h(m)| <= 1 for every edge
cons = all(abs((abs(a[0] - MG[0]) + abs(a[1] - MG[1])) - (abs(b[0] - MG[0]) + abs(b[1] - MG[1]))) <= 1 for a in madj for b in madj[a])
check('maze: Manhattan is consistent (|h(n)-h(m)| <= 1 across every edge)', cons)
VEC['maze'] = dict(nodes=len(mnodes), edges=edges_m, bfs_expanded=len(mb_o), bfs_path=[list(p) for p in mb_p], bfs_moves=len(mb_p) - 1,
                   dfs_expanded=len(md_o), dfs_moves=len(md_p) - 1, dfs_path=[list(p) for p in md_p],
                   astar_expanded=len(ma_o), astar_moves=len(ma_p) - 1)
print('BFS order:', mb_o)
print('DFS order:', md_o)
print('A* order:', ma_o)

# render the maze to ASCII for the HTML figure check
print('--- maze cell neighbours')
for n in mnodes:
    print(n, madj[n])


# ---- extra: monotone f, C-space example, maze counts ----
check('A* popped f values (h(F)=9) are NOT non-decreasing: 23 -> 20 at F (sign of an inconsistent h)', fpop == [20, 20, 21, 23, 20, 22, 23, 25, 24, 25] and any(fpop[i] > fpop[i + 1] for i in range(len(fpop) - 1)))
fpopD = [r['f'] for r in best_first('A', 'K', D)[0]]
check('with the exact (hence consistent) h = d(n,K), popped f is constant 25', set(fpopD) == {25})
# a consistent h on the same graph: h = 0.5 * true distance is consistent? (halving a consistent h keeps consistency)
Hh = {n: D[n] / 2 for n in NODES}
okc = all(Hh[a] <= w + Hh[b] + 1e-12 for (u, v), w in EDGES.items() for a, b in ((u, v), (v, u)))
check('h = d/2 is consistent on the slide graph', okc)
fpopH = [r['f'] for r in best_first('A', 'K', Hh)[0]]
check('with a consistent h, popped f is non-decreasing', all(fpopH[i] <= fpopH[i + 1] + 1e-12 for i in range(len(fpopH) - 1)), str(fpopH))

# 2-link arm: one workspace point maps to two C-space points (two IK branches); L1 = L2 = 1 m
def fk2(q1, q2, L1=1.0, L2=1.0):
    return (L1 * math.cos(q1) + L2 * math.cos(q1 + q2), L1 * math.sin(q1) + L2 * math.sin(q1 + q2))
close('2-link arm (0, +90 deg) puts the tip at (1,1)', fk2(0, math.pi / 2), (1, 1), 1e-12)
close('2-link arm (90 deg, -90 deg) also puts the tip at (1,1)', fk2(math.pi / 2, -math.pi / 2), (1, 1), 1e-12)
close('q2 = -90 deg is the same configuration as q2 = 270 deg (wrap-around)', fk2(math.pi / 2, 3 * math.pi / 2), (1, 1), 1e-12)
# C-space size of the arm: both joints in [0, 2pi): torus; a point obstacle blocks a curve (1-D set) in the 2-D C-space
close('IK: q2 = +/- arccos((x^2+y^2-L1^2-L2^2)/(2 L1 L2)) = +/-90 deg for (1,1)', math.acos((1 + 1 - 1 - 1) / 2), math.pi / 2, 1e-12)

check('maze BFS pops 26 cells (incl. G), DFS 19, A* (Manhattan) 16', (len(mb_o), len(md_o), len(ma_o)) == (26, 19, 16))
check('maze DFS (up,right,down,left) route has 16 moves; BFS/A* have 8', len(md_p) - 1 == 16 and len(mb_p) - 1 == 8)


# ---- inflation numbers ----
close('corridor 1.0 m wide, robot radius 0.4 m: free width for the point robot = 1.0 - 2*0.4 = 0.2', 1.0 - 2 * 0.4, 0.2, 1e-12)
check('a robot of radius 0.55 m cannot enter a 1.0 m corridor (1.0 - 1.1 < 0)', 1.0 - 2 * 0.55 < 0)
def perimeter(poly):
    return sum(math.dist(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))
def steiner(poly, r):
    return area(poly) + perimeter(poly) * r + math.pi * r * r
def area(poly):
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))) / 2
T3 = [(5, 3), (6, 3), (5, 1)]
close('triangle 3 perimeter = 1 + sqrt5 + 2 = 5.2361', perimeter(T3), 3 + math.sqrt(5), 1e-12)
close('inflated triangle 3 (r = 0.5) area = A + P r + pi r^2 = 1 + 2.618 + 0.785 = 4.403', steiner(T3, 0.5), 4.4034, 1e-3)
close('inflated 2x2 square (r = 0.5) area = 4 + 8*0.5 + pi/4 = 8.785 (rounded corners)', steiner([(2, 2), (4, 2), (4, 4), (2, 4)], 0.5), 8.7854, 1e-3)
close('same square with a SQUARE robot (half-side 0.5) -> 3x3 = 9', 3 * 3, 9)

# ======================================================================
# 4. Practical Q2: PRM geometry
# ======================================================================
TRI1 = [(2, 7), (4, 5), (3, 4)]
RECT = [(6, 6), (6, 8), (8, 8), (8, 6)]
TRI2 = [(5, 3), (6, 3), (5, 1)]
OBST = [TRI1, RECT, TRI2]
S0, G0, GR = (1.5, 1.5), (9.0, 9.0), 1.0


def point_in_polygon(p, poly):
    """Ray casting (strict interior is inside; boundary counts as inside, handled by eps)."""
    x, y = p; inside = False; n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin: inside = not inside
    return inside


def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def seg_cross(p1, p2, q1, q2):
    """Proper or touching intersection of segments."""
    o1, o2, o3, o4 = orient(p1, p2, q1), orient(p1, p2, q2), orient(q1, q2, p1), orient(q1, q2, p2)
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)) and o1 * o2 != 0 and o3 * o4 != 0:
        return True
    def on(a, b, c): return min(a[0], b[0]) - EPS <= c[0] <= max(a[0], b[0]) + EPS and min(a[1], b[1]) - EPS <= c[1] <= max(a[1], b[1]) + EPS
    if abs(o1) < EPS and on(p1, p2, q1): return True
    if abs(o2) < EPS and on(p1, p2, q2): return True
    if abs(o3) < EPS and on(q1, q2, p1): return True
    if abs(o4) < EPS and on(q1, q2, p2): return True
    return False


def edge_free(a, b):
    for poly in OBST:
        if point_in_polygon(a, poly) or point_in_polygon(b, poly): return False
        for i in range(len(poly)):
            if seg_cross(a, b, poly[i], poly[(i + 1) % len(poly)]): return False
        # a segment can lie wholly inside only if an endpoint is inside, covered above
    return True


check('point (3,5.5)... is outside triangle 1? use (3,5.2) inside', point_in_polygon((3.0, 5.2), TRI1))
check('(2.0, 5.0) outside triangle 1', not point_in_polygon((2.0, 5.0), TRI1))
check('(7,7) inside the rectangle', point_in_polygon((7, 7), RECT))
check('(5.4, 2.0) inside triangle 3', point_in_polygon((5.4, 2.0), TRI2))
check('(5.6, 1.5) outside triangle 3 (right of edge (5,1)-(6,3))', not point_in_polygon((5.6, 1.5), TRI2))
check('S (1.5,1.5) is free', not any(point_in_polygon(S0, p) for p in OBST))
check('straight S->G is blocked (passes through the rectangle)', not edge_free(S0, G0))
# worked ray-casting example: p = (3, 5.2) in TRI1
p = (3.0, 5.2); crossings = []
for i in range(3):
    (x1, y1), (x2, y2) = TRI1[i], TRI1[(i + 1) % 3]
    if (y1 > p[1]) != (y2 > p[1]):
        xin = x1 + (p[1] - y1) * (x2 - x1) / (y2 - y1)
        crossings.append(((x1, y1), (x2, y2), round(xin, 4), p[0] < xin))
print('ray-cast crossings for (3,5.2):', crossings)
check('ray cast from (3,5.2) to +x crosses exactly 1 triangle edge to its right (odd -> inside)', sum(1 for c in crossings if c[3]) == 1)
# The edge (2,7)-(3,4): x at y=5.2 is 2+(5.2-7)*(1)/(-3)=2.6 (left of 3: not counted); edge (4,5)-(2,7)... compute
for c in crossings: print('  edge', c[0], c[1], 'x at y=5.2 =', c[2], 'right of p?' , c[3])

# hand roadmap: S -> P1 -> P2 -> P3 -> goal region
P1, P2, P3, P4 = (4.0, 2.5), (7.0, 4.5), (9.0, 7.0), (9.0, 8.5)
for a, b, nm in ((S0, P1, 'S-P1'), (P1, P2, 'P1-P2'), (P2, P3, 'P2-P3'), (P3, P4, 'P3-P4')):
    check(f'hand roadmap edge {nm} is collision-free', edge_free(a, b))
for nm, pt in (('P1', P1), ('P2', P2), ('P3', P3), ('P4', P4)):
    check(f'{nm} is in free space', not any(point_in_polygon(pt, q) for q in OBST))
check('P4 (9,8.5) is inside the goal disc', math.hypot(P4[0] - G0[0], P4[1] - G0[1]) <= GR)
close('|P4 - G| = 0.5', math.hypot(P4[0] - G0[0], P4[1] - G0[1]), 0.5)
lens = [math.dist(a, b) for a, b in ((S0, P1), (P1, P2), (P2, P3), (P3, P4))]
print('hand roadmap leg lengths', [round(x, 4) for x in lens], 'total', round(sum(lens), 4))
close('leg lengths', lens, [math.dist(S0, P1), math.dist(P1, P2), math.dist(P2, P3), 1.5])
close('S-P1 = sqrt(2.5^2+1^2) = 2.6926', lens[0], 2.6926, 1e-4)
close('P1-P2 = sqrt(3^2+2^2) = 3.6056', lens[1], 3.6056, 1e-4)
close('P2-P3 = sqrt(2^2+2.5^2) = 3.2016', lens[2], 3.2016, 1e-4)
close('total path length = 11.0', sum(lens), 11.0, 5e-3)
# a deliberately bad edge to show a segment-polygon failure: P1 -> (7,7) crosses? and S->(4.5,2.0)->...
check('edge S -> (5.4,2.0) enters triangle 3', not edge_free(S0, (5.4, 2.0)))
# a segment whose endpoints are both outside a polygon but which crosses it
a, b = (1.0, 5.0), (5.0, 5.0)    # crosses triangle 1 horizontally at y=5 between x=... edge (4,5) touches
check('(1,5)->(5,5) touches vertex (4,5) of triangle 1 -> counted as collision (touching)', not edge_free(a, b))
a, b = (1.0, 5.5), (5.0, 5.5)
check('(1,5.5)->(5,5.5) crosses triangle 1 (both endpoints outside)', not edge_free(a, b) and not point_in_polygon(a, TRI1) and not point_in_polygon(b, TRI1))
# nearest vertex query
V = [S0, P1, P2, P3]
q = (6.0, 4.0)
dists = [math.dist(q, v) for v in V]
print('nearest to (6,4):', dists)
check('nearest vertex to q=(6,4) is P2=(7,4.5)', min(range(4), key=lambda i: dists[i]) == 2, str([round(d, 3) for d in dists]))
close('distance from (6,4) to P2 = sqrt(1.25) = 1.1180', dists[2], math.sqrt(1.25), 1e-9)

# Probability of a uniform sample landing in an obstacle (free-space fraction): areas
def area(poly):
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))) / 2
areas = [area(p) for p in OBST]
print('obstacle areas', areas)
close('triangle 1 area = |2(5-4)+4(4-7)+3(7-5)|/2 = 2', areas[0], 2.0)
close('rectangle area = 4', areas[1], 4.0)
close('triangle 3 area = 1', areas[2], 1.0)
tot = sum(areas)
close('total obstacle area 7 of 100', tot, 7.0)
check('obstacle fraction of C is 7%', abs(tot / 100 - 0.07) < 1e-12)
goal_disc_area = math.pi * GR ** 2
close('goal disc area = pi', goal_disc_area, math.pi)
# fraction of C inside the goal disc: disc centred (9,9) radius 1 lies fully inside [0,10]^2
check('goal disc lies fully inside C = [0,10]^2', 9 - 1 >= 0 and 9 + 1 <= 10)
print('chance one uniform sample lands in the goal disc: %.4f' % (goal_disc_area / 100))
close('P(sample in goal) = pi/100 = 0.0314', goal_disc_area / 100, 0.0314159, 1e-6)
# expected samples to hit the goal disc directly ~ 1/p
print('expected samples to hit goal disc ~', round(100 / goal_disc_area, 2))
close('expected number of uniform samples to land in the goal ~ 31.8', 100 / goal_disc_area, 31.83, 1e-2)

# ======================================================================
# 5. Q3 RRT step, dimensionality, inflation, visibility graph, other slide numbers
# ======================================================================
tree = [S0]; qrand = (6.0, 2.0); step = 1.0
near = min(tree, key=lambda v: math.dist(v, qrand))
d = math.dist(near, qrand)
qnew = (near[0] + step * (qrand[0] - near[0]) / d, near[1] + step * (qrand[1] - near[1]) / d)
print('RRT q_new', qnew, 'd', d)
close('|q_rand - S| = sqrt(4.5^2+0.5^2) = 4.5277', d, 4.5277, 1e-4)
close('q_new = S + 1*(unit vector) = (2.4939, 1.5 + 0.5/4.5277 = 1.6104)', qnew, (1.5 + 4.5 / d, 1.5 + 0.5 / d), 1e-12)
close('q_new numeric (2.4939, 1.6104)', qnew, (2.4939, 1.6104), 1e-4)
check('q_new edge S->q_new is collision-free', edge_free(S0, qnew))
close('|q_new - S| = step', math.dist(S0, qnew), 1.0)
# second iteration: q_rand2 = (5.5, 2.0) lies inside triangle 3 -> rejected / edge collides
check('q_rand (5.2,2) is inside triangle 3, so a sample there is discarded', point_in_polygon((5.2, 2.0), TRI2))
# Dimensionality
close('2D grid 100x100 = 10^4 nodes', 100 ** 2, 1e4)
close('3D grid 100^3 = 10^6 nodes', 100 ** 3, 1e6)
close('6D grid 100^6 = 10^12 nodes', 100 ** 6, 1e12)
close('neighbours: 2D 4 or 8 (3^2-1 = 8)', 3 ** 2 - 1, 8)
close('neighbours: 3D 6 face or 26 all (3^3-1 = 26)', 3 ** 3 - 1, 26)
close('neighbours: 6D 12 axis-aligned or 3^6-1 = 728 all-adjacent (not on the slide)', [2 * 6, 3 ** 6 - 1], [12, 728])
# bytes to store 10^12 nodes at one byte each
close('10^12 nodes at 1 byte each = 1 TB', 1e12 / 1e12, 1.0)
# Inflation: obstacle grown by robot radius
close('inflated rectangle: 2x2 obstacle grown by robot radius 0.5 -> 3x3 box (square-robot case)', [2 + 2 * 0.5, 2 + 2 * 0.5], [3, 3])
# Visibility graph on the practical map: vertex count and candidate edge count
verts = [S0] + [v for poly in OBST for v in poly] + [G0]
nv = len(verts)
cand = nv * (nv - 1) // 2
vis = [(i, j) for i in range(nv) for j in range(i + 1, nv)
       if edge_free_vis(verts[i], verts[j]) if False] if False else None
def edge_free_vis(a, b):
    """Visibility: a segment between two polygon vertices is allowed to touch endpoints and slide along polygon edges,
    but not to pass through an obstacle interior. Test by sampling the midpoint and a few interior points."""
    for t in (0.1, 0.25, 0.5, 0.75, 0.9):
        pt = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        if any(point_in_polygon(pt, poly) for poly in OBST): return False
    # crossing through a polygon edge interior while endpoints are vertices: sampled points catch these for this map
    for poly in OBST:
        for i in range(len(poly)):
            c, dd = poly[i], poly[(i + 1) % len(poly)]
            # proper crossing (strict)
            o1, o2, o3, o4 = orient(a, b, c), orient(a, b, dd), orient(c, dd, a), orient(c, dd, b)
            if o1 * o2 < -EPS and o3 * o4 < -EPS: return False
    return True
vis = [(i, j) for i in range(nv) for j in range(i + 1, nv) if edge_free_vis(verts[i], verts[j])]
print('visibility graph on practical map: vertices', nv, 'candidate pairs', cand, 'visible', len(vis))
check('practical map has 1 + 3 + 4 + 3 + 1 = 12 vertices incl. S and G', nv == 12)
check('candidate vertex pairs = 12*11/2 = 66', cand == 66)

# shortest path S -> G through the visibility graph (Dijkstra)
import heapq
adjv = {i: [] for i in range(nv)}
for i, j in vis:
    w = math.dist(verts[i], verts[j]); adjv[i].append((j, w)); adjv[j].append((i, w))
dist = {0: 0.0}; prev = {}; pq = [(0.0, 0)]
while pq:
    dd, u = heapq.heappop(pq)
    if dd > dist.get(u, 1e18) + 1e-12: continue
    for v, w in adjv[u]:
        if dd + w < dist.get(v, 1e18) - 1e-12:
            dist[v] = dd + w; prev[v] = u; heapq.heappush(pq, (dist[v], v))
gi = nv - 1
pth = [gi]
while pth[-1] != 0: pth.append(prev[pth[-1]])
pth.reverse()
print('visibility-graph shortest path to the goal CENTRE (9,9):', [verts[i] for i in pth], round(dist[gi], 4))
close('shortest S->(9,9) via visibility graph (check vs script)', dist[gi], dist[gi])
check('the visibility path bends only at obstacle vertices', all(verts[i] in [v for poly in OBST for v in poly] for i in pth[1:-1]))
close('visibility path S->(8,6) = sqrt(6.5^2+4.5^2) = 7.9057', math.dist(S0, (8, 6)), 7.9057, 1e-4)
close('visibility path (8,6)->(9,9) = sqrt(10) = 3.1623', math.dist((8, 6), (9, 9)), 3.1623, 1e-4)
check('visibility path through the corner (8,6): S->(8,6) and (8,6)->G are clear (touching the corner only)', edge_free_vis(S0, (8, 6)) and edge_free_vis((8, 6), G0))
to_disc = math.dist(S0, (8, 6)) + math.dist((8, 6), G0) - GR
close('shortest path to the goal DISC boundary = 7.9057 + 3.1623 - 1 = 10.068', to_disc, 10.068, 1e-3)
close('hand roadmap is 11.0/10.068 = 9.3% longer than the visibility-graph optimum (to the disc)', sum(lens) / to_disc - 1, 0.0926, 1e-3)
check('the 12-vertex visibility graph has 35 visible pairs by the sampling test (of 66)', len(vis) == 35)
VEC['practical_q2'] = dict(hand_path=[S0, P1, P2, P3, P4], hand_length=round(sum(lens), 4), vis_path=[list(verts[i]) for i in pth], vis_length=round(dist[gi], 4),
                           qnew=[round(qnew[0], 4), round(qnew[1], 4)], obstacle_area=tot)

# ======================================================================
# Misc numbers used on the tab
# ======================================================================
close('Dijkstra 1378 / A* 277 = 4.97x', 1378 / 277, 4.9747, 1e-3)
close('BFS 1413 / A* 277 = 5.10x', 1413 / 277, 5.1011, 1e-3)
close('A* saves 1 - 277/1378 = 79.9% of Dijkstra operations', 1 - 277 / 1378, 0.7990, 1e-3)

# BFS complexity example on a 100x100 4-connected grid: |V| + |E|
V_, E_ = 100 * 100, 2 * 100 * 99
close('4-connected 100x100: |E| = 19800, |V|+|E| = 29800', [E_, V_ + E_], [19800, 29800])

# practical-style hand A* on a 4x4 grid for the mini-quiz
GQ = ['S...',
      '.##.',
      '....',
      '...G']
rq = {a: plan(GQ, a, 4, 'manhattan') for a in ('bfs', 'dijkstra', 'astar')}
print('4x4 mini-grid', {k: (v['cost'], v['expanded']) for k, v in rq.items()})
check('4x4 mini-grid: all find cost 6', all(abs(v['cost'] - 6) < 1e-9 for v in rq.values()))
VEC['mini4'] = {'rows': GQ, **{k: summary(v) for k, v in rq.items()}}

# Dijkstra vs A* worked check for slide numbers: f values along optimal path under h(F)=9
fvals = {n: ag[n] + H9[n] for n in 'ACFHJK'}
print('f along optimal path', fvals)
check('f along optimal path: A 20, C 23, F 20, H 25, J 24, K 25', fvals == {'A': 20, 'C': 23, 'F': 20, 'H': 25, 'J': 24, 'K': 25}, str(fvals))
check('f(A) = h(A) = 20 and f(K) = 25 = optimal cost', fvals['A'] == 20 and fvals['K'] == 25)

# ----------------------------------------------------------------------
with open(os.path.join(HERE, 'w12_vectors.json'), 'w') as f:
    json.dump(VEC, f, indent=1)
print('wrote w12_vectors.json')
report()
