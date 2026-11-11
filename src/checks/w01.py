"""W1 mechanisms: verifies every number on the W1 tab and writes checks/w01_vectors.json for the mobility widget.
Run: python3 site/src/checks/w01.py"""
import json, os, sys, itertools
sys.path.insert(0, os.path.dirname(__file__))
from common import *

HERE = os.path.dirname(os.path.abspath(__file__))
DOF = dict(R=1, P=1, H=1, C=2, U=2, S=3)       # slide p.34
out = {}

# ---------------------------------------------------------------- G-K, both forms
def gk_form2(D, N, F, RC=0, RM=0):
    """M = D(N-J-1) + sum F_i + R_C - R_M   (slide p.41; extended terms W13 p.16)"""
    J = len(F)
    return D * (N - J - 1) + sum(F) + RC - RM

def gk_form1(D, N, F):
    """First form (slide p.40): M = D(N-1) - sum (D - F_i): same thing counted joint-type by joint-type."""
    if D == 3:
        P = {i: sum(1 for f in F if f == i) for i in (1, 2)}
        return 3 * (N - 1) - 2 * P[1] - 1 * P[2]
    P = {i: sum(1 for f in F if f == i) for i in range(1, 6)}
    return 6 * (N - 1) - 5 * P[1] - 4 * P[2] - 3 * P[3] - 2 * P[4] - 1 * P[5]

def counts_to_F(counts):
    F = []
    for t, n in counts.items():
        F += [DOF[t]] * n
    return F

def gk_check(name, D, N, counts, want, RC=0, RM=0, both_forms=True):
    F = counts_to_F(counts)
    m2 = gk_form2(D, N, F, RC, RM)
    check(f'{name}: M = {want} (form 2: D(N-J-1)+sumF{"+RC-RM" if (RC or RM) else ""}; J={len(F)}, sumF={sum(F)})', m2 == want, f'got {m2}')
    if both_forms and not (RC or RM):
        m1 = gk_form1(D, N, F)
        check(f'{name}: first form agrees', m1 == m2, f'form1={m1} form2={m2}')
    return m2

# --- DOF of single bodies (pp. 28-33, nb Q1)
check('rigid body in 2D = 3 DoF (2 translations + 1 rotation)', 2 + 1 == 3)
check('rigid body in 3D = 6 DoF', 3 + 3 == 6)
check('two unjoined planar links = 3+3 = 6 DoF; R joint removes 2 -> 4 (p.36)', 3 + 3 == 6 and 6 - (3 - 1) == 4)
check('constraints removed by joint = D - F: R in 2D removes 2, in 3D removes 5; S in 3D removes 3; U,C remove 4',
      (3 - 1, 6 - 1, 6 - 3, 6 - 2, 6 - 2) == (2, 5, 3, 4, 4))

# --- mobility presets (the widget uses these exact rows)
PRESETS = {}
def preset(key, title, D, N, counts, want, RC=0, RM=0, note=''):
    m = gk_check(title, D, N, counts, want, RC, RM)
    PRESETS[key] = dict(title=title, D=D, N=N, counts=counts, RC=RC, RM=RM, M=m, note=note)

preset('rpr', 'planar serial RPR (pp.42-43)', 3, 4, dict(R=2, P=1), 3)
preset('fourbar', 'planar four-bar (pp.44-45, nb Q2)', 3, 4, dict(R=4), 1)
preset('fivebar', 'planar five-bar', 3, 5, dict(R=5), 2)
preset('slidercrank', 'slider-crank', 3, 4, dict(R=3, P=1), 1)
preset('rpr3', 'planar 3-RPR parallel (pp.46-47, W13 Q6)', 3, 8, dict(R=6, P=3), 3)
preset('gkfail', 'three equal parallel legs (p.49)', 3, 5, dict(R=6), 0)
preset('delta', 'delta, plain G-K (W13 Q7, p.15)', 6, 11, dict(R=3, S=12), 9)
preset('delta_ext', 'delta, extended with R_M = 6 (W13 p.17)', 6, 11, dict(R=3, S=12), 3, RC=0, RM=6)
preset('stewart', 'Stewart platform, 6 UPS legs (p.48)', 6, 14, dict(U=6, P=6, S=6), 6)
preset('ups_leg', 'one UPS leg only (p.48 drawing)', 6, 4, dict(U=1, P=1, S=1), 6)
preset('sps', 'six-SPS platform (added: why U matters)', 6, 14, dict(S=12, P=6), 12)
preset('serial6', 'serial 6R arm (3D)', 6, 7, dict(R=6), 6)

# explicit arithmetic quoted on the tab
check('RPR: 3(4-1) - 2x3 - 1x0 = 3', 3 * (4 - 1) - 2 * 3 - 1 * 0 == 3)
check('RPR form 2: 3(4-3-1) + (1+1+1) = 3', 3 * (4 - 3 - 1) + 3 == 3)
check('four-bar: 3(4-1) - 2x4 = 1; form 2: 3(4-4-1)+4 = 1', 3 * 3 - 8 == 1 and 3 * (4 - 4 - 1) + 4 == 1)
check('3-RPR: 3(8-1) - 2x9 = 3; form 2: 3(8-9-1)+9 = 3', 3 * 7 - 18 == 3 and 3 * (8 - 9 - 1) + 9 == 3)
check('3-RPR link count: ground + platform + 3 legs x 2 = 8; joints 3x3 = 9', 1 + 1 + 3 * 2 == 8 and 3 * 3 == 9)
check('p.49: 3(5-1) - 2x6 - 1x0 = 0', 3 * (5 - 1) - 2 * 6 - 0 == 0)
check('p.49: N = ground + 3 legs + platform = 5, J = 6', 1 + 3 + 1 == 5 and 3 * 2 == 6)
check('p.49 form 2: 3(5-6-1)+6 = 0', 3 * (5 - 6 - 1) + 6 == 0)
check('p.49 remove one middle leg: N=4, J=4 -> 3(4-1) - 2x4 = 1 (the slide: "it has 1 DoF")', 3 * 3 - 8 == 1)
check('Stewart: N = ground + platform + 6x2 = 14; J = 6x3 = 18; sumF = 6(2+1+3) = 36', 1 + 1 + 12 == 14 and 18 == 18 and 6 * (2 + 1 + 3) == 36)
check('Stewart: 6(14-18-1) + 36 = 6', 6 * (14 - 18 - 1) + 36 == 6)
check('Stewart wrong: forgetting ground (N=13) gives 6(13-18-1)+36 = 0', 6 * (13 - 18 - 1) + 36 == 0)
check('Stewart wrong: using D=3 gives 3(14-18-1)+36 = 21', 3 * (14 - 18 - 1) + 36 == 21)
check('one UPS leg: 6(4-3-1) + (2+1+3) = 6 (= sum of joint DoF, as for any serial chain)', 6 * (4 - 3 - 1) + 6 == 6)
check('six-SPS: 6(14-18-1) + 6x3 + 6x1 + 6x3 -> sumF with 12 S, 6 P = 42; M = 12 (6 real + 6 rod spins)', 6 * (14 - 18 - 1) + 12 * 3 + 6 == 12)
check('serial chain: N = J+1 so M = sum F_i (6R arm: 6)', 6 * (7 - 6 - 1) + 6 == 6)
check('delta: N = base + platform + 3 x (upper arm + 2 rods) = 11; J = 3 x (1 R + 4 S) = 15', 1 + 1 + 3 * 3 == 11 and 3 * 5 == 15)
check('delta: sumF = (1 + 4x3) x 3 = 39 and 6(11-15-1) + 39 = 9', (1 + 4 * 3) * 3 == 39 and 6 * (11 - 15 - 1) + 39 == 9)
check('delta extended: 9 + R_C(0) - R_M(6) = 3  (W13 p.17)', 9 + 0 - 6 == 3)
check('wrong D on four-bar: 6(4-4-1)+4 = -2 (negative: nonsense, planar needs D=3)', 6 * (4 - 4 - 1) + 4 == -2)
check('forgot ground on 3-RPR (N=7): 3(7-1) - 2x9 = 0', 3 * (7 - 1) - 2 * 9 == 0)
check('wrong coefficient on four-bar (using constraints = F): 3(4-1) - 1x4 = 5', 3 * 3 - 4 == 5)
# Hoberman (p.51): only "24 links and 36 revolute joints" is on the slide
check('Hoberman (added): D=3 -> 3(24-36-1)+36 = -3; D=6 -> 6(24-36-1)+36 = -42 (both negative, yet it moves)',
      3 * (24 - 36 - 1) + 36 == -3 and 6 * (24 - 36 - 1) + 36 == -42)

# ---------------------------------------------------------------- p.49 geometry: why G-K says 0 but the real DoF is 1
def jac_rank(g, L, s, theta, pose):
    """Closed-loop constraints of 3 legs (ground pivot (g_i,0), length L_i, angle th_i) and a platform with pivots s_i:
       G_i + L_i u(th_i) - (x,y) - R(phi) s_i = 0  (6 equations, 6 unknowns th1..3, x, y, phi)."""
    x, y, phi = pose
    c, s_ = np.cos(phi), np.sin(phi)
    J = np.zeros((6, 6)); res = np.zeros(6)
    for i in range(3):
        u = np.array([np.cos(theta[i]), np.sin(theta[i])])
        du = L[i] * np.array([-np.sin(theta[i]), np.cos(theta[i])])
        Rs = np.array([c * s[i][0] - s_ * s[i][1], s_ * s[i][0] + c * s[i][1]])
        dRs = np.array([-s_ * s[i][0] - c * s[i][1], c * s[i][0] - s_ * s[i][1]])
        res[2 * i:2 * i + 2] = np.array([g[i], 0]) + L[i] * u - np.array([x, y]) - Rs
        J[2 * i:2 * i + 2, i] = du
        J[2 * i:2 * i + 2, 3] = -np.array([1, 0]); J[2 * i:2 * i + 2, 4] = -np.array([0, 1])
        J[2 * i:2 * i + 2, 5] = -dRs
    return J, res

# parallelogram (equal parallel legs, platform pivot spacing = ground pivot spacing)
g = [0.0, 1.0, 2.0]; Lq = 1.2
s = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]       # platform pivots in platform frame, same spacing as the ground pivots
for th0 in (60 * deg, 75 * deg, 100 * deg):
    th = [th0] * 3
    pose = (g[0] + Lq * np.cos(th0) - s[0][0], Lq * np.sin(th0), 0.0)
    Jm, res = jac_rank(g, [Lq] * 3, s, th, pose)
    check(f'p.49 geometry, legs at {th0/deg:.0f} deg: closure residual = 0', np.allclose(res, 0, atol=1e-12))
    r_ = np.linalg.matrix_rank(Jm, tol=1e-9)
    check(f'p.49 geometry, legs at {th0/deg:.0f} deg: rank of the 6x6 constraint Jacobian = 5 -> 1 free motion (G-K assumed 6 -> 0)', r_ == 5, f'rank {r_}')
# finite motion (not only instantaneous): the whole family closes exactly
ok = True
for th0 in np.linspace(30 * deg, 150 * deg, 25):
    pose = (Lq * np.cos(th0), Lq * np.sin(th0), 0.0)
    ok &= np.allclose(jac_rank(g, [Lq] * 3, s, [th0] * 3, pose)[1], 0, atol=1e-12)
check('p.49 geometry: platform translates on a circle of radius L; all 25 sampled angles close exactly (finite DoF = 1)', ok)
# generic (unequal legs, platform skewed): lock-up
rng = np.random.default_rng(1)
ranks = []
for _ in range(20):
    th = rng.uniform(0.3, 2.8, 3); pose = (rng.uniform(-1, 1), rng.uniform(1, 2), rng.uniform(-0.3, 0.3))
    sg = [(0.0, 0.0), (1.0 + rng.uniform(-0.2, 0.2), rng.uniform(-0.1, 0.1)), (2.0 + rng.uniform(-0.2, 0.2), rng.uniform(-0.1, 0.1))]
    c_, s_ = np.cos(pose[2]), np.sin(pose[2])
    Ls = []
    for i in range(3):   # choose leg lengths so that this random pose closes: L_i u(th_i) = (x,y)+R s_i - G_i  needs consistent th_i -> instead set th from the vector
        v = np.array([pose[0], pose[1]]) + np.array([c_ * sg[i][0] - s_ * 0 - s_ * sg[i][1], s_ * sg[i][0] + c_ * sg[i][1]]) - np.array([g[i], 0])
        th[i] = np.arctan2(v[1], v[0]); Ls.append(np.hypot(*v))
    Jm, res = jac_rank(g, Ls, sg, th, pose)
    assert np.allclose(res, 0, atol=1e-9)
    ranks.append(np.linalg.matrix_rank(Jm, tol=1e-9))
check('p.49 generic geometry (unequal legs, 20 random assemblies): constraint Jacobian rank 6 -> 0 DoF, G-K is right', set(ranks) == {6}, f'ranks {sorted(set(ranks))}')

# ---------------------------------------------------------------- four-bar kinematics (widget)
def fourbar(d, a, b, c, th, mode=0):
    """Ground pivots O2=(0,0), O4=(d,0); crank a at angle th; coupler b; rocker c.
       B = intersection of circle(A,b) and circle(O4,c). mode 0 = '+' (B on the left of the A->O4 direction), 1 = '-'."""
    A = np.array([a * np.cos(th), a * np.sin(th)]); O4 = np.array([d, 0.0])
    v = O4 - A; rho = np.hypot(*v)
    if rho > b + c + 1e-12 or rho < abs(b - c) - 1e-12 or rho == 0:
        return dict(closed=False, A=A, B=None, rho=rho)
    x = (b * b - c * c + rho * rho) / (2 * rho)
    h = np.sqrt(max(b * b - x * x, 0.0))
    P = A + x * v / rho
    perp = np.array([-v[1], v[0]]) / rho
    B = P + (h if mode == 0 else -h) * perp
    return dict(closed=True, A=A, B=B, rho=rho)

def grashof(d, a, b, c):
    ls = sorted([d, a, b, c])
    return ls[0] + ls[3] <= ls[1] + ls[2] + 1e-12

def closing_range(d, a, b, c):
    """crank angles theta in (-pi, pi] for which the loop closes: (b-c)^2 <= d^2+a^2-2ad cos(th) <= (b+c)^2."""
    cmin = (d * d + a * a - (b + c) ** 2) / (2 * a * d)
    cmax = (d * d + a * a - (b - c) ** 2) / (2 * a * d)
    if cmin > 1 or cmax < -1: return None
    lo = np.arccos(cmax) if cmax < 1 else 0.0
    hi = np.arccos(cmin) if cmin > -1 else np.pi
    return (lo, hi)      # |theta| in [lo, hi]

D0, A0, B0, C0 = 4.0, 1.5, 3.5, 3.0
vec_fb = []
for th_d in (0, 30, 60, 90, 120, 180, 240, 300):
    for mode in (0, 1):
        r_ = fourbar(D0, A0, B0, C0, th_d * deg, mode)
        ok_ = r_['closed'] and abs(np.hypot(*(r_['B'] - r_['A'])) - B0) < 1e-12 and abs(np.hypot(r_['B'][0] - D0, r_['B'][1]) - C0) < 1e-12
        check(f'four-bar (d,a,b,c)=({D0},{A0},{B0},{C0}) theta={th_d} mode {mode}: closes and both link lengths exact', ok_)
        vec_fb.append(dict(d=D0, a=A0, b=B0, c=C0, th_deg=th_d, mode=mode, closed=bool(r_['closed']),
                           A=[round(float(r_['A'][0]), 6), round(float(r_['A'][1]), 6)],
                           B=[round(float(r_['B'][0]), 6), round(float(r_['B'][1]), 6)] if r_['closed'] else None))
r60 = fourbar(D0, A0, B0, C0, 60 * deg, 0)
print('  four-bar at 60 deg, mode 0: A =', np.round(r60['A'], 4), ' B =', np.round(r60['B'], 4))
check('default four-bar is Grashof (1.5 + 4.0 = 5.5 <= 3.5 + 3.0 = 6.5): crank turns fully', grashof(D0, A0, B0, C0))
check('default four-bar: closing range is the full circle', closing_range(D0, A0, B0, C0) == (0.0, np.pi))
check('mode 0 and mode 1 are mirror images about line A-O4 (same A, different B)',
      not np.allclose(fourbar(D0, A0, B0, C0, 60 * deg, 0)['B'], fourbar(D0, A0, B0, C0, 60 * deg, 1)['B']))
# non-Grashof example
dn, an, bn, cn = 4.0, 2.0, 2.5, 2.5
check('non-Grashof (d,a,b,c)=(4,2,2.5,2.5): 2 + 4 = 6 > 2.5 + 2.5 = 5', not grashof(dn, an, bn, cn))
rg = closing_range(dn, an, bn, cn)
print('  non-Grashof closing range |theta| <=', np.round(np.degrees(rg), 3))
check('non-Grashof: closes only for |theta| <= acos(-5/16) = 108.21 deg', abs(np.degrees(rg[1]) - np.degrees(np.arccos(-5 / 16))) < 1e-9 and rg[0] == 0.0)
# brute-force the closing range for both examples
for (d_, a_, b_, c_) in [(D0, A0, B0, C0), (dn, an, bn, cn), (4.0, 1.5, 1.0, 1.0), (3.0, 2.0, 1.0, 5.0), (5.0, 1.0, 1.5, 3.0)]:
    ths = np.radians(np.arange(-180, 180.0001, 0.25))
    cl = np.array([fourbar(d_, a_, b_, c_, t, 0)['closed'] for t in ths])
    rg_ = closing_range(d_, a_, b_, c_)
    if rg_ is None:
        pred = np.zeros_like(cl)
    else:
        pred = (np.abs(ths) >= rg_[0] - 1e-9) & (np.abs(ths) <= rg_[1] + 1e-9)
    check(f'closing-range formula vs sampling for (d,a,b,c)=({d_},{a_},{b_},{c_})', np.array_equal(pred, cl), f'range={None if rg_ is None else np.round(np.degrees(rg_), 2).tolist()}')
fail = fourbar(4.0, 1.5, 1.0, 1.0, 0.0, 0)
check('(4,1.5,1,1): |A O4| = 2.5 > b + c = 2 -> cannot close', not fail['closed'])
vec_fb.append(dict(d=4.0, a=1.5, b=1.0, c=1.0, th_deg=0, mode=0, closed=False, A=[1.5, 0.0], B=None))
vec_fb.append(dict(d=dn, a=an, b=bn, c=cn, th_deg=120, mode=0, closed=False, A=[round(an * np.cos(120 * deg), 6), round(an * np.sin(120 * deg), 6)], B=None))
check('non-Grashof at 120 deg > 108.21: cannot close', not fourbar(dn, an, bn, cn, 120 * deg, 0)['closed'])
r100 = fourbar(dn, an, bn, cn, 100 * deg, 0)
vec_fb.append(dict(d=dn, a=an, b=bn, c=cn, th_deg=100, mode=0, closed=bool(r100['closed']),
                   A=[round(float(r100['A'][0]), 6), round(float(r100['A'][1]), 6)],
                   B=[round(float(r100['B'][0]), 6), round(float(r100['B'][1]), 6)] if r100['closed'] else None))
check('non-Grashof at 100 deg closes', r100['closed'])
# the four-bar's mobility is 1 and a slider-crank is 1: both are in the presets above.

# ---------------------------------------------------------------- workspace examples (added)
def reach2R(L1, L2, r):  # point at distance r reachable by 2R (positions only)
    return abs(L1 - L2) - 1e-12 <= r <= L1 + L2 + 1e-12

check('planar 2R, L = 0.5, 0.3 m: reachable annulus 0.2 m <= r <= 0.8 m', abs(abs(0.5 - 0.3) - 0.2) < 1e-12 and abs(0.5 + 0.3 - 0.8) < 1e-12)
# planar 3R: tool at p with orientation phi reachable iff wrist w = p - L3 u(phi) is inside the 2R annulus
def reach3R(L, p, phi):
    w = np.array(p) - L[2] * np.array([np.cos(phi), np.sin(phi)])
    return reach2R(L[0], L[1], np.hypot(*w))
L3r = (0.3, 0.3, 0.3)
rs = np.linspace(0, 1.0, 201)
phis = np.radians(np.arange(0, 360, 2.0))
Wr = [r for r in rs if any(reach3R(L3r, (r, 0), ph) for ph in phis)]
Wd = [r for r in rs if all(reach3R(L3r, (r, 0), ph) for ph in phis)]
check('planar 3R, L = 0.3 each: reachable out to r = 0.9 m (sampled max)', abs(max(Wr) - 0.9) < 0.006 and min(Wr) == 0.0, f'{min(Wr)}..{max(Wr)}')
check('planar 3R, L = 0.3 each: dexterous workspace is the disc r <= 0.3 m (sampled max)', abs(max(Wd) - 0.3) < 0.006 and min(Wd) == 0.0, f'{min(Wd)}..{max(Wd)}')
check('W_D is a subset of W_R at every sampled radius', set(Wd) <= set(Wr))
check('W_D is a strict subset here (0.3 m disc inside 0.9 m disc): so "W_R subset-of W_D" and "W_R = W_D" are false', max(Wd) < max(Wr))
check('planar 3R L=(0.5,0.4,0.2): reachable out to L1+L2+L3 = 1.1 m', abs(0.5 + 0.4 + 0.2 - 1.1) < 1e-12)
# a planar 2R cannot set orientation independently, so it has no dexterous workspace in the 3-DoF pose sense
check('planar 2R has 2 DoF < 3 pose DoF, so no point has all orientations (added reasoning)', 2 < 3)
# actuation classification examples used on the tab
def classify(controllable, total):
    return 'fully actuated' if controllable == total else ('under-actuated' if controllable < total else 'redundant')
check('6R arm, 6 motors: fully actuated', classify(6, 6) == 'fully actuated')
check('6R arm, 4 motors: under-actuated', classify(4, 6) == 'under-actuated')
check('slide definition: 7R arm with 7 motors has controllable 7 = total 7 -> "fully actuated" by p.37 wording', classify(7, 7) == 'fully actuated')
check('same arm against the 6-DoF pose task: 7 > 6 -> redundant (standard meaning)', classify(7, 6) == 'redundant')
check('planar 3R positioning only (2 task DoF): 3 > 2 -> redundant (standard meaning)', classify(3, 2) == 'redundant')
check('Stewart platform with 6 actuators, M = 6: fully actuated; with 4 actuators under-actuated', classify(6, 6) == 'fully actuated' and classify(4, 6) == 'under-actuated')
check('delta with 3 actuators, M = 3 (extended): fully actuated', classify(3, 3) == 'fully actuated')
check('delta with 3 actuators vs plain G-K M = 9 would wrongly say under-actuated', classify(3, 9) == 'under-actuated')

# ---------------------------------------------------------------- vectors for the widget
out['mobility'] = []
for k, p in PRESETS.items():
    out['mobility'].append(dict(key=k, D=p['D'], N=p['N'], counts=p['counts'], RC=p['RC'], RM=p['RM'], J=sum(p['counts'].values()),
                                sumF=sum(counts_to_F(p['counts'])), M=p['M']))
# widget edge cases (verdict logic): M<0, M=0, M>0
out['verdict_rules'] = dict(negative='over-constrained', zero='structure', positive='mechanism')
out['fourbar'] = vec_fb
out['closing'] = [dict(d=dn, a=an, b=bn, c=cn, lo_deg=round(float(np.degrees(rg[0])), 4), hi_deg=round(float(np.degrees(rg[1])), 4)),
                  dict(d=D0, a=A0, b=B0, c=C0, lo_deg=0.0, hi_deg=180.0)]
out['grashof'] = [dict(d=D0, a=A0, b=B0, c=C0, grashof=True), dict(d=dn, a=an, b=bn, c=cn, grashof=False)]
with open(os.path.join(HERE, 'w01_vectors.json'), 'w') as f:
    json.dump(out, f, indent=1)
print('wrote w01_vectors.json')
report()
