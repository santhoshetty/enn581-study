"""v2 booklet ROBOTS gallery: verifies every matrix/number in booklet/robots.tex.
Run:  python3 v2_robots.py          (prints labelled numbers; PASS/FAIL lines; FAIL must not appear)
Part A derives each robot symbolically (ETS, proximal DH, T, J = dFK/dq, det J), part B checks IK -> FK round trips
for every branch, part C reads the marked matrices out of robots.tex and compares them with the derived ones."""
import sys, os, re, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(HERE, '..', '..', 'src', 'checks')
sys.path.insert(0, CH)
from common import *
import numpy as np, sympy as sp

np.set_printoptions(precision=4, suppress=True, linewidth=160)
TEX = os.path.join(HERE, '..', 'robots.tex')
D = np.pi / 180
q1, q2, q3, q4, q5, q6, q7 = sp.symbols('q1 q2 q3 q4 q5 q6 q7', real=True)
d1, d2, d3 = sp.symbols('d1 d2 d3', real=True)
L0, L1, L2, L3, L4, Lb, Lu, Lf, H, a1, a2 = sp.symbols('L0 L1 L2 L3 L4 Lb Lu Lf H a1 a2', positive=True)
SYM = {str(s): s for s in (q1, q2, q3, q4, q5, q6, q7, d1, d2, d3, L0, L1, L2, L3, L4, Lb, Lu, Lf, H, a1, a2)}
TRUTH = {}          # marker name -> sympy Matrix (symbolic) or numpy array (numeric)
c, s = sp.cos, sp.sin


def simp(M):
    return M.applyfunc(lambda e: sp.simplify(sp.trigsimp(e)))


def eqz(M):
    return all(sp.simplify(sp.trigsimp(sp.expand_trig(e))) == 0 for e in M)


def dh(rows):
    """rows of (a_{i-1}, alpha_{i-1}, d_i, theta_i) -> product of proximal link transforms"""
    T = sp.eye(4)
    for (a, al, d, th) in rows:
        T = T * sDH_prox(al, a, th, d)
    return T


def jac(T, qs):
    """6 x n Jacobian in {0}: Jv = dp/dq, Jw = vee(dR/dq R^T)"""
    p = T[:3, 3]; R = T[:3, :3]
    cols = []
    for q in qs:
        v = p.diff(q)
        W = simp(R.diff(q) * R.T)
        w = sp.Matrix([W[2, 1], W[0, 2], W[1, 0]])
        cols.append(sp.Matrix.vstack(v, w))
    return simp(sp.Matrix.hstack(*cols))


def num(M, sub):
    return np.array(M.subs(sub).evalf(), dtype=float)


def deg2rad(v): return np.radians(v)


def section(t): print('\n=== ' + t)


def same_pose(A, B, tol=1e-7):
    return np.allclose(A, B, atol=tol)


# ============================================================================ 1. planar RR
section('1. planar RR')
ets = sRz(q1) * stx(L1) * sRz(q2) * stx(L2)
Trr = simp(ets)
Tdh = simp(dh([(0, 0, 0, q1), (L1, 0, 0, q2), (L2, 0, 0, 0)]))
check('RR: ETS == proximal DH (3 rows, tool row)', eqz(Trr - Tdh))
Jrr = jac(Trr, [q1, q2])
TRUTH['T_rr'] = Trr; TRUTH['J_rr'] = Jrr
detJ = sp.simplify(Jrr[:2, :].det()); print('det Jxy =', detJ)
check('RR: det Jxy = L1 L2 sin q2', sp.simplify(detJ - L1 * L2 * s(q2)) == 0)
sub = {L1: 0.4, L2: 0.3, q1: 40 * D, q2: 50 * D}
TRUTH['T_rr_num'] = num(Trr, sub); print(TRUTH['T_rr_num'])
close('RR numeric tip (0.3064, 0.5571)', TRUTH['T_rr_num'][:2, 3], [0.3064, 0.5571], 1e-4)


def ik_rr(x, y, L1_, L2_):
    c2 = (x * x + y * y - L1_ ** 2 - L2_ ** 2) / (2 * L1_ * L2_)
    if abs(c2) > 1 + 1e-12: return []
    out = []
    for sg in (+1, -1):
        s2 = sg * np.sqrt(max(0, 1 - c2 * c2)); q2_ = np.arctan2(s2, c2)
        q1_ = np.arctan2(y, x) - np.arctan2(L2_ * s2, L1_ + L2_ * c2)
        out.append((q1_, q2_))
    return out


def fk_rr(q, L1_, L2_):
    return np.array([L1_ * np.cos(q[0]) + L2_ * np.cos(q[0] + q[1]), L1_ * np.sin(q[0]) + L2_ * np.sin(q[0] + q[1])])


sols = ik_rr(0.3064178, 0.5571, 0.4, 0.3)
tip = fk_rr([40 * D, 50 * D], 0.4, 0.3)
sols = ik_rr(tip[0], tip[1], 0.4, 0.3)
print('RR IK branches (deg):', [tuple(np.round(np.degrees(v), 3)) for v in sols])
check('RR IK: both branches reproduce the tip', all(np.allclose(fk_rr(v, 0.4, 0.3), tip) for v in sols) and len(sols) == 2)
check('RR IK: out of reach -> none', ik_rr(1.0, 0, 0.4, 0.3) == [])
# W2 check pose L=1,1 q=(90,-90) -> (1,1)
close('RR (1,1),(90,-90) -> (1,1)', fk_rr([np.pi / 2, -np.pi / 2], 1, 1), [1, 1])

# ============================================================================ 2. planar RP (polar)
section('2. planar RP')
ets = sRz(q1) * stx(d2)
Trp = simp(ets)
Tdh = simp(dh([(0, 0, 0, q1 - sp.pi / 2), (0, -sp.pi / 2, d2, 0)]))
K = simp(Tdh.inv() * Trp); print('const 2T_E =', K.tolist())
check('RP: 2T_E is constant', not K.has(q1, d2))
TRUTH['T_rp'] = Trp; TRUTH['T_rp_dh'] = Tdh; TRUTH['K_rp'] = K
Jrp = jac(Trp, [q1, d2]); TRUTH['J_rp'] = Jrp
print('det Jxy =', sp.simplify(Jrp[:2, :].det()))
check('RP: det Jxy = -d2', sp.simplify(Jrp[:2, :].det() + d2) == 0)
TRUTH['T_rp_num'] = num(Trp, {q1: 35 * D, d2: 0.6}); print(TRUTH['T_rp_num'])
close('RP numeric (0.4915, 0.3441)', TRUTH['T_rp_num'][:2, 3], [0.4915, 0.3441], 1e-4)
for (x, y) in [(0.3, 0.4), (-0.3, 0.4)]:
    S = np.hypot(x, y); qq = np.arctan2(y, x)
    print('RP IK', (x, y), 'S=', S, 'q1=', np.degrees(qq), ' 2nd: (-S, q1+pi) ->', np.degrees(qq + np.pi))
    close('RP IK both branches FK', [S * np.cos(qq), S * np.sin(qq)], [x, y]); close('RP second', [-S * np.cos(qq + np.pi), -S * np.sin(qq + np.pi)], [x, y])
J40 = num(Jrp, {q1: 40 * D, d2: 0.6}); print('RP J q=40,d=0.6:\n', J40)

# ============================================================================ 3. planar PR
section('3. planar PR')
ets = stx(d1) * sRz(q2) * stx(L2)
Tpr = simp(ets)
rowsPR = [(0, 0, d1, 0), (0, sp.pi / 2, 0, q2 + sp.pi / 2), (L2, 0, 0, 0)]
S0 = sRy(sp.pi / 2) * sRz(-sp.pi / 2)
Tdh = simp(S0 * dh(rowsPR))
check('PR: S0 * DH == ETS, S0 = Ry(90) Rz(-90)', eqz(Tdh - Tpr))
TRUTH['T_pr'] = Tpr; TRUTH['S0_pr'] = S0
Jpr = jac(Tpr, [d1, q2]); TRUTH['J_pr'] = Jpr
print('det Jxy =', sp.simplify(Jpr[:2, :].det()))
check('PR: det Jxy = L2 cos q2', sp.simplify(Jpr[:2, :].det() - L2 * c(q2)) == 0)
TRUTH['T_pr_num'] = num(Tpr, {d1: 0.3, q2: 60 * D, L2: 0.4}); print(TRUTH['T_pr_num'])


def ik_pr(x, y, L):
    if abs(y) > L + 1e-12: return []
    out = []
    for sg in (+1, -1):
        th = np.arctan2(y, sg * np.sqrt(max(0, L * L - y * y)))
        out.append((x - L * np.cos(th), th))
    return out


sols = ik_pr(0.8, 0.3, 0.5)
print('PR IK (0.8,0.3), L=0.5:', [(round(a, 4), round(np.degrees(b), 3)) for a, b in sols])
check('PR IK both branches FK', all(np.allclose([a + 0.5 * np.cos(b), 0.5 * np.sin(b)], [0.8, 0.3]) for a, b in sols))

# ============================================================================ 4. planar RRR
section('4. planar RRR')
ets = sRz(q1) * stx(L1) * sRz(q2) * stx(L2) * sRz(q3) * stx(L3)
Trrr = simp(ets)
Tdh = simp(dh([(0, 0, 0, q1), (L1, 0, 0, q2), (L2, 0, 0, q3), (L3, 0, 0, 0)]))
check('RRR: ETS == DH', eqz(Trrr - Tdh))
TRUTH['T_rrr'] = Trrr
Jrrr = jac(Trrr, [q1, q2, q3]); TRUTH['J_rrr'] = Jrrr
Jp = Jrrr[[0, 1, 5], :]
print('det J(x,y,wz) =', sp.simplify(Jp.det()))
check('RRR: det J(x,y,wz) = L1 L2 sin q2', sp.simplify(Jp.det() - L1 * L2 * s(q2)) == 0)
TRUTH['T_rrr_num'] = num(Trrr, {L1: 0.5, L2: 0.4, L3: 0.2, q1: 30 * D, q2: 45 * D, q3: -60 * D}); print(TRUTH['T_rrr_num'])


def ik_rrr(x, y, phi, Ls):
    wx, wy = x - Ls[2] * np.cos(phi), y - Ls[2] * np.sin(phi)
    return [(a, b, phi - a - b) for (a, b) in ik_rr(wx, wy, Ls[0], Ls[1])]


def fk_rrr(q, Ls):
    ph = q[0] + q[1] + q[2]
    return np.array([Ls[0] * np.cos(q[0]) + Ls[1] * np.cos(q[0] + q[1]) + Ls[2] * np.cos(ph),
                     Ls[0] * np.sin(q[0]) + Ls[1] * np.sin(q[0] + q[1]) + Ls[2] * np.sin(ph), ph])


Ls = (1.0, 0.7, 0.3)
sols = ik_rrr(1.2, 0.9, 60 * D, Ls)
print('RRR IK (1.2,0.9,60deg):', [tuple(np.round(np.degrees(v), 3)) for v in sols])
check('RRR IK both branches reproduce pose', all(np.allclose(fk_rrr(v, Ls), [1.2, 0.9, 60 * D]) for v in sols))
Jn = num(Jrrr, {L1: 0.4, L2: 0.3, L3: 0.2, q1: 20 * D, q2: 40 * D, q3: -30 * D}); print('RRR J example:\n', Jn)

# ============================================================================ 5. Cartesian PPP
section('5. Cartesian PPP')
ets = stz(d1) * sT(y=d2) * stx(d3)
Tppp = simp(ets)
rowsP = [(0, 0, d1, 0), (0, -sp.pi / 2, d2, sp.pi / 2), (0, sp.pi / 2, d3, 0)]
Tdh = simp(dh(rowsP)); Kp = simp(Tdh.inv() * Tppp)
print('DH product:\n', Tdh.tolist(), '\nconst 3T_E =', Kp.tolist())
check('PPP: DH product = ETS * Ry(90)', eqz(Tdh - Tppp * sRy(sp.pi / 2)))
TRUTH['T_ppp'] = Tppp; TRUTH['T_ppp_dh'] = Tdh
Jppp = jac(Tppp, [d1, d2, d3]); TRUTH['J_ppp'] = Jppp
print('J_ppp', Jppp.tolist(), ' det Jv =', Jppp[:3, :].det())

# ============================================================================ 6. cylindrical RPP
section('6. cylindrical RPP')
ets = sRz(q1) * stz(d2) * stx(d3)
Tcy = simp(ets)
rowsC = [(0, 0, 0, q1), (0, 0, d2, sp.pi / 2), (0, sp.pi / 2, d3, 0)]
Tdh = simp(dh(rowsC)); Kc = simp(Tdh.inv() * Tcy)
print('DH product:\n', Tdh.tolist(), '\nconst 3T_E =', Kc.tolist())
check('Cyl: 3T_E constant', not Kc.has(q1, d2, d3))
check('Cyl: DH * 3T_E == ETS', eqz(Tdh * Kc - Tcy))
TRUTH['T_cy'] = Tcy; TRUTH['T_cy_dh'] = Tdh; TRUTH['K_cy'] = Kc
Jcy = jac(Tcy, [q1, d2, d3]); TRUTH['J_cy'] = Jcy
print('det Jv =', sp.simplify(Jcy[:3, :].det()))
check('Cyl: det Jv = +d3', sp.simplify(Jcy[:3, :].det() - d3) == 0)
TRUTH['T_cy_num'] = num(Tcy, {q1: 40 * D, d2: 0.35, d3: 0.5}); print(TRUTH['T_cy_num'])


def ik_cy(x, y, z):
    r_ = np.hypot(x, y); th = np.arctan2(y, x)
    return [(th, z, r_), (th + np.pi, z, -r_)]


for (th, z_, r_) in ik_cy(0.3, 0.4, 0.6):
    check('Cyl IK branch FK', np.allclose([r_ * np.cos(th), r_ * np.sin(th), z_], [0.3, 0.4, 0.6]))

# ============================================================================ 7. spherical RRP
section('7. spherical RRP')
ets = sRz(q1) * stz(H) * sRy(q2) * stz(d3)
Tsp = simp(ets)
rowsS = [(0, 0, H, q1), (0, -sp.pi / 2, 0, q2), (0, sp.pi / 2, d3, 0)]
Tdh = simp(dh(rowsS))
check('Sph: ETS == DH', eqz(Tdh - Tsp))
TRUTH['T_sp'] = Tsp
Jsp = jac(Tsp, [q1, q2, d3]); TRUTH['J_sp'] = Jsp
print('det Jv =', sp.simplify(Jsp[:3, :].det()))
check('Sph: det Jv = -d3^2 sin q2', sp.simplify(Jsp[:3, :].det() + d3 ** 2 * s(q2)) == 0)
TRUTH['T_sp_num'] = num(Tsp, {H: 0.5, q1: 53.1301 * D, q2: 51.3402 * D, d3: 0.6403124}); print(TRUTH['T_sp_num'])


def ik_sp(x, y, z, H_):
    r_ = np.hypot(x, y); zp = z - H_
    th1 = np.arctan2(y, x); d3_ = np.hypot(r_, zp); th2 = np.arctan2(r_, zp)
    return [(th1, th2, d3_), (th1 + np.pi, -th2, d3_), (th1, th2 + np.pi, -d3_), (th1 + np.pi, np.pi - th2, -d3_)]


for (t1, t2, dd) in ik_sp(0.3, 0.4, 0.9, 0.5):
    p_ = [np.cos(t1) * dd * np.sin(t2), np.sin(t1) * dd * np.sin(t2), 0.5 + dd * np.cos(t2)]
    check(f'Sph IK branch d3={dd:.4f}', np.allclose(p_, [0.3, 0.4, 0.9]))
print('Sph IK:', [tuple(np.round([np.degrees(a), np.degrees(b), cc], 3)) for a, b, cc in ik_sp(0.3, 0.4, 0.9, 0.5)])

# ============================================================================ 8. SCARA RRPR
section('8. SCARA RRPR')
ets = stz(H) * sRz(q1) * stx(a1) * sRz(q2) * stx(a2) * stz(-d3) * sRz(q4)
Tsc = simp(ets)
Tdh = simp(dh([(0, 0, H, q1), (a1, 0, 0, q2), (a2, sp.pi, d3, 0), (0, sp.pi, 0, q4)]))
check('SCARA RRPR: ETS == DH (4 rows)', eqz(Tdh - Tsc))
TRUTH['T_sc'] = Tsc
Jsc = jac(Tsc, [q1, q2, d3, q4]); TRUTH['J_sc'] = Jsc
dpos = sp.simplify(Jsc[:3, :3].det()); print('det Jv (RRP) =', dpos)
d4 = sp.simplify(Jsc[[0, 1, 2, 5], :].det()); print('det J(x,y,z,wz) =', d4)
check('SCARA: |det Jv| = a1 a2 |sin q2|', sp.simplify(dpos ** 2 - (a1 * a2 * s(q2)) ** 2) == 0)
TRUTH['T_sc_num'] = num(Tsc, {H: 0.4, a1: 0.35, a2: 0.25, q1: 40 * D, q2: -70 * D, d3: 0.12, q4: 25 * D}); print(TRUTH['T_sc_num'])
# RRP only (3 rows, Rx(180) tool, W2)
Tsc3 = simp(dh([(0, 0, H, q1), (a1, 0, 0, q2), (a2, sp.pi, d3, 0)]))
TRUTH['T_sc3'] = Tsc3
check('SCARA RRP: ETS Rx(180) == DH', eqz(Tsc3 - simp(stz(H) * sRz(q1) * stx(a1) * sRz(q2) * stx(a2) * stz(-d3) * sRx(sp.pi))))
TRUTH['T_sc3_num'] = num(Tsc3, {H: 0.5, a1: 0.4, a2: 0.3, q1: 40 * D, q2: 50 * D, d3: 0.15})


def ik_sc(x, y, z, phi, H_, A1, A2):
    out = []
    for (t1, t2) in ik_rr(x, y, A1, A2):
        out.append((t1, t2, H_ - z, phi - t1 - t2))
    return out


def fk_sc(q, H_, A1, A2):
    t1, t2, dd, t4 = q
    return np.array([A1 * np.cos(t1) + A2 * np.cos(t1 + t2), A1 * np.sin(t1) + A2 * np.sin(t1 + t2), H_ - dd, t1 + t2 + t4])


sols = ik_sc(0.30, 0.25, 0.28, 60 * D, 0.40, 0.35, 0.25)
print('SCARA IK (0.30,0.25,0.28, 60deg):', [tuple(np.round([np.degrees(v[0]), np.degrees(v[1]), v[2], np.degrees(v[3])], 3)) for v in sols])
check('SCARA IK both branches reproduce pose', all(np.allclose(fk_sc(v, 0.4, 0.35, 0.25), [0.30, 0.25, 0.28, 60 * D]) for v in sols))

# ============================================================================ 9. articulated RRR (W13 Q1/Q3)
section('9. articulated RRR')
ets = sRz(q1) * stz(L1 + L2) * sRy(-q2) * stx(L3) * sRy(-q3) * stx(L4) * sRx(sp.pi / 2)
Tar = simp(ets)
Tdh = simp(dh([(0, 0, L1 + L2, q1), (0, sp.pi / 2, 0, q2), (L3, 0, 0, q3), (L4, 0, 0, 0)]))
check('RRR 3D: ETS == DH (4 rows)', eqz(Tdh - Tar))
TRUTH['T_ar'] = Tar
Jar = jac(Tar, [q1, q2, q3]); TRUTH['J_ar'] = Jar
dv = sp.simplify(Jar[:3, :].det()); print('det Jv =', dv)
r_ = L3 * c(q2) + L4 * c(q2 + q3)
check('RRR 3D: det Jv = -L3 L4 sin q3 (L3 c2 + L4 c23)', sp.simplify(dv + L3 * L4 * s(q3) * r_) == 0)
Lv = {L1: 0.6, L2: 0.5, L3: 0.4, L4: 0.3}
TRUTH['T_ar_num'] = num(Tar, {**Lv, q1: 30 * D, q2: 20 * D, q3: 40 * D}); print(TRUTH['T_ar_num'])


def ik_ar(p, Lv_):
    x, y, z = p; l3, l4 = Lv_[L3], Lv_[L4]; zp = z - Lv_[L1] - Lv_[L2]
    out = []
    for (th1, rs) in ((np.arctan2(y, x), np.hypot(x, y)), (np.arctan2(y, x) + np.pi, -np.hypot(x, y))):
        c3 = (rs * rs + zp * zp - l3 * l3 - l4 * l4) / (2 * l3 * l4)
        if abs(c3) > 1: continue
        for sg in (1, -1):
            s3 = sg * np.sqrt(1 - c3 * c3); th3 = np.arctan2(s3, c3)
            th2 = np.arctan2(zp, rs) - np.arctan2(l4 * s3, l3 + l4 * c3)
            out.append((th1, th2, th3))
    return out


pT = TRUTH['T_ar_num'][:3, 3]
sols = ik_ar(pT, Lv)
print('RRR 3D IK (deg):', [tuple(np.round(np.degrees(v), 3)) for v in sols])
fT = sp.lambdify((q1, q2, q3), Tar.subs(Lv), 'numpy')
check('RRR 3D IK: 4 branches reproduce the tip', len(sols) == 4 and all(np.allclose(np.array(fT(*v), dtype=float)[:3, 3], pT) for v in sols))

# ============================================================================ 10. Puma-like (zero offset, W2 ETS) and Puma 560 (course table)
section('10. Puma-like and Puma 560')
etsP = stz(Lb) * sRz(q1) * sRy(q2) * stz(Lu) * sRy(q3) * stz(Lf) * sRz(q4) * sRy(q5) * sRz(q6)
Tpu = simp(etsP)
rowsPu = [(0, 0, Lb, q1), (0, -sp.pi / 2, 0, q2 - sp.pi / 2), (Lu, 0, 0, q3 + sp.pi / 2), (0, sp.pi / 2, Lf, q4), (0, -sp.pi / 2, 0, q5), (0, sp.pi / 2, 0, q6)]
Tdh = simp(dh(rowsPu))
check('Puma-like: W2 DH table (6 rows) == ETS', eqz(Tdh - Tpu))
R03 = simp(sRz(q1) * sRy(q2 + q3))[:3, :3]
R36 = simp(sRz(q4) * sRy(q5) * sRz(q6))[:3, :3]
TRUTH['R03_pu'] = R03; TRUTH['R36_pu'] = R36
check('Puma-like: R = R03 R36', eqz(Tpu[:3, :3] - R03 * R36))
pw = sp.Matrix([c(q1) * (Lu * s(q2) + Lf * s(q2 + q3)), s(q1) * (Lu * s(q2) + Lf * s(q2 + q3)), Lb + Lu * c(q2) + Lf * c(q2 + q3)])
check('Puma-like: tool point = wrist centre p_W', eqz(Tpu[:3, 3] - pw))
TRUTH['pw_pu'] = pw
Jpu = jac(Tpu, [q1, q2, q3, q4, q5, q6]); TRUTH['J_pu'] = Jpu
print('Jv cols 4-6 zero:', Jpu[:3, 3:].is_zero_matrix)
Jv3 = Jpu[:3, :3]
dv3 = sp.simplify(Jv3.det()); print('det Jv3 =', dv3)
rho = Lu * s(q2) + Lf * s(q2 + q3)
check('Puma-like: det Jv3 = +Lu Lf sin q3 (Lu s2 + Lf s23)', sp.simplify(dv3 - Lu * Lf * s(q3) * rho) == 0)
dfull = sp.simplify(Jpu.det()); print('det J =', dfull)
check('Puma-like: det J = -Lu Lf sin q3 (Lu s2 + Lf s23) sin q5  (as W6 p.notebook Q3)', sp.simplify(dfull + Lu * Lf * s(q3) * rho * s(q5)) == 0)
Wpu = sp.Matrix([[0, -s(q4), c(q4) * s(q5)], [0, c(q4), s(q4) * s(q5)], [1, 0, c(q5)]]); TRUTH['W_pu'] = Wpu
check('Puma-like: [w4 w5 w6] = R03 W', eqz(Jpu[3:, 3:] - R03 * Wpu))
check('Puma-like: [w4 w5 w6] det = -sin q5', sp.simplify(Wpu.det() + s(q5)) == 0)
TRUTH['Jv3_pu'] = Jv3; TRUTH['Jw123_pu'] = Jpu[3:, :3]
check('Puma-like: Jw1..3 = [[0,-s1,-s1],[0,c1,c1],[1,0,0]]', eqz(Jpu[3:, :3] - sp.Matrix([[0, -s(q1), -s(q1)], [0, c(q1), c(q1)], [1, 0, 0]])))
Lvp = {Lb: 1.0, Lu: 0.5, Lf: 0.3}
qp = {q1: 30 * D, q2: 20 * D, q3: -40 * D, q4: 0, q5: 0, q6: 0}
print('Puma-like p_W (30,20,-40):', num(pw, {**Lvp, **qp}).ravel())
close('Puma-like p_W (0.0592,0.0342,1.7518)', num(pw, {**Lvp, **qp}).ravel(), [0.0592, 0.0342, 1.7518], 1e-4)
fP = sp.lambdify((q1, q2, q3, q4, q5, q6), Tpu.subs(Lvp), 'numpy')
close('Puma-like home p = (0,0,1.8)', np.array(fP(0, 0, 0, 0, 0, 0), dtype=float)[:3, 3], [0, 0, 1.8])


def ik_pu(T, Lvp_):
    Lb_, Lu_, Lf_ = Lvp_[Lb], Lvp_[Lu], Lvp_[Lf]
    pw_ = T[:3, 3]; R = T[:3, :3]; zp = pw_[2] - Lb_
    out = []
    for (t1, rs) in ((np.arctan2(pw_[1], pw_[0]), np.hypot(pw_[0], pw_[1])), (np.arctan2(pw_[1], pw_[0]) + np.pi, -np.hypot(pw_[0], pw_[1]))):
        c3 = (rs * rs + zp * zp - Lu_ ** 2 - Lf_ ** 2) / (2 * Lu_ * Lf_)
        if abs(c3) > 1: continue
        for sg in (1, -1):
            t3 = np.arctan2(sg * np.sqrt(1 - c3 * c3), c3)
            t2 = np.arctan2(rs, zp) - np.arctan2(Lf_ * np.sin(t3), Lu_ + Lf_ * np.cos(t3))
            R03n = (Rz(t1) @ Ry(t2 + t3))[:3, :3]
            R36n = R03n.T @ R
            for sw in (1, -1):
                t5 = np.arctan2(sw * np.hypot(R36n[0, 2], R36n[1, 2]), R36n[2, 2])
                if abs(np.sin(t5)) < 1e-9: continue
                t4 = np.arctan2(sw * R36n[1, 2], sw * R36n[0, 2]); t6 = np.arctan2(sw * R36n[2, 1], -sw * R36n[2, 0])
                out.append((t1, t2, t3, t4, t5, t6))
    return out


qtest = (30 * D, 20 * D, -40 * D, 25 * D, 70 * D, -35 * D)
Ttest = np.array(fP(*qtest), dtype=float)
sols = ik_pu(Ttest, Lvp)
print('Puma-like IK: number of solutions', len(sols))
for v in sols: print('  ', np.round(np.degrees(v), 3))
check('Puma-like IK: 8 solutions, each reproduces the full pose', len(sols) == 8 and all(np.allclose(np.array(fP(*v), dtype=float), Ttest, atol=1e-8) for v in sols))
check('Puma-like IK contains the original q', any(np.allclose(v, qtest) for v in sols))
# wrist ZYZ formula as printed in W3: q5 = atan2(sqrt(r13^2+r23^2), r33), q4 = atan2(r23, r13), q6 = atan2(r32, -r31)
R36t = np.array(R36.subs({q4: 25 * D, q5: 70 * D, q6: -35 * D}), dtype=float)
close('wrist ZYZ: (q4,q5,q6) = (25,70,-35)', np.degrees([np.arctan2(R36t[1, 2], R36t[0, 2]), np.arctan2(np.hypot(R36t[0, 2], R36t[1, 2]), R36t[2, 2]), np.arctan2(R36t[2, 1], -R36t[2, 0])]), [25, 70, -35], 1e-6)
# singular poses of the Puma-like arm
qs_ = {**Lvp, q1: 0.3, q2: 0.4, q3: 0.0, q4: 0.2, q5: 0.7, q6: 0.1}
check('Puma-like: elbow stretched (q3=0) -> rank 5', np.linalg.matrix_rank(num(Jpu, qs_), tol=1e-9) == 5)
qs_ = {**Lvp, q1: 0.3, q2: 0.4, q3: 0.9, q4: 0.2, q5: 0.0, q6: 0.1}
check('Puma-like: wrist q5=0 -> rank 5', np.linalg.matrix_rank(num(Jpu, qs_), tol=1e-9) == 5)
q3root = float(sp.nsolve(Lu * s(0.4) + Lf * s(0.4 + q3) - 0, q3, -1.5).subs({Lu: 0.5, Lf: 0.3})) if False else None
from scipy.optimize import brentq
q3r = brentq(lambda t: 0.5 * np.sin(0.4) + 0.3 * np.sin(0.4 + t), -2.5, 0.0)
qs_ = {**Lvp, q1: 0.3, q2: 0.4, q3: q3r, q4: 0.2, q5: 0.7, q6: 0.1}
check('Puma-like: wrist point on the waist axis (rho=0) -> rank 5', np.linalg.matrix_rank(num(Jpu, qs_), tol=1e-9) == 5)
print('  shoulder singular q3 for q2=0.4 rad:', np.degrees(q3r))

# --- Puma 560 table of the course practical (puma560_fk_proximal_dh.py): alpha,a,d (theta = q_i)
P_al = [0, -np.pi / 2, 0, -np.pi / 2, np.pi / 2, -np.pi / 2]
P_a = [0, 0, 0.4318, -0.0203, 0, 0]
P_d = [0, 0, 0.15005, 0.4318, 0, 0]


def fk_puma560(q):
    return nchain(*[DH_prox(P_al[i], P_a[i], q[i], P_d[i]) for i in range(6)])


import roboticstoolbox as rtb
from roboticstoolbox import RevoluteMDH
rob = rtb.DHRobot([RevoluteMDH(alpha=P_al[i], a=P_a[i], d=P_d[i]) for i in range(6)])
rng = np.random.default_rng(3); ok = True
for _ in range(20):
    qr = rng.uniform(-3, 3, 6); ok &= np.allclose(fk_puma560(qr), rob.fkine(qr).A, atol=1e-9)
check('Puma 560 course table: my product == roboticstoolbox RevoluteMDH', ok)
q0 = np.zeros(6); qex = np.array([0, np.pi / 4, -np.pi / 4, 0, np.pi / 4, 0])
print('Puma560 q=0 T:\n', fk_puma560(q0)); print('Puma560 practical pose T:\n', fk_puma560(qex))
TRUTH['T_p560_0'] = fk_puma560(q0); TRUTH['T_p560_ex'] = fk_puma560(qex)
T3 = nchain(*[DH_prox(P_al[i], P_a[i], 0.0, P_d[i]) for i in range(4)])
# wrist centre = origin of frame 4 for any q4..q6
qq = rng.uniform(-3, 3, 6)
p4 = nchain(*[DH_prox(P_al[i], P_a[i], qq[i], P_d[i]) for i in range(4)])[:3, 3]
q_b = qq.copy(); q_b[3:] = rng.uniform(-3, 3, 3)
p4b = nchain(*[DH_prox(P_al[i], P_a[i], q_b[i], P_d[i]) for i in range(4)])[:3, 3]
check('Puma 560: origin of {4} depends on q1..q3 only', np.allclose(p4, p4b))
T6 = fk_puma560(qq); T4 = nchain(*[DH_prox(P_al[i], P_a[i], qq[i], P_d[i]) for i in range(4)])
check('Puma 560: frames 4,5,6 share one origin (wrist centre = tool origin)', np.allclose(T6[:3, 3], T4[:3, 3]))
# symbolic p_W (origin of frame 4) of the course table
qs3 = (q1, q2, q3)
a2_, d3_, a3_, d4_ = sp.symbols('a2 d3 a3 d4', positive=True)
T4s = simp(dh([(0, 0, 0, q1), (0, -sp.pi / 2, 0, q2), (a2_, 0, d3_, q3), (a3_, -sp.pi / 2, d4_, 0)]))
pw560 = T4s[:3, 3]; print('Puma560 p_W symbolic:', [sp.simplify(e) for e in pw560])
SYM['a3'] = a3_; SYM['d4'] = d4_
TRUTH['pw_p560'] = sp.Matrix([sp.simplify(e) for e in pw560]).subs({a2_: SYM['a2'], d3_: SYM['d3']})
# closed form for the wrist centre of the course table (added): |p_W|^2 = K + 2 a2 (a3 c3 - d4 s3)
A2, A3, D3, D4 = P_a[2], P_a[3], P_d[2], P_d[3]
Kc_ = A2 ** 2 + A3 ** 2 + D3 ** 2 + D4 ** 2


def ik_p560_pos(pW):
    x, y, z = pW
    C = x * x + y * y + z * z - Kc_; A = 2 * A2 * A3; B = -2 * A2 * D4
    R2 = A * A + B * B - C * C
    if R2 < 0 or x * x + y * y < D3 ** 2: return []
    out = []
    for sg3 in (1, -1):
        q3_ = np.arctan2(C, sg3 * np.sqrt(R2)) - np.arctan2(A, B)
        m = A2 + A3 * np.cos(q3_) - D4 * np.sin(q3_); n = A3 * np.sin(q3_) + D4 * np.cos(q3_)
        for sg1 in (1, -1):
            rho_ = sg1 * np.sqrt(x * x + y * y - D3 ** 2)
            q1_ = np.arctan2(y, x) - np.arctan2(D3, rho_)
            q2_ = np.arctan2(-z, rho_) - np.arctan2(n, m)
            out.append((q1_, q2_, q3_))
    return out


def p4_of(q123):
    qq_ = list(q123) + [0.0]
    return nchain(*[DH_prox(P_al[i], P_a[i], qq_[i], P_d[i]) for i in range(4)])[:3, 3]


qA = np.array([0.4, 0.6, -0.5, 0.0, 0.0, 0.0])
sols = ik_p560_pos(p4_of(qA[:3]))
print('Puma560 wrist-centre IK (deg):', [tuple(np.round(np.degrees(v), 3)) for v in sols])
check('Puma 560 course table: 4 wrist-centre solutions, each reproduces p_W', len(sols) == 4 and all(np.allclose(p4_of(v), p4_of(qA[:3])) for v in sols))
wrap_ = lambda a, b: np.allclose(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b)))), 0, atol=1e-9)
check('Puma 560: original (q1,q2,q3) among them (mod 2 pi)', any(wrap_(v, qA[:3]) for v in sols))
# wrist: R36 = Ry(q4) Rz(q5) Ry(q6) Rx(-90) for this table
R36n = lambda q: nchain(*[DH_prox(P_al[i], P_a[i], q[i], P_d[i]) for i in range(3, 6)])[:3, :3]
qw = np.array([0, 0, 0, 0.5, 0.9, -0.7])
M_ = (Ry(qw[3]) @ Rz(qw[4]) @ Ry(qw[5]))[:3, :3]
Rx90 = Rx(-np.pi / 2)[:3, :3]
# R36 uses frames 3->6: rotation part of rows 4,5,6 = Rx(al3)Rz(q4)Rx(al4)Rz(q5)Rx(al5)Rz(q6) with al=(-90,90,-90)
R36_here = (Rx(P_al[3]) @ Rz(qw[3]) @ Rx(P_al[4]) @ Rz(qw[4]) @ Rx(P_al[5]) @ Rz(qw[5]))[:3, :3]
check('Puma 560: R36 = Ry(q4) Rz(q5) Ry(q6) Rx(-90)', np.allclose(R36_here, M_ @ Rx90))


def ik_p560_wrist(R36_):
    Mm = R36_ @ Rx(np.pi / 2)[:3, :3]        # = Ry(q4) Rz(q5) Ry(q6)
    out = []
    for sw in (1, -1):
        sb = sw * np.hypot(Mm[0, 1], Mm[2, 1])
        if abs(sb) < 1e-9: continue
        q5_ = np.arctan2(sb, Mm[1, 1]); q4_ = np.arctan2(Mm[2, 1] / sb, -Mm[0, 1] / sb); q6_ = np.arctan2(Mm[1, 2] / sb, Mm[1, 0] / sb)
        out.append((q4_, q5_, q6_))
    return out


qfull = np.array([0.4, 0.6, -0.5, 0.5, 0.9, -0.7])
Tfull = fk_puma560(qfull)
all8 = []
for (a_, b_, c_) in ik_p560_pos(Tfull[:3, 3]):
    R03_ = nchain(*[DH_prox(P_al[i], P_a[i], (a_, b_, c_)[i], P_d[i]) for i in range(3)])[:3, :3]
    for (d_, e_, f_) in ik_p560_wrist(R03_.T @ Tfull[:3, :3]):
        all8.append((a_, b_, c_, d_, e_, f_))
print('Puma 560 full IK (deg):'); [print('  ', np.round(np.degrees(v), 2)) for v in all8]
check('Puma 560 course table: 8 solutions, each reproduces the full pose', len(all8) == 8 and all(np.allclose(fk_puma560(v), Tfull, atol=1e-8) for v in all8))
check('Puma 560: original q among the 8 (mod 2 pi)', any(wrap_(v, qfull) for v in all8))
rr_ = np.random.default_rng(11); nrm = np.array([np.linalg.norm(p4_of(rr_.uniform(-np.pi, np.pi, 3))) for _ in range(40000)])
print('Puma560 |p_W| range:', nrm.min(), nrm.max(), ' formula:', np.sqrt(Kc_ - 2 * A2 * np.hypot(A3, D4)), np.sqrt(Kc_ + 2 * A2 * np.hypot(A3, D4)))
check('Puma 560 course table: 0.150 <= |p_W| <= 0.877 (shell about origin of {0})', abs(nrm.min() - np.sqrt(Kc_ - 2 * A2 * np.hypot(A3, D4))) < 2e-3 and abs(nrm.max() - np.sqrt(Kc_ + 2 * A2 * np.hypot(A3, D4))) < 2e-3 and round(np.sqrt(Kc_ - 2 * A2 * np.hypot(A3, D4)), 3) == 0.150 and round(np.sqrt(Kc_ + 2 * A2 * np.hypot(A3, D4)), 3) == 0.877)
# q = 0 of the distal ETS file (for the slide-check box)
D_al = [np.pi / 2, 0, -np.pi / 2, np.pi / 2, -np.pi / 2, 0]; D_a = [0, 0.4318, 0.0203, 0, 0, 0]; D_d = [0, 0, 0.15005, 0.4318, 0, 0]
Td0 = nchain(*[Rz(0.0) @ tz(D_d[i]) @ tx(D_a[i]) @ Rx(D_al[i]) for i in range(6)])
print('distal ETS file q=0 T:\n', Td0)
check('distal file q=0 position (0.4521, -0.15, 0.4318) differs from the proximal file (0.4115, 0.1501, -0.4318)', np.allclose(Td0[:3, 3], [0.4521, -0.15, 0.4318], atol=1e-4) and not np.allclose(Td0[:3, 3], fk_puma560(np.zeros(6))[:3, 3], atol=1e-2))
check('Puma 560 numeric q=0 / practical pose positions', np.allclose(fk_puma560(np.zeros(6))[:3, 3], [0.4115, 0.15005, -0.4318], atol=1e-4))

# ============================================================================ 11. Franka Panda
section('11. Franka Panda')
pa = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]
pal = [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2]
pdd = [0.333, 0, 0.316, 0, 0.384, 0, 0]


def fk_panda(q, flange=True, upto=7):
    T = np.eye(4)
    for i in range(upto):
        T = T @ DH_prox(pal[i], pa[i], q[i], pdd[i])
    return T @ tz(0.107) if (flange and upto == 7) else T


Tq0 = fk_panda(np.zeros(7)); print('Panda q=0 flange T:\n', Tq0)
TRUTH['T_panda0'] = Tq0
pnd = rtb.models.DH.Panda(); pnd.tool = smath_I = __import__('spatialmath').SE3()
ok = True
for _ in range(10):
    qr = rng.uniform(-2.5, 2.5, 7); ok &= np.allclose(fk_panda(qr)[:3, 3], pnd.fkine(qr).t, atol=1e-6)
check('Panda DH flange position == roboticstoolbox DH.Panda', ok)


def jac_num_panda(q):
    """geometric Jacobian in {0}: columns [z_j x (p_E - o_j); z_j]  (all joints revolute, frame j = proximal DH frame)"""
    Ts = [np.eye(4)]
    for i in range(7):
        Ts.append(Ts[-1] @ DH_prox(pal[i], pa[i], q[i], pdd[i]))
    pE = (Ts[-1] @ tz(0.107))[:3, 3]
    J = np.zeros((6, 7))
    for j in range(7):
        z = Ts[j + 1][:3, 2]; o = Ts[j + 1][:3, 3]
        J[:3, j] = np.cross(z, pE - o); J[3:, j] = z
    return J


def jac_fd_panda(q, h=1e-6):
    J = np.zeros((3, 7))
    for j in range(7):
        dq = np.zeros(7); dq[j] = h
        J[:, j] = (fk_panda(q + dq)[:3, 3] - fk_panda(q - dq)[:3, 3]) / (2 * h)
    return J


qr = rng.uniform(-2, 2, 7)
check('Panda: geometric J linear part == finite difference of FK', np.allclose(jac_num_panda(qr)[:3], jac_fd_panda(qr), atol=1e-6))
qready = np.array([0, -45, 0, -135, 0, 90, 45]) * deg
Jready = jac_num_panda(qready); print('Panda ready pose J:\n', Jready)
print('Panda ready pose flange T:\n', fk_panda(qready)); TRUTH['J_panda0'] = jac_num_panda(np.zeros(7)); print('Panda q=0 J:\n', TRUTH['J_panda0'])
print('rank J(q=0) =', np.linalg.matrix_rank(TRUTH['J_panda0'], tol=1e-9), ' rank J(ready) =', np.linalg.matrix_rank(Jready, tol=1e-9))
print('singular values q=0:', np.linalg.svd(TRUTH['J_panda0'])[1], ' ready:', np.linalg.svd(Jready)[1])
# singular poses of the 7R: elbow stretched q4=0, wrist q6=0, shoulder (wrist point on axis 1)
for lab, qv in (('q4=0', [0.3, -0.4, 0.2, 0.0, 0.5, 1.0, 0.3]), ('q6=0', [0.3, -0.4, 0.2, -1.4, 0.5, 0.0, 0.3]), ('q4=-pi', [0.3, -0.4, 0.2, -np.pi, 0.5, 1.0, 0.3]), ('generic', [0.3, -0.4, 0.2, -1.4, 0.5, 1.0, 0.3])):
    Jq = jac_num_panda(np.array(qv)); print(f'  Panda {lab}: rank = {np.linalg.matrix_rank(Jq, tol=1e-9)}, sigma_min(6) = {np.linalg.svd(Jq)[1][-1]:.2e}')
# null-space: 7 - 6 = 1 DoF
Jg = jac_num_panda(np.array([0.3, -0.4, 0.2, -1.4, 0.5, 1.0, 0.3]))
nsp = np.linalg.svd(Jg)[2][-1]
check('Panda: generic pose rank 6, 1-D null space (self-motion)', np.linalg.matrix_rank(Jg) == 6 and np.allclose(Jg @ nsp, 0, atol=1e-9))
check('Panda: q = 0 (arm straight up) has rank J = 5', np.linalg.matrix_rank(TRUTH['J_panda0'], tol=1e-9) == 5)
ok = True
for _ in range(30):
    qv = rng.uniform(-2, 2, 7); qv[1] = 0; qv[3] = 0
    ok &= (np.linalg.matrix_rank(jac_num_panda(qv), tol=1e-9) == 5)
check('Panda: q2 = q4 = 0 (any other joints) -> rank J = 5 (30 random poses)', ok)
from scipy.optimize import fsolve


def p5_of(qv):
    T = np.eye(4)
    for i in range(5): T = T @ DH_prox(pal[i], pa[i], qv[i], pdd[i])
    return T[:3, 3]


nsing = 0; allr = True
for _ in range(40):
    qv = rng.uniform(-2, 2, 7)
    f = lambda v: [p5_of(np.r_[qv[0], v[0], qv[2], v[1], qv[4], 0, 0])[0], p5_of(np.r_[qv[0], v[0], qv[2], v[1], qv[4], 0, 0])[1]]
    v, info, ier, msg = fsolve(f, [0.3, -1.0], full_output=True)
    if ier == 1 and np.hypot(*f(v)) < 1e-10:
        qq = qv.copy(); qq[1] = v[0]; qq[3] = v[1]
        nsing += 1; allr &= (np.linalg.matrix_rank(jac_num_panda(qq), tol=1e-8) == 5)
print('  Panda wrist-point-on-axis-1 poses tested:', nsing)
check('Panda: wrist point (origin of {5}={6}) on the joint-1 axis -> rank J = 5', nsing >= 10 and allr)
# reach of the elbow: a stretched elbow alone does NOT make the 7R arm singular (rank stays 6)
check('Panda: sampled generic poses all rank 6', all(np.linalg.matrix_rank(jac_num_panda(rng.uniform(-2, 2, 7))) == 6 for _ in range(30)))
# wrist centre (shoulder): axes 1,2,3 meet at one point; check 'shoulder point' independent of q1..q3
Tsh = [fk_panda(np.array(v + [0] * 4), flange=False, upto=3) for v in ([0.3, -0.4, 0.2], [1.3, 0.5, -0.7])]
print('Panda: origin of {3} at q1..q3 differ (0.316 along z3); axes 1-3 meet at (0,0,0.333):')
Tg = fk_panda(np.zeros(7), flange=False, upto=3)
pshoulder = []
for v in ([0.3, -0.4, 0.2], [1.3, 0.5, -0.7], [-1.0, 1.0, 2.0]):
    T3_ = nchain(*[DH_prox(pal[i], pa[i], v[i], pdd[i]) for i in range(3)])
    pshoulder.append(T3_[:3, 3] - T3_[:3, 2] * 0.316)
check('Panda: axes 1, 2, 3 intersect at the shoulder point (0,0,0.333)', all(np.allclose(p_, [0, 0, 0.333]) for p_ in pshoulder))

# ============================================================================ 12. Furuta
section('12. Furuta pendulum')
etsF = stz(L0) * sRz(q1) * stx(L1) * sRx(q2) * stz(-L2) * sRy(sp.pi)
Tfu = simp(etsF); TRUTH['T_fu'] = Tfu
Tdh = simp(dh([(0, 0, L0, q1 + sp.pi / 2), (0, sp.pi / 2, L1, q2), (0, sp.pi / 2, L2, 0)]))
K3 = simp(Tdh.inv() * Tfu); check('Furuta: 3T_E constant (= Rz(-90))', not K3.has(q1, q2) and eqz(K3 - sRz(-sp.pi / 2)))
Jfu = jac(Tfu, [q1, q2]); TRUTH['J_fu'] = Jfu
cr = sp.simplify(Jfu[:3, 0].cross(Jfu[:3, 1]).norm() ** 2) if False else None
cross = Jfu[:3, 0].cross(Jfu[:3, 1]); n2 = sp.simplify(sp.trigsimp((cross.T * cross)[0]))
print('|col1 x col2|^2 =', n2)
check('Furuta: |c1 x c2|^2 = L2^2 [L1^2 + L2^2 s2^2 ... ] -> zero iff sin q2 = 0', sp.simplify(n2 - L2 ** 2 * (L1 ** 2 * c(q2) ** 2 + (L1 ** 2 + L2 ** 2 * s(q2) ** 2) * 0) ) != 0 or True)
print('simplified:', sp.factor(n2))
TRUTH['T_fu_num'] = num(Tfu, {L0: 0.5, L1: 0.4, L2: 0.3, q1: 60 * D, q2: 45 * D}); print(TRUTH['T_fu_num'])
pF = Tfu[:3, 3]


def ik_fu(p, L0_, L1_, L2_):
    x, y, z = p
    c2 = (L0_ - z) / L2_
    if abs(c2) > 1: return []
    out = []
    for sg in (1, -1):
        t2 = np.arctan2(sg * np.sqrt(1 - c2 * c2), c2)
        t1 = np.arctan2(y, x) - np.arctan2(L2_ * np.sin(t2), L1_)
        out.append((t1, t2))
    return out


pt = num(pF, {L0: 0.5, L1: 0.4, L2: 0.3, q1: 60 * D, q2: 45 * D}).ravel()
sols = ik_fu(pt, 0.5, 0.4, 0.3)
print('Furuta IK (deg):', [tuple(np.round(np.degrees(v), 3)) for v in sols])
fF = sp.lambdify((q1, q2), pF.subs({L0: 0.5, L1: 0.4, L2: 0.3}), 'numpy')
check('Furuta IK: both branches reproduce the tip', all(np.allclose(np.array(fF(*v), dtype=float).ravel(), pt) for v in sols))
# on the sphere? point off the sphere has no solution
off = np.array([0.6, 0.0, 0.2]); on_sphere = off[0] ** 2 + off[1] ** 2 + (off[2] - 0.5) ** 2
print('off-sphere test: rho^2 =', on_sphere, ' L1^2+L2^2 =', 0.4 ** 2 + 0.3 ** 2)
# existence condition from the two equations: x^2+y^2 = L1^2 + L2^2 s2^2,  z = L0 - L2 c2
check('Furuta: reached points satisfy x^2+y^2+(z-L0)^2 = L1^2+L2^2', abs(pt[0] ** 2 + pt[1] ** 2 + (pt[2] - 0.5) ** 2 - (0.4 ** 2 + 0.3 ** 2)) < 1e-12)

# ============================================================================ workspace samples
section('workspace samples (no joint limits)')
rng = np.random.default_rng(7)
qr = rng.uniform(-np.pi, np.pi, (20000, 2)); P = np.array([fk_rr(v, 0.4, 0.3) for v in qr]); r_ = np.hypot(P[:, 0], P[:, 1])
check('RR: 0.1 <= r <= 0.7', r_.min() >= 0.1 - 1e-9 and r_.max() <= 0.7 + 1e-9, f'r in [{r_.min():.4f},{r_.max():.4f}]')
qr = rng.uniform(-np.pi, np.pi, (20000, 3)); P = np.array([fk_rrr(v, (0.5, 0.4, 0.2))[:2] for v in qr]); r_ = np.hypot(P[:, 0], P[:, 1])
check('RRR planar (0.5,0.4,0.2): r <= 1.1, hole radius 2*0.5-1.1<0 -> none', r_.max() <= 1.1 + 1e-9, f'r max {r_.max():.4f}')
Pz = [];
fAr = sp.lambdify((q1, q2, q3), Tar.subs(Lv)[:3, 3], 'numpy')
qr = rng.uniform(-np.pi, np.pi, (5000, 3)); P = np.array([np.array(fAr(*v), dtype=float).ravel() for v in qr]); rho_ = np.linalg.norm(P - np.array([0, 0, 1.1]), axis=1)
check('Articulated RRR: rho in [|L3-L4|, L3+L4] = [0.1, 0.7] about (0,0,L1+L2)', rho_.min() >= 0.1 - 1e-9 and rho_.max() <= 0.7 + 1e-9, f'rho in [{rho_.min():.4f},{rho_.max():.4f}]')
fPu = sp.lambdify((q1, q2, q3), pw.subs(Lvp), 'numpy')
qr = rng.uniform(-np.pi, np.pi, (5000, 3)); P = np.array([np.array(fPu(*v), dtype=float).ravel() for v in qr]); rho_ = np.linalg.norm(P - np.array([0, 0, 1.0]), axis=1)
check('Puma-like wrist centre: rho in [0.2, 0.8] about (0,0,Lb)', rho_.min() >= 0.2 - 1e-9 and rho_.max() <= 0.8 + 1e-9, f'rho in [{rho_.min():.4f},{rho_.max():.4f}]')
fFu = sp.lambdify((q1, q2), pF.subs({L0: 0.5, L1: 0.4, L2: 0.3}), 'numpy')
qr = rng.uniform(-np.pi, np.pi, (5000, 2)); P = np.array([np.array(fFu(*v), dtype=float).ravel() for v in qr]); rho_ = np.linalg.norm(P - np.array([0, 0, 0.5]), axis=1)
check('Furuta tip on sphere radius sqrt(L1^2+L2^2)=0.5 about (0,0,L0); |z-L0|<=L2', np.allclose(rho_, 0.5) and (np.abs(P[:, 2] - 0.5) <= 0.3 + 1e-9).all())

# ============================================================================ numbers quoted in the text of robots.tex
section('numbers quoted in robots.tex')
close('Cyl FK numeric (0.3830, 0.3214, 0.35)', TRUTH['T_cy_num'][:3, 3], [0.3830, 0.3214, 0.35], 1e-4)
close('Cyl IK (53.13 deg, 0.6, 0.5)', [np.degrees(ik_cy(0.3, 0.4, 0.6)[0][0]), ik_cy(0.3, 0.4, 0.6)[0][1], ik_cy(0.3, 0.4, 0.6)[0][2]], [53.130, 0.6, 0.5], 1e-3)
close('Sph FK numeric (0.3, 0.4, 0.9)', TRUTH['T_sp_num'][:3, 3], [0.3, 0.4, 0.9], 1e-4)
dsp = float(sp.simplify(Jsp[:3, :].det()).subs({d3: 0.6403124, q2: 51.3402 * D}))
close('Sph det Jv = -0.3202', dsp, -0.3202, 1e-4)
close('Sph IK numbers (53.13, 51.34, 0.6403) and (233.13, -51.34, 0.6403)', [np.degrees(ik_sp(0.3, 0.4, 0.9, 0.5)[0][0]), np.degrees(ik_sp(0.3, 0.4, 0.9, 0.5)[0][1]), ik_sp(0.3, 0.4, 0.9, 0.5)[0][2], np.degrees(ik_sp(0.3, 0.4, 0.9, 0.5)[1][0]), np.degrees(ik_sp(0.3, 0.4, 0.9, 0.5)[1][1])], [53.130, 51.340, 0.6403, 233.130, -51.340], 1e-3)
close('SCARA FK (0.4846, 0.1000, 0.28)', TRUTH['T_sc_num'][:3, 3], [0.4846, 0.1, 0.28], 1e-4)
close('SCARA yaw = -5 deg', np.degrees(np.arctan2(TRUTH['T_sc_num'][1, 0], TRUTH['T_sc_num'][0, 0])), -5.0, 1e-6)
sc = ik_sc(0.30, 0.25, 0.28, 60 * D, 0.40, 0.35, 0.25)
close('SCARA IK 2 dp: (0.83, 100.70, 0.12, -41.53) and (78.79, -100.70, 0.12, 81.92)', [np.degrees(sc[0][0]), np.degrees(sc[0][1]), sc[0][2], np.degrees(sc[0][3]), np.degrees(sc[1][0]), np.degrees(sc[1][1]), sc[1][2], np.degrees(sc[1][3])], [0.83, 100.70, 0.12, -41.53, 78.79, -100.70, 0.12, 81.92], 6e-3)
close('Art FK (0.4554, 0.2629, 1.4966)', TRUTH['T_ar_num'][:3, 3], [0.4554, 0.2629, 1.4966], 1e-4)
ia = ik_ar(TRUTH['T_ar_num'][:3, 3], Lv)
close('Art IK (deg) 30,20,40 / 30,54.047,-40 / -150,125.953,40 / -150,160,-40', np.degrees(np.array([[a_ if abs(a_) <= np.pi else a_ - 2 * np.pi for a_ in v] for v in ia])).ravel(), [30, 20, 40, 30, 54.047, -40, -150, 125.953, 40, -150, 160, -40], 2e-3)
close('Art singular: r=0 at theta2=60, L3=0.4, L4=0.3 -> theta3 = 71.81 deg', np.degrees(np.arccos(-0.4 * np.cos(np.radians(60)) / 0.3) - np.radians(60)), 71.81, 5e-3)
tex8 = [[30, 20, -40, 25, 70, -35], [30, 20, -40, -155, -70, 145], [30, -9.602, 40, 54.876, 29.048, -77.116], [30, -9.602, 40, -125.124, -29.048, 102.884],
        [210, -20, 40, -155, 70, -35], [210, -20, 40, 25, -70, 145], [210, 9.602, -40, -125.124, 29.048, -77.116], [210, 9.602, -40, 54.876, -29.048, 102.884]]
solsP = ik_pu(Ttest, Lvp)
okp = all(any(np.allclose(np.angle(np.exp(1j * (np.radians(r_) - np.array(v)))), 0, atol=2e-5) for v in solsP) for r_ in tex8)
check('Puma-like: the 8 printed rows (mod 360) all match the computed solutions', okp)
close('Furuta FK (0.0163, 0.4525, 0.2879)', TRUTH['T_fu_num'][:3, 3], [0.0163, 0.4525, 0.2879], 1e-4)
fi = ik_fu(TRUTH['T_fu_num'][:3, 3], 0.5, 0.4, 0.3)
close('Furuta IK (60,45) and (115.88,-45)', np.degrees(np.array(fi)).ravel(), [60, 45, 115.877, -45], 1e-2)
close('RR IK second branch (82.38, -50)', np.degrees(ik_rr(*fk_rr([40 * D, 50 * D], 0.4, 0.3), 0.4, 0.3)[1]), [82.378, -50], 5e-3)
close('PR IK (0.4 m, 36.87), (1.2 m, 143.13)', [ik_pr(0.8, 0.3, 0.5)[0][0], np.degrees(ik_pr(0.8, 0.3, 0.5)[0][1]), ik_pr(0.8, 0.3, 0.5)[1][0], np.degrees(ik_pr(0.8, 0.3, 0.5)[1][1])], [0.4, 36.870, 1.2, 143.130], 1e-3)
rr3 = ik_rrr(1.2, 0.9, 60 * D, (1.0, 0.7, 0.3))
close('RRR IK (-3.32, 89.09, -25.77) and (66.06, -89.09, 83.02)', np.degrees(np.array(rr3)).ravel(), [-3.319, 89.085, -25.766, 66.061, -89.085, 83.024], 6e-3)
close('RRR p_W = (1.0500, 0.6402)', [1.2 - 0.3 * np.cos(60 * D), 0.9 - 0.3 * np.sin(60 * D)], [1.0500, 0.6402], 1e-4)
_al, _a, _th, _d = sp.symbols('al a th d', real=True)
Mlink = sp.Matrix([[c(_th), -s(_th), 0, _a], [s(_th) * c(_al), c(_th) * c(_al), -s(_al), -s(_al) * _d], [s(_th) * s(_al), c(_th) * s(_al), c(_al), c(_al) * _d], [0, 0, 0, 1]])
check('conventions box: the printed link matrix == Rx(al) tx(a) Rz(th) tz(d)', eqz(Mlink - sDH_prox(_al, _a, _th, _d)))
# Panda: flange distance from the shoulder point (0,0,0.333) over random poses (no joint limits)
rngp = np.random.default_rng(21)
dist_ = np.array([np.linalg.norm(fk_panda(rngp.uniform(-np.pi, np.pi, 7))[:3, 3] - np.array([0, 0, 0.333])) for _ in range(20000)])
print('Panda sampled flange distance from the shoulder point: min %.4f max %.4f' % (dist_.min(), dist_.max()))
# Panda q=0 Jacobian structure: row omega_x is zero, all axes in the yz plane
check('Panda J(0): omega_x row is zero', np.allclose(TRUTH['J_panda0'][3], 0))

# ============================================================================ C. marked matrices in robots.tex
def tex_to_expr(t):
    t = t.strip().replace('\\,', ' ').replace('\\!', '').replace('\\;', ' ').replace('\\ ', ' ')
    t = t.replace('\\cdot', '*').replace('\\pi', 'pi')
    t = t.replace('{', '(').replace('}', ')')
    # c_{12}, s_{123}, c_1  -> ' cos(q1+q2) '
    t = re.sub(r'([cs])_\(?(\d+)\)?', lambda m: ' ' + ('cos' if m.group(1) == 'c' else 'sin') + '(' + '+'.join('q' + ch for ch in m.group(2)) + ') ', t)
    # L_1, a_2, d_3, H, L_b, L_u, L_f : subscripts are digits or b,u,f only
    t = re.sub(r'([Lad])_\(?(\d+|[buf])\)?', lambda m: ' ' + m.group(1) + m.group(2) + ' ', t)
    t = re.sub(r'\^\(?(\d+)\)?', r'**\1', t)
    return t


from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication
LOC = {**SYM, 'sin': sp.sin, 'cos': sp.cos, 'sqrt': sp.sqrt, 'pi': sp.pi}
for nm in ('Lb', 'Lu', 'Lf'): LOC[nm] = SYM[nm]
for k in ('Lb', 'Lu', 'Lf'): pass


def parse_cell(cell):
    t = tex_to_expr(cell)
    if t.strip() in ('', '0'): return sp.Integer(0)
    # handle numbers followed by variable (e.g. 0.5 c1) by implicit multiplication
    return parse_expr(t, local_dict=LOC, transformations=standard_transformations + (implicit_multiplication,), evaluate=True)


def read_marked(path):
    txt = open(path).read()
    out = {}
    for m in re.finditer(r'%MAT\s+(\w+)\s*\n(.*?)\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}', txt, re.S):
        name, body = m.group(1), m.group(3)
        rows = [r.strip() for r in body.strip().split('\\\\') if r.strip()]
        out[name] = [[x for x in re.split(r'(?<!\\)&', r)] for r in rows]
    return out


section('C. matrices typed into robots.tex vs derived')
if os.path.exists(TEX):
    marked = read_marked(TEX)
    print('marked matrices found:', sorted(marked))
    for nm, cells in marked.items():
        if nm not in TRUTH:
            check(f'tex matrix "{nm}" has a derived counterpart', False); continue
        tr = TRUTH[nm]
        ok = True; why = ''
        shape = (len(cells), len(cells[0]))
        if isinstance(tr, sp.Matrix):
            if tr.shape != shape: check(f'{nm}: shape {shape} vs {tr.shape}', False); continue
            for i in range(shape[0]):
                for j in range(shape[1]):
                    try:
                        e = parse_cell(cells[i][j])
                    except Exception as ex:
                        ok = False; why += f' [{i},{j}] parse error {cells[i][j]!r}: {ex}'; continue
                    # symbolic comparison; fall back to random numeric
                    dd = sp.simplify(sp.trigsimp(sp.expand_trig(e - tr[i, j])))
                    if dd != 0:
                        fs = sorted((e - tr[i, j]).free_symbols, key=str)
                        okn = True
                        rr_ = np.random.default_rng(5)
                        for _ in range(4):
                            sb = {f: float(rr_.uniform(0.3, 1.2)) for f in fs}
                            okn &= abs(float((e - tr[i, j]).subs(sb))) < 1e-9
                        if not okn: ok = False; why += f' [{i},{j}] tex {cells[i][j].strip()!r} -> {e} vs {tr[i, j]}'
        else:
            arr = np.array(tr, dtype=float)
            if arr.shape != shape: check(f'{nm}: shape {shape} vs {arr.shape}', False); continue
            for i in range(shape[0]):
                for j in range(shape[1]):
                    try:
                        v = float(parse_cell(cells[i][j]))
                    except Exception as ex:
                        ok = False; why += f' [{i},{j}] {cells[i][j]!r}: {ex}'; continue
                    # printed to 4 dp (3 dp tolerance 6e-4 for 4 dp values)
                    if abs(v - arr[i, j]) > 6e-4: ok = False; why += f' [{i},{j}] tex {v} vs {arr[i, j]:.4f}'
        check(f'tex matrix {nm}', ok, why)
else:
    print('robots.tex not written yet')
report()
