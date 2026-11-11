"""Booklet v2, Week 2 (forward kinematics): verifies every matrix and number used in booklet/w02.tex.
Run: python3 -I? no: python3 vcheck/v2_w02.py   (prints labelled numbers + PASS/FAIL)"""
import sys, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'checks')
sys.path.insert(0, ROOT)
from common import *
import roboticstoolbox as rtb
import spatialmath as smath
np.set_printoptions(precision=4, suppress=True, linewidth=150)
S = sp
q1, q2, q3, q4, q5, q6 = sp.symbols('q1 q2 q3 q4 q5 q6', real=True)
L0, L1, L2, L3, L4, H, a1, a2, d3s = sp.symbols('L0 L1 L2 L3 L4 H a1 a2 d3', real=True)
c, s = sp.cos, sp.sin
def eqz(M): return sp.simplify(sp.trigsimp(M)) == sp.zeros(*M.shape)
def show(name, M):
    print('---', name); sp.pprint(sp.simplify(M)); print()
def num(M, sub): return np.array(M.subs(sub).evalf(), dtype=float)

# ===== 0. rotations, homogeneous transform, inverse (slide p.12; Rx/Ry/Rz matrices are 'added') =====
for nm, R in (('Rx', sRx(q1)), ('Ry', sRy(q1)), ('Rz', sRz(q1))):
    R3 = R[:3, :3]
    check(f'{nm}: R^T R = I', eqz(R3.T*R3 - sp.eye(3)))
    check(f'{nm}: det R = +1', sp.simplify(R3.det() - 1) == 0)
# columns = axes of B in A, rows of R^T = same
th = 90*deg
R = Rz(th)[:3, :3]
t = np.array([2, 1, 3.0])
TAB = np.eye(4); TAB[:3, :3] = R; TAB[:3, 3] = t
BP = np.array([1.0, 0.5, 0.0])
AP = R @ BP + t
print('EX1 AP =', AP, '(want 1.5 2 3)')
close('EX1 A_P = (1.5, 2, 3)', AP, [1.5, 2, 3])
TBA = np.eye(4); TBA[:3, :3] = R.T; TBA[:3, 3] = -R.T @ t
print('EX1 BTA =\n', TBA)
close('EX1 inverse formula = numpy inv', TBA, np.linalg.inv(TAB))
close('EX1 BP back = (1,0.5,0)', (TBA @ np.r_[AP, 1])[:3], BP)
close('EX1 wrong inverse -t differs', ((np.r_[AP, 1] - np.r_[t, 0])[:3]) , AP - t, show=False)  # trivial
print('EX1 R^T t =', R.T @ t, ' -R^T t =', -R.T @ t)
close('EX1 -R^T t = (-1, 2, -3)', -R.T @ t, [-1, 2, -3])

# ===== 1. compounding (slide p.13) =====
th, d = 30*deg, 2.0
ATC = Rx(th) @ tz(d)
print('ATC = Rx(30) tz(2) =\n', ATC)
close('compound origin (0,-1,1.7321)', ATC[:3, 3], [0, -1.0, 1.7321], tol=1e-4)
TZR = tz(d) @ Rx(th)
close('order: tz Rx origin (0,0,2)', TZR[:3, 3], [0, 0, 2])
print('tz(2) Rx(30) =\n', TZR)
# current vs fixed reading of M = Rz(90) tx(1)
M = Rz(90*deg) @ tx(1)
print('M = Rz(90)tx(1) =\n', M)
close('M origin (0,1,0)', M[:3, 3], [0, 1, 0])
N = tx(1) @ Rz(90*deg)
print('N = tx(1)Rz(90) =\n', N); close('N origin (1,0,0)', N[:3, 3], [1, 0, 0])
close('M and N same rotation', M[:3, :3], N[:3, :3])

# ===== 2. elementary transforms ETS notation: full matrices are standard (checked in common.py) =====

# ===== 3. planar RRR by trigonometry (slides pp.6-9) =====
Ls = (0.5, 0.4, 0.2); qd = (30, 45, -60)
qq = np.array(qd)*deg
phi = np.cumsum(qq)
xA = Ls[0]*np.cos(phi[0]); yA = Ls[0]*np.sin(phi[0])
xB = xA + Ls[1]*np.cos(phi[1]); yB = yA + Ls[1]*np.sin(phi[1])
xE = xB + Ls[2]*np.cos(phi[2]); yE = yB + Ls[2]*np.sin(phi[2])
print('RRR: phi(deg)=', phi/deg, 'A=', (xA, yA), 'B=', (xB, yB), 'E=', (xE, yE))
close('RRR E = (0.7297, 0.6881)', [xE, yE], [0.7297, 0.6881], tol=1e-4)
print('RRR cos/sin of phi3=15deg', np.cos(phi[2]), np.sin(phi[2]))
T3 = nchain(Rz(qq[0]), tx(Ls[0]), Rz(qq[1]), tx(Ls[1]), Rz(qq[2]), tx(Ls[2]))
close('RRR ETS = trig', T3[:2, 3], [xE, yE])
print('RRR T =\n', T3)
# symbolic full matrix
T3s = chain(sRz(q1), stx(L1), sRz(q2), stx(L2), sRz(q3), stx(L3))
phis = q1+q2+q3
T3hand = sp.Matrix([[c(phis), -s(phis), 0, L1*c(q1)+L2*c(q1+q2)+L3*c(phis)],
                    [s(phis), c(phis), 0, L1*s(q1)+L2*s(q1+q2)+L3*s(phis)], [0, 0, 1, 0], [0, 0, 0, 1]])
check('RRR symbolic T == hand matrix', eqz(T3s - T3hand))
DHr = chain(sDH_prox(0, 0, q1, 0), sDH_prox(0, L1, q2, 0), sDH_prox(0, L2, q3, 0), sDH_prox(0, L3, 0, 0))
check('RRR proximal DH (4 rows incl. tool row) == ETS', eqz(DHr - T3s))

# ===== 4. planar RR =====
T2s = chain(sRz(q1), stx(L1), sRz(q2), stx(L2))
T2hand = sp.Matrix([[c(q1+q2), -s(q1+q2), 0, L1*c(q1)+L2*c(q1+q2)], [s(q1+q2), c(q1+q2), 0, L1*s(q1)+L2*s(q1+q2)], [0, 0, 1, 0], [0, 0, 0, 1]])
check('RR symbolic T == hand matrix', eqz(T2s - T2hand))
DHrr = chain(sDH_prox(0, 0, q1, 0), sDH_prox(0, L1, q2, 0), sDH_prox(0, L2, 0, 0))
check('RR proximal DH (3 rows incl. tool row) == ETS', eqz(DHrr - T2s))
DHrrd = chain(sDH_dist(q1, 0, L1, 0), sDH_dist(q2, 0, L2, 0))
check('RR distal DH (2 rows, no tool row) == ETS', eqz(DHrrd - T2s))
# numeric RR example
qa = np.array([40, 50])*deg; La = (0.4, 0.3)
T = nchain(Rz(qa[0]), tx(La[0]), Rz(qa[1]), tx(La[1]))
print('RR num (L=0.4,0.3; q=40,50):\n', T)
print('RR elbow', La[0]*np.cos(qa[0]), La[0]*np.sin(qa[0]))
# slide-style check: with L=1, q=(90,-90): (1,1)
T = nchain(Rz(90*deg), tx(1), Rz(-90*deg), tx(1)); close('RR (90,-90) tip (1,1)', T[:2, 3], [1, 1])

# ===== 5. RP (polar) =====
# ETS Rz(q1) tx(q2); DH: row1 (0,0,0,q1-90), row2 (-90,0,q2,0); tool constant 2T_E = Rx(90)Rz(90)
qpar = sp.Symbol('d2', real=True)
RPets = chain(sRz(q1), stx(qpar))
RPdh = chain(sDH_prox(0, 0, q1 - sp.pi/2, 0), sDH_prox(-sp.pi/2, 0, 0, qpar))
Ctool = chain(sRx(sp.pi/2), sRz(sp.pi/2))
check('RP: DH (q1-90deg ; alpha1=-90, d2) * Rx(90)Rz(90) == ETS Rz(q1) tx(d2)', eqz(RPdh*Ctool - RPets))
show('RP 0T_2 (DH frame {2})', RPdh)
RPhand = sp.Matrix([[c(q1), -s(q1), 0, qpar*c(q1)], [s(q1), c(q1), 0, qpar*s(q1)], [0, 0, 1, 0], [0, 0, 0, 1]])
check('RP ETS matrix == hand', eqz(RPets - RPhand))
print('RP num q1=35, d2=0.6 ->', num(RPets, {q1: 35*sp.pi/180, qpar: 0.6})[:3, 3], 'matrix\n', num(RPets, {q1: 35*sp.pi/180, qpar: 0.6}))
print('RP dh 0T2 num\n', num(RPdh, {q1: 35*sp.pi/180, qpar: 0.6}))

# ===== 6. PR (horizontal slide then revolute about vertical) =====
d1s = sp.Symbol('d1', real=True)
PRets = chain(stx(d1s), sRz(q2), stx(L2))
PRhand = sp.Matrix([[c(q2), -s(q2), 0, d1s+L2*c(q2)], [s(q2), c(q2), 0, L2*s(q2)], [0, 0, 1, 0], [0, 0, 0, 1]])
check('PR ETS matrix == hand', eqz(PRets - PRhand))
# base frame {0}: z0 = x_S (slide), x0 = -y_S, y0 = -z_S ; ST0 = [x0 y0 z0]
ST0 = sp.Matrix([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
check('PR: ST0 is a proper rotation', sp.simplify(ST0[:3, :3].det()) == 1)
PRdh = chain(sDH_prox(0, 0, 0, d1s), sDH_prox(sp.pi/2, 0, q2 + sp.pi/2, 0), sDH_prox(0, L2, 0, 0))
check('PR: ST0 * DH(0,0,0,d1)(90,0,q2+90,0)(0,L2,0,0) == ETS', eqz(ST0*PRdh - PRets))
show('PR 0T_3', PRdh)
# ST0 is Ry(90)Rz(90)? print factorisation check
check('PR: ST0 = Ry(90)Rz(-90)', eqz(chain(sRy(sp.pi/2), sRz(-sp.pi/2)) - ST0))
print('PR num d1=0.3, q2=60:', num(PRets, {d1s: 0.3, q2: sp.pi/3, L2: 0.4})[:3, 3])
print(num(PRets, {d1s: 0.3, q2: sp.pi/3, L2: 0.4}))

# ===== 7. spatial RRP (SCARA): ETS tz(H)Rz(q1)tx(a1)Rz(q2)tx(a2)tz(-d3)Rx(180) ; DH rows (0,0,H,q1)(0,a1,0,q2)(180,a2,d3,0) =====
SCets = chain(stz(H), sRz(q1), stx(a1), sRz(q2), stx(a2), stz(-d3s), sRx(sp.pi))
SCdh = chain(sDH_prox(0, 0, q1, H), sDH_prox(0, a1, q2, 0), sDH_prox(sp.pi, a2, 0, d3s))
check('SCARA RRP: DH (0,0,H,q1)(0,a1,0,q2)(180,a2,d3,0) == ETS', eqz(SCdh - SCets))
SChand = sp.Matrix([[c(q1+q2), s(q1+q2), 0, a1*c(q1)+a2*c(q1+q2)], [s(q1+q2), -c(q1+q2), 0, a1*s(q1)+a2*s(q1+q2)], [0, 0, -1, H-d3s], [0, 0, 0, 1]])
check('SCARA matrix == hand', eqz(SCets - SChand))
show('SCARA', SCets)
sub = {H: 0.5, a1: 0.4, a2: 0.3, q1: 40*sp.pi/180, q2: 50*sp.pi/180, d3s: 0.15}
Tsc = num(SCets, sub); print('SCARA num H=.5 a1=.4 a2=.3 q=(40,50) d3=.15\n', Tsc)
# mock 1 numbers
sub2 = {H: 0.4, a1: 0.35, a2: 0.25, q1: 30*sp.pi/180, q2: -60*sp.pi/180, d3s: 0.10}
Tm = num(SCets, sub2); print('MOCK1 Q1(b)\n', Tm)
close('mock1 Q1 matrix', Tm, [[0.8660, -0.5, 0, 0.5196], [-0.5, -0.8660, 0, 0.05], [0, 0, -1, 0.3], [0, 0, 0, 1]], tol=1e-4)
# QA SCARA 4-joint (rotation q4 about vertical, tool z up? their ETS: tz(H)Rz(q1)tx(L1)Rz(q2)tx(L2)tz(-d3)Rz(q4))
Hh, L1q, L2q = 0.40, 0.35, 0.25
qq4 = np.array([40, -70, 25])*deg
Tqa = nchain(tz(Hh), Rz(qq4[0]), tx(L1q), Rz(qq4[1]), tx(L2q), tz(-0.12), Rz(qq4[2]))
print('QA SCARA\n', Tqa)
close('QA SCARA pos (0.4846,0.1,0.28)', Tqa[:3, 3], [0.4846, 0.1, 0.28], tol=1e-4)

# ===== 8. Furuta =====
ETS = chain(stz(L0), sRz(q1), stx(L1), sRx(q2), stz(-L2), sRy(sp.pi))
T_B1 = stz(L0)*sRz(q1); T_12 = stx(L1)*sRx(q2); T_2E = stz(-L2)*sRy(sp.pi)
check('Furuta: product of the 3 intermediate transforms == ETS', eqz(T_B1*T_12*T_2E - ETS))
show('Furuta B_T_1', T_B1); show('Furuta 1_T_2', T_12); show('Furuta 2_T_E', T_2E); show('Furuta B_T_E', ETS)
Fhand = sp.Matrix([[-c(q1), -s(q1)*c(q2), -s(q1)*s(q2), L1*c(q1) - L2*s(q1)*s(q2)],
                   [-s(q1), c(q1)*c(q2), c(q1)*s(q2), L1*s(q1) + L2*c(q1)*s(q2)],
                   [0, s(q2), -c(q2), L0 - L2*c(q2)], [0, 0, 0, 1]])
check('Furuta ETS == hand matrix', eqz(ETS - Fhand))
DHhand = sp.Matrix([[-s(q1)*c(q2), c(q1), -s(q1)*s(q2), L1*c(q1) - L2*s(q1)*s(q2)],
                   [c(q1)*c(q2), s(q1), c(q1)*s(q2), L1*s(q1) + L2*c(q1)*s(q2)],
                   [s(q2), 0, -c(q2), L0 - L2*c(q2)], [0, 0, 0, 1]])
DHf = chain(sDH_prox(0, 0, q1 + sp.pi/2, L0), sDH_prox(sp.pi/2, 0, q2, L1), sDH_prox(sp.pi/2, 0, 0, L2))
check('Furuta DH * Rz(-90) == ETS', eqz(DHf*sRz(-sp.pi/2) - ETS))
check('Furuta 0T_3 (DH) == hand matrix', eqz(DHf - DHhand))
show('Furuta 0T_3 (DH)', DHf)
TfN = num(ETS, {L0: 0.5, L1: 0.4, L2: 0.3, q1: sp.pi/3, q2: sp.pi/4}); print('Furuta num\n', TfN)
etsr = rtb.ET.tz(0.5)*rtb.ET.Rz()*rtb.ET.tx(0.4)*rtb.ET.Rx()*rtb.ET.tz(-0.3)*rtb.ET.Ry(np.pi)
close('Furuta rtb', etsr.eval([np.pi/3, np.pi/4]), TfN, show=False)
# DH rows numeric intermediates (distinct frames)
print('Furuta B_T_1 at q1=60 L0=.5\n', num(T_B1, {L0: 0.5, q1: sp.pi/3}))
print('Furuta 1_T_2 at q2=45 L1=.4\n', num(T_12, {L1: 0.4, q2: sp.pi/4}))
print('Furuta 2_T_E L2=.3\n', num(T_2E, {L2: 0.3}))
# DH numeric: DH_prox rows, check 3T_E
print('3T_E=Rz(-90)\n', num(sRz(-sp.pi/2), {}))

# ===== 9. Puma-like (notebook Q5, own numbers Lb=1.0, Lu=0.5, Lf=0.3) =====
Lb, Lu, Lf = 1.0, 0.5, 0.3
def puma_ets(qv): return nchain(tz(Lb), Rz(qv[0]), Ry(qv[1]), tz(Lu), Ry(qv[2]), tz(Lf), Rz(qv[3]), Ry(qv[4]), Rz(qv[5]))
pr = [(0, 0, Lb, 0), (-90, 0, 0, -90), (0, Lu, 0, 90), (90, 0, Lf, 0), (-90, 0, 0, 0), (90, 0, 0, 0)]  # (alpha_{i-1}, a_{i-1}, d_i, theta offset)
def puma_dh(qv):
    T = np.eye(4)
    for (al, a, d, off), qi in zip(pr, qv): T = T @ DH_prox(al*deg, a, qi + off*deg, d)
    return T
rng = np.random.default_rng(3); ok = True
for _ in range(50):
    qv = rng.uniform(-np.pi, np.pi, 6); ok &= np.allclose(puma_dh(qv), puma_ets(qv), atol=1e-10)
check('PUMA-like: proximal DH table (own) == ETS of notebook Q5 for 50 random poses', ok)
print('PUMA home ETS\n', puma_ets(np.zeros(6)))
links = [rtb.RevoluteMDH(alpha=al*deg, a=a, d=d, offset=off*deg) for (al, a, d, off) in pr]
rob = rtb.DHRobot(links)
ok2 = True
for _ in range(20):
    qv = rng.uniform(-np.pi, np.pi, 6); ok2 &= np.allclose(rob.fkine(qv).A, puma_ets(qv), atol=1e-10)
check('PUMA-like: rtb RevoluteMDH(table) == ETS', ok2)
qtest = np.array([30, 20, -40, 0, 0, 0])*deg
print('PUMA at q=(30,20,-40,0,0,0) pos', puma_ets(qtest)[:3, 3])
# wrist centre = elbow + Lf f
phi_ = qtest[1]+qtest[2]
wc = np.array([np.cos(qtest[0])*(Lu*np.sin(qtest[1]) + Lf*np.sin(phi_)), np.sin(qtest[0])*(Lu*np.sin(qtest[1]) + Lf*np.sin(phi_)), Lb + Lu*np.cos(qtest[1]) + Lf*np.cos(phi_)])
close('PUMA wrist centre formula', puma_ets(qtest)[:3, 3], wc)
# slot: with q4..q6 changed, wrist centre unchanged
q_b = qtest.copy(); q_b[3:] = [1.0, -0.7, 2.0]
close('PUMA wrist centre indep of q4-6', puma_ets(q_b)[:3, 3], puma_ets(qtest)[:3, 3])
print('PUMA home orientation: ', puma_ets(np.zeros(6))[:3, :3])

# ===== 10. Panda (Franka published proximal DH; added) =====
a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]; dd = [0.333, 0, 0.316, 0, 0.384, 0, 0]
al = [0, -90, 90, 90, -90, 90, 90]
def panda(qv):
    T = np.eye(4)
    for i in range(7): T = T @ DH_prox(al[i]*deg, a[i], qv[i], dd[i])
    return T @ tz(0.107)
print('Panda q=0 flange T\n', panda(np.zeros(7)))
p = rtb.models.DH.Panda(); p.tool = smath.SE3()
ok = True
for _ in range(10):
    qv = rng.uniform(-np.pi, np.pi, 7); ok &= np.allclose(panda(qv)[:3, 3], p.fkine(qv).t, atol=1e-6)
check('Panda DH flange position == rtb built-in', ok)
# notebook cell 42 ETS vs Panda
nbets = (rtb.ET.tz(0.333)*rtb.ET.Rz()*rtb.ET.tx(0.316)*rtb.ET.Rx()*rtb.ET.tz(0.384)*rtb.ET.Rx()*rtb.ET.tz(0.107)*rtb.ET.Rx()*rtb.ET.tz(0.0825)*rtb.ET.Rx()*rtb.ET.tz(0.088)*rtb.ET.Rx(np.pi))
print('notebook cell 42 ETS at q=0 position:', nbets.eval(np.zeros(5))[:3, 3], ' (Franka DH flange (0.088,0,0.926); tool at 0.107 more)')
check('Panda: notebook cell-42 ETS is NOT the Panda (position at q=0 differs)', not np.allclose(nbets.eval(np.zeros(5))[:3, 3], panda(np.zeros(7))[:3, 3], atol=1e-3))
# Panda ETS from DH rows: each row Rx(al) tx(a) Rz(q) tz(d)
print('Panda link-transform ETS rows (alpha,a,d):', list(zip(al, a, dd)))
for i in range(7):
    print('   row', i+1, 'Rx(%d) tx(%.4f) Rz(q%d) tz(%.3f)' % (al[i], a[i], i+1, dd[i]))

# ===== 11. proximal vs distal =====
# distal rows for planar RR: (theta,d,a,alpha) = (q1,0,L1,0),(q2,0,L2,0). Verified above (RR distal)
# example from Q&A: DH table reading
Tq = nchain(DH_prox(0, 0, 30*deg, 0.3), DH_prox(90*deg, 0.2, -20*deg, 0.05))
print('QA reading DH: origin of {2} in {0}:', Tq[:3, 3]); close('QA DH origin', Tq[:3, 3], [0.1982, 0.0567, 0.3], tol=1e-4)
Td = nchain(sDH_dist and Rz(30*deg), tz(0.3), tx(0), Rx(0), Rz(-20*deg), tz(0.05), tx(0.2), Rx(90*deg))
print('same numbers in distal order:', Td[:3, 3])
# elbow arm (QA): rows (0,0,L0,q1)(90,0,0,q2)(0,L2,0,q3)(0,L3,0,0)
L0e, L2e, L3e = 0.45, 0.40, 0.30; qe = np.array([35, 50, -30])*deg
Te = nchain(DH_prox(0, 0, qe[0], L0e), DH_prox(np.pi/2, 0, qe[1], 0), DH_prox(0, L2e, qe[2], 0), DH_prox(0, L3e, 0, 0))
print('QA elbow arm', Te[:3, 3])
close('QA elbow arm pos (0.4415,0.3092,0.8590)', Te[:3, 3], [0.4415, 0.3092, 0.8590], tol=1e-4)


# ===== 11b. proximal -> distal conversion rule (shift a, alpha by one row) =====
DHfd = chain(sDH_dist(q1 + sp.pi/2, L0, 0, sp.pi/2), sDH_dist(q2, L1, 0, sp.pi/2), sDH_dist(0, L2, 0, 0))
check('Furuta: distal rows (q1+90,L0,0,90)(q2,L1,0,90)(0,L2,0,0) == proximal 0T_3', eqz(DHfd - DHf))
DHrd = chain(sDH_dist(q1, 0, L1, 0), sDH_dist(q2, 0, L2, 0), sDH_dist(q3, 0, L3, 0))
check('RRR distal (3 rows) == ETS', eqz(DHrd - T3s))
SCd = chain(sDH_dist(q1, H, a1, 0), sDH_dist(q2, 0, a2, sp.pi), sDH_dist(0, d3s, 0, 0))
check('SCARA distal rows (q1,H,a1,0)(q2,0,a2,180)(0,d3,0,0) == ETS', eqz(SCd - SCets))

# ===== 12. notebook Q3 two frames =====
R_true = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0.0]])
Tet = nchain(Tr(3, 5, 2), Ry(90*deg), Rx(-90*deg))
close('two frames matrix', Tet[:3, :3], R_true); print(Tet)
# alternative ETS: Rz(-90) then Rx(...)? equivalent forms
for nm, M in (('t Rz(-90) Ry(-90)?', nchain(Tr(3, 5, 2), Rz(-90*deg), Ry(-90*deg))),
              ('t Ry(90) Rz(90)?', nchain(Tr(3, 5, 2), Ry(90*deg), Rz(90*deg))),
              ('t Rx(90) Rz(90)?', nchain(Tr(3, 5, 2), Rx(90*deg), Rz(90*deg)))):
    print(nm, np.allclose(M[:3, :3], R_true))
# columns reading: xB=-zA, yB=-xA, zB=+yA
print('x_B', R_true[:, 0], 'y_B', R_true[:, 1], 'z_B', R_true[:, 2])
check('two frames: x_B x y_B = z_B', np.allclose(np.cross(R_true[:, 0], R_true[:, 1]), R_true[:, 2]))
Tnb = nchain(ty(5), tx(3), Ry(90*deg), Rx(90*deg)); check('two frames: notebook answer wrong', not np.allclose(Tnb, Tet)); print('nb answer\n', Tnb)

# ===== 13. W13 Q1 =====
t1, t2, t3 = sp.symbols('theta1 theta2 theta3', real=True)
LL1, LL2, LL3, LL4 = sp.symbols('L1 L2 L3 L4', positive=True)
Q1 = chain(sRz(t1), stz(LL1+LL2), sRy(-t2), stx(LL3), sRy(-t3), stx(LL4), sRx(sp.pi/2))
c23, s23 = c(t2+t3), s(t2+t3); rr = LL3*c(t2)+LL4*c23
Q1hand = sp.Matrix([[c(t1)*c23, -c(t1)*s23, s(t1), c(t1)*rr], [s(t1)*c23, -s(t1)*s23, -c(t1), s(t1)*rr], [s23, c23, 0, LL1+LL2+LL3*s(t2)+LL4*s23], [0, 0, 0, 1]])
check('W13 Q1 symbolic == hand matrix', eqz(Q1 - Q1hand))
TQ = num(Q1, {LL1: 0.6, LL2: 0.5, LL3: 0.4, LL4: 0.3, t1: 30*sp.pi/180, t2: 20*sp.pi/180, t3: 40*sp.pi/180}); print('Q1 num\n', TQ)
close('Q1 num', TQ, [[0.4330, -0.75, 0.5, 0.4554], [0.25, -0.4330, -0.8660, 0.2629], [0.8660, 0.5, 0, 1.4966], [0, 0, 0, 1]], tol=1e-4)
# intermediate numbers at theta=(30,20,40)
Tn = {LL1: 0.6, LL2: 0.5, LL3: 0.4, LL4: 0.3, t1: 30*sp.pi/180, t2: 20*sp.pi/180, t3: 40*sp.pi/180}
print('S_T_1\n', num(sRz(t1), Tn)); print('1_T_2\n', num(stz(LL1+LL2)*sRy(-t2), Tn)); print('2_T_3\n', num(stx(LL3)*sRy(-t3), Tn)); print('3_T_T\n', num(stx(LL4)*sRx(sp.pi/2), Tn))
# frame origins in {S}
for nm, M in (('{2} origin', chain(sRz(t1), stz(LL1+LL2), sRy(-t2))), ('{3} origin', chain(sRz(t1), stz(LL1+LL2), sRy(-t2), stx(LL3), sRy(-t3)))):
    print(nm, num(M, Tn)[:3, 3])
# W13 Q1 (notation per slide): with the same ETS, tool at theta=0:
print('Q1 theta=0 T\n', num(Q1, {LL1: 0.6, LL2: 0.5, LL3: 0.4, LL4: 0.3, t1: 0, t2: 0, t3: 0}))
# theta2 = 90: arm straight up
print('Q1 theta2=90, theta3=0 position', num(Q1, {LL1: 0.6, LL2: 0.5, LL3: 0.4, LL4: 0.3, t1: 0, t2: sp.pi/2, t3: 0})[:3, 3])

# ===== 14. 'given the transform, find the frame' scenario =====
Tg = np.array([[0, 0, 1, 0.4], [1, 0, 0, 0.2], [0, 1, 0, 0.5], [0, 0, 0, 1.0]])
check('given T is a proper pose', is_rotation(Tg[:3, :3]))
print('given T: x_B=', Tg[:3, 0], 'y_B=', Tg[:3, 1], 'z_B=', Tg[:3, 2])
# as ETS: tz(0.5) tx(0.4)... any decomposition:  R = Rz(90) Rx(90)?  columns (0,1,0),(0,0,1),(1,0,0)  -> x_B=y_A
Tg2 = nchain(Tr(0.4, 0.2, 0.5), Ry(90*deg), Rz(90*deg))
close('given T == t(0.4,0.2,0.5) Ry(90) Rz(90)', Tg2, Tg)
close('given T == t(0.4,0.2,0.5) Rx(90) Ry(90)', nchain(Tr(0.4, 0.2, 0.5), Rx(90*deg), Ry(90*deg)), Tg)
# point (0.1,0,0)_B ->
print('P_B=(0,0,0.1) ->', (Tg @ np.array([0, 0, 0.1, 1]))[:3])

# ===== 15. camera chain and 'frame is not the tool' (Practice Q&A W2) =====
BTW = Tr(0.5, 0.2, 0.6) @ Rz(90*deg); WTC = Tr(0.05, 0, 0.1) @ Ry(90*deg)
PC = np.array([0, 0, 0.8, 1.0])
PB = BTW @ WTC @ PC
print('camera: P in W', (WTC @ PC)[:3], ' P in B', PB[:3], ' camera origin in B', (BTW @ WTC)[:3, 3], ' wrong order', (WTC @ BTW @ PC)[:3])
close('camera P_B = (0.50,1.05,0.70)', PB[:3], [0.5, 1.05, 0.7])
close('camera wrong order (1.45,0.2,-0.4)', (WTC @ BTW @ PC)[:3], [1.45, 0.2, -0.4])
q12 = np.array([30, 45])*deg
T02 = nchain(DH_prox(0, 0, q12[0], 0), DH_prox(0, 0.5, q12[1], 0))
Ttip = T02 @ tx(0.3)
print('spot error: origin 0T2', T02[:3, 3], ' tip', Ttip[:3, 3])
close('spot-error tip (0.5107,0.5398)', Ttip[:2, 3], [0.5107, 0.5398], tol=1e-4)
# rotation facts used in text
print('Ry(90) z ->', (Ry(90*deg)[:3, :3] @ [0, 0, 1]).round(3), ' Rz(90) x ->', (Rz(90*deg)[:3, :3] @ [1, 0, 0]).round(3))
# example in 'given T find frame': check P_B -> A

TgI = np.eye(4); TgI[:3,:3]=Tg[:3,:3].T; TgI[:3,3]=-Tg[:3,:3].T@Tg[:3,3]
print('given T inverse t =', TgI[:3,3], ' R^T=\n', TgI[:3,:3])
close('given T: inverse', TgI, np.linalg.inv(Tg))
close('given T: -R^T t = (-0.2,-0.5,-0.4)', TgI[:3,3], [-0.2,-0.5,-0.4])
close('given T: P_A = (0.5,0.2,0.5)', (Tg@np.array([0,0,0.1,1]))[:3], [0.5,0.2,0.5])
report()
