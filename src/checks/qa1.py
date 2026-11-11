"""Practice Q&A part 1 (W1-W6, 36 questions): verifies every number used in frag/qa/part1.html.
Run: python3 site/src/checks/qa1.py   (prints 'ANS <id> <label> = value' for each answer box, then ALL CHECKS PASSED)
Also: python3 checks/qa1.py --html  cross-checks every input[data-ans] in frag/qa/part1.html against the ANS registry."""
import os, sys, re
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REG = {}
def ans(qid, label, val):
    REG[(qid, label)] = float(val)
    print(f'ANS {qid} {label} = {float(val):.6g}')

def gk(D, N, J, SF, RC=0, RM=0):
    return D * (N - J - 1) + SF + RC - RM

def d(x): return np.degrees(x)

# =====================================================================  W1
print('--- W1')
# Q1 Watt six-bar: 6 links incl ground, 7 revolutes, planar
M = gk(3, 6, 7, 7); check('w1q1 Watt six-bar M=1', M == 1); ans('qa-w01-1', 'M', M)
check('w1q1 wrong D=6 gives -5', gk(6, 6, 7, 7) == -5)
check('w1q1 form 1: 3(N-1)-2*P1', 3 * (6 - 1) - 2 * 7 == 1)

# Q2 planar 3R unequal links: reachable and dexterous workspace
L1, L2, L3 = 0.5, 0.3, 0.2
h, H = abs(L1 - L2), L1 + L2
check('w1q2 annulus 0.2..0.8', np.isclose(h, 0.2) and np.isclose(H, 0.8))
def tip_reachable(r, phi):
    p = np.array([r, 0.0]); w = p - L3 * np.array([np.cos(phi), np.sin(phi)])
    n = np.linalg.norm(w); return h - 1e-12 <= n <= H + 1e-12
phis = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
def dexterous(r): return all(tip_reachable(r, f) for f in phis)
def reachable(r): return any(tip_reachable(r, f) for f in phis)
check('w1q2 r=0.39 not dexterous', not dexterous(0.39))
check('w1q2 r=0.401 dexterous', dexterous(0.401))
check('w1q2 r=0.599 dexterous', dexterous(0.599))
check('w1q2 r=0.61 not dexterous', not dexterous(0.61))
check('w1q2 r=0.3 not dexterous (circle crosses hole)', not dexterous(0.3))
check('w1q2 r=0.0 dexterous (wrist circle radius 0.2 sits on hole edge)', dexterous(0.0))
check('w1q2 r=0.999 reachable, 1.001 not', reachable(0.999) and not reachable(1.001))
rin = L3 + h   # inner: r - L3 >= h
rout = H - L3
check('w1q2 inner 0.4, outer 0.6, reach 1.0', np.isclose(rin, 0.4) and np.isclose(rout, 0.6) and np.isclose(L1 + L2 + L3, 1.0))
ans('qa-w01-2', 'r_in', rin); ans('qa-w01-2', 'r_out', rout); ans('qa-w01-2', 'r_reach', L1 + L2 + L3)

# Q3 four RPR legs
N = 2 + 4 * 2; J = 4 * 3; M = gk(3, N, J, J)
check('w1q3 4-RPR N=10 J=12 M=3', (N, J, M) == (10, 12, 3))
ans('qa-w01-3', 'M', M); ans('qa-w01-3', 'extra', 4 - M)
check('w1q3 wrong N=9 gives 0', gk(3, 9, 12, 12) == 0)

# Q4 RSSR spatial four-bar
M = gk(6, 4, 4, 1 + 3 + 3 + 1); check('w1q4 RSSR plain=2', M == 2)
ans('qa-w01-4', 'plain', M); ans('qa-w01-4', 'true', M - 1)
check('w1q4 planar D=3 would give 3(4-4-1)+8=5', gk(3, 4, 4, 8) == 5)

# Q5 two 3-joint fingers, one motor
N = 1 + 6; J = 6; M = gk(3, N, J, 6); check('w1q5 M=6', M == 6); ans('qa-w01-5', 'uncontrolled', M - 1)

# Q6 five-bar spot the error
M = gk(3, 5, 5, 5); check('w1q6 five-bar M=2', M == 2); ans('qa-w01-6', 'M', M)
check('w1q6 student N=4 gives -1', gk(3, 4, 5, 5) == -1)

# =====================================================================  W2
print('--- W2')
# Q1 SCARA ETS: tz(H) Rz(q1) tx(L1) Rz(q2) tx(L2) tz(-d3) Rz(q4)
H0, A1, A2 = 0.4, 0.35, 0.25
q1, q2, d3, q4 = 40 * deg, -70 * deg, 0.12, 25 * deg
T = nchain(tz(H0), Rz(q1), tx(A1), Rz(q2), tx(A2), tz(-d3), Rz(q4))
x = A1 * np.cos(q1) + A2 * np.cos(q1 + q2); y = A1 * np.sin(q1) + A2 * np.sin(q1 + q2); z = H0 - d3
close('w2q1 position', T[:3, 3], [x, y, z])
yaw = np.arctan2(T[1, 0], T[0, 0]); close('w2q1 yaw -5 deg', d(yaw), -5.0)
ans('qa-w02-1', 'x', x); ans('qa-w02-1', 'y', y); ans('qa-w02-1', 'z', z)
check('w2q1 wrong z=0.52', np.isclose(H0 + d3, 0.52))
# value used in W3 Q1 (same arm): reachable target
# Q2 elbow arm proximal DH
L0, B2, B3 = 0.45, 0.40, 0.30
qq = np.array([35, 50, -30]) * deg
Td = nchain(DH_prox(0, 0, qq[0], L0), DH_prox(np.pi / 2, 0, qq[1], 0), DH_prox(0, B2, qq[2], 0), tx(B3))
Te = nchain(tz(L0), Rz(qq[0]), Rx(np.pi / 2), Rz(qq[1]), tx(B2), Rz(qq[2]), tx(B3))
close('w2q2 DH = ETS', Td, Te)
r = B2 * np.cos(qq[1]) + B3 * np.cos(qq[1] + qq[2]); zz = L0 + B2 * np.sin(qq[1]) + B3 * np.sin(qq[1] + qq[2])
close('w2q2 closed form', Td[:3, 3], [r * np.cos(qq[0]), r * np.sin(qq[0]), zz])
ans('qa-w02-2', 'x', Td[0, 3]); ans('qa-w02-2', 'y', Td[1, 3]); ans('qa-w02-2', 'z', Td[2, 3])
print('  r, z =', r, zz)
# Q3 camera
TBW = nchain(Tr(0.5, 0.2, 0.6), Rz(np.pi / 2)); TWC = nchain(Tr(0.05, 0, 0.1), Ry(np.pi / 2))
pC = np.array([0, 0, 0.8, 1.0]); pW = TWC @ pC; pB = TBW @ TWC @ pC
close('w2q3 pW', pW[:3], [0.85, 0, 0.1]); close('w2q3 pB', pB[:3], [0.5, 1.05, 0.7])
camB = (TBW @ TWC)[:3, 3]; close('w2q3 camera origin in B', camB, [0.5, 0.25, 0.7])
wrong = (TWC @ TBW @ pC)[:3]
print('  wrong order TWC*TBW*p =', wrong)
ans('qa-w02-3', 'x', pB[0]); ans('qa-w02-3', 'y', pB[1]); ans('qa-w02-3', 'z', pB[2])
ans('qa-w02-3', 'wrong_x', wrong[0])
# Q4 proximal DH chain position
rows = [(0, 0, 30 * deg, 0.3), (0.2, 0, 0, 0)]
T01 = DH_prox(0, 0, 30 * deg, 0.3); T12 = DH_prox(np.pi / 2, 0.2, -20 * deg, 0.05)
# NB DH_prox(alpha_im1, a_im1, theta_i, d_i)
T02 = T01 @ T12
print('  T02 position =', T02[:3, 3])
ans('qa-w02-4', 'x', T02[0, 3]); ans('qa-w02-4', 'y', T02[1, 3]); ans('qa-w02-4', 'z', T02[2, 3])
# single-row position (alpha=-90, a=0.3,d=0.1, any theta): (0.3, 0.1, 0)
for th in (0, 40 * deg, 100 * deg):
    close(f'w2q4 single-row origin independent of theta ({th:.2f})', DH_prox(-np.pi / 2, 0.3, th, 0.1)[:3, 3], [0.3, 0.1, 0.0])
# distal misread: the same four numbers used with the distal order for each row (a_{i-1} read as a_i)
Td1 = sDH_dist(30 * deg, 0.3, 0.0, 0.0); Td2 = sDH_dist(-20 * deg, 0.05, 0.2, np.pi / 2)
Tm = np.array((Td1 * Td2).evalf().tolist(), dtype=float)
print('  distal misread position =', Tm[:3, 3])
ans('qa-w02-4', 'wrong_y', Tm[1, 3]); ans('qa-w02-4', 'wrong_z', Tm[2, 3])
# also a 'forgot the twist' check: position of T12 alone
print('  T12 origin =', T12[:3, 3])
# Q5 order
P1 = nchain(Rz(np.pi / 2), tx(1)); P2 = nchain(tx(1), Rz(np.pi / 2))
close('w2q5 Rz(90)tx(1) origin (0,1,0)', P1[:3, 3], [0, 1, 0]); close('w2q5 tx(1)Rz(90) origin (1,0,0)', P2[:3, 3], [1, 0, 0])
close('w2q5 same orientation', P1[:3, :3], P2[:3, :3])
ans('qa-w02-5', 'y', P1[1, 3])
# Q6 spot the error: proximal frame {2} is at the elbow
Lq1, Lq2 = 0.5, 0.3; a, b = 30 * deg, 45 * deg
T2 = nchain(DH_prox(0, 0, a, 0), DH_prox(0, Lq1, b, 0))
tip = nchain(T2, tx(Lq2))
close('w2q6 frame 2 origin = elbow? (L1 c1 + L1 c12?)', T2[:3, 3], [Lq1 * np.cos(a), Lq1 * np.sin(a), 0])
close('w2q6 tip', tip[:3, 3], [Lq1 * np.cos(a) + Lq2 * np.cos(a + b), Lq1 * np.sin(a) + Lq2 * np.sin(a + b), 0])
print('  frame-2 origin', T2[:2, 3], ' tip', tip[:2, 3])
ans('qa-w02-6', 'x', tip[0, 3]); ans('qa-w02-6', 'y', tip[1, 3])

# =====================================================================  W3
print('--- W3')
def planar_ik(x, y, l1, l2):
    c2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    out = []
    for s in (+1, -1):
        q2 = np.arctan2(s * np.sqrt(1 - c2 ** 2), c2)
        q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        out.append((q1, q2))
    return c2, out
# Q1 SCARA IK
tx_, ty_, tz_, phi = 0.30, 0.25, 0.28, 60 * deg
c2, sols = planar_ik(tx_, ty_, A1, A2)
print('  cos q2 =', c2)
(q1p, q2p), (q1m, q2m) = sols
d3s = H0 - tz_;
for (qa, qb) in sols:
    q4s = phi - qa - qb
    Tk = nchain(tz(H0), Rz(qa), tx(A1), Rz(qb), tx(A2), tz(-d3s), Rz(q4s))
    close(f'w3q1 FK reproduces target (q2={d(qb):.3f})', [Tk[0, 3], Tk[1, 3], Tk[2, 3], np.arctan2(Tk[1, 0], Tk[0, 0])], [tx_, ty_, tz_, phi])
print('  q2+ =', d(q2p), 'q1+ =', d(q1p), 'q4+ =', d(phi - q1p - q2p), '| q2- =', d(q2m), 'q1- =', d(q1m), 'q4- =', d(phi - q1m - q2m))
ans('qa-w03-1', 'q2', d(q2p)); ans('qa-w03-1', 'q1', d(q1p)); ans('qa-w03-1', 'd3', d3s); ans('qa-w03-1', 'q4', d(phi - q1p - q2p))
ans('qa-w03-1', 'wrong_q1', d(q1m))
check('w3q1 reachable: 0.1 <= r <= 0.6', 0.1 <= np.hypot(tx_, ty_) <= 0.6)
# Q2 wrist centre
Hs, Gs, Fs, d6 = 0.30, 0.35, 0.30, 0.10
pE = np.array([0.55, 0.20, 0.65]); RE = (Rz(20 * deg) @ Ry(60 * deg))[:3, :3]
zE = RE[:, 2]; pW = pE - d6 * zE
print('  approach axis', zE, ' pW', pW)
q1w = np.arctan2(pW[1], pW[0]); rW = np.hypot(pW[0], pW[1])
c3 = (rW ** 2 + (pW[2] - Hs) ** 2 - Fs ** 2 - Gs ** 2) / (2 * Fs * Gs)
q3w = -np.arccos(c3)
sg = np.arctan2(pW[2] - Hs, rW); mu = np.arctan2(Fs * np.sin(q3w), Gs + Fs * np.cos(q3w)); q2w = sg - mu
print('  q1,q3,q2 =', d(q1w), d(q3w), d(q2w), 'cos q3', c3)
check('w3q2 FK of planar part reproduces wrist centre',
      np.allclose([Gs * np.cos(q2w) + Fs * np.cos(q2w + q3w), Hs + Gs * np.sin(q2w) + Fs * np.sin(q2w + q3w)], [rW, pW[2]]))
# also full check with wrist-centre ETS: tz(H) Rz(q1) Ry(-q2)... use planar FK only (done)
ans('qa-w03-2', 'pWx', pW[0]); ans('qa-w03-2', 'q1', d(q1w)); ans('qa-w03-2', 'q3', d(q3w)); ans('qa-w03-2', 'q2', d(q2w))
q3o = +np.arccos(c3); mu2 = np.arctan2(Fs * np.sin(q3o), Gs + Fs * np.cos(q3o)); q2o = sg - mu2
print('  elbow-down q3, q2 =', d(q3o), d(q2o))
ans('qa-w03-2', 'wrong_q2', d(q2o))
# Q3 joint limits
l1, l2 = 0.6, 0.45
c2, sols = planar_ik(0.7, -0.3, l1, l2)
print('  cos q2 =', c2, ' branches', [(d(a), d(b)) for a, b in sols])
(q1a, q2a), (q1b, q2b) = sols
check('w3q3 only q2>0 branch inside [0,150]', 0 <= d(q2a) <= 150 and not (0 <= d(q2b) <= 150))
ans('qa-w03-3', 'q1', d(q1a)); ans('qa-w03-3', 'q2', d(q2a)); ans('qa-w03-3', 'wrong_q1', d(q1b))
for qa, qb in sols:
    close('w3q3 FK', [l1 * np.cos(qa) + l2 * np.cos(qa + qb), l1 * np.sin(qa) + l2 * np.sin(qa + qb)], [0.7, -0.3])
# Q4 Newton step
def fk2(q): return np.array([l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]), l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])])
def jac2(q):
    s1, c1, s12, c12 = np.sin(q[0]), np.cos(q[0]), np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
    return np.array([[-l1 * s1 - l2 * s12, -l2 * s12], [l1 * c1 + l2 * c12, l2 * c12]])
q0 = np.array([0.3, 0.8]); tgt = np.array([0.7, 0.3])
e = tgt - fk2(q0); Jn = jac2(q0); dq = np.linalg.solve(Jn, e)
print('  FK0', fk2(q0), 'e', e, 'J', Jn.tolist(), 'det', np.linalg.det(Jn), 'dq', dq)
close('w3q4 det = L1L2 sin q2', np.linalg.det(Jn), l1 * l2 * np.sin(0.8))
r1 = np.linalg.norm(tgt - fk2(q0 + dq)); print('  residual after step', r1, 'before', np.linalg.norm(e))
dqT = Jn.T @ e; print('  J^T e =', dqT)
ans('qa-w03-4', 'dq1', dq[0]); ans('qa-w03-4', 'dq2', dq[1]); ans('qa-w03-4', 'res', r1); ans('qa-w03-4', 'wrong_dq1', dqT[0])
# Q5 count of solutions at the boundary
c2b, _ = planar_ik(l1 + l2, 0, l1, l2); check('w3q5 outer boundary cos q2 = 1', np.isclose(c2b, 1))
c2c, _ = planar_ik(0.0, 0.0, 0.5, 0.5); check('w3q5 origin, L1=L2: cos q2 = -1', np.isclose(c2c, -1))
ans('qa-w03-5', 'n', 1)
# Q6 spot the error
l1s, l2s = 0.5, 0.4; xs, ys = -0.45, 0.30
c2, sols = planar_ik(xs, ys, l1s, l2s)
q1c, q2c = sols[0]
mu_s = np.arctan2(l2s * np.sin(q2c), l1s + l2s * np.cos(q2c))
q1bad = np.arctan(ys / xs) - mu_s
print('  cos q2 =', c2, 'q2 =', d(q2c), 'q1 correct', d(q1c), 'student', d(q1bad), 'mu', d(mu_s))
close('w3q6 FK of correct', [l1s * np.cos(q1c) + l2s * np.cos(q1c + q2c), l1s * np.sin(q1c) + l2s * np.sin(q1c + q2c)], [xs, ys])
bx = l1s * np.cos(q1bad) + l2s * np.cos(q1bad + q2c); by = l1s * np.sin(q1bad) + l2s * np.sin(q1bad + q2c)
print('  student lands at', bx, by)
ans('qa-w03-6', 'q1', d(q1c)); ans('qa-w03-6', 'wrong_q1', d(q1bad)); ans('qa-w03-6', 'q2', d(q2c))
print('  atan(y/x) =', d(np.arctan(ys / xs)), 'atan2 =', d(np.arctan2(ys, xs)))

# =====================================================================  W4
print('--- W4')
# Q1 3-RPR leg lengths
b = np.array([[0, 0], [1.0, 0], [0.5, 0.9]]); rho = 0.2
a_loc = np.array([[rho * np.cos(np.radians(t)), rho * np.sin(np.radians(t))] for t in (0, 120, 240)])
px, py, ph = 0.5, 0.35, 10 * deg
R2 = np.array([[np.cos(ph), -np.sin(ph)], [np.sin(ph), np.cos(ph)]])
P = np.array([px, py]) + a_loc @ R2.T
legs = np.linalg.norm(P - b, axis=1)
print('  platform points', P.round(4).tolist(), ' legs', legs)
check('w4q1 mobility 3', gk(3, 8, 9, 9) == 3)
for i in range(3): ans('qa-w04-1', f'l{i+1}', legs[i])
wrongl = np.linalg.norm((np.array([px, py]) + a_loc) - b, axis=1)  # forgetting the rotation
print('  forgetting rotation:', wrongl); ans('qa-w04-1', 'wrong_l1', wrongl[0])
# Q2 four-bar
L0f, L1f, L2f, L3f = 0.8, 0.25, 0.6, 0.5
check('w4q2 Grashof: s+l <= p+q', min(L0f, L1f, L2f, L3f) + max(L0f, L1f, L2f, L3f) <= (L0f + L1f + L2f + L3f) - min(L0f, L1f, L2f, L3f) - max(L0f, L1f, L2f, L3f))
th1 = 60 * deg
A = L1f * np.array([np.cos(th1), np.sin(th1)]); D = np.array([L0f, 0.0])
AD = np.linalg.norm(D - A)
u = (D - A) / AD; nrm = np.array([-u[1], u[0]])
xa = (L2f ** 2 - L3f ** 2 + AD ** 2) / (2 * AD); hh = np.sqrt(L2f ** 2 - xa ** 2)
Bs = [A + xa * u + s * hh * nrm for s in (+1, -1)]
for Bp in Bs:
    close('w4q2 circle distances', [np.linalg.norm(Bp - A), np.linalg.norm(Bp - D)], [L2f, L3f])
Bup = max(Bs, key=lambda p: p[1]); Bdn = min(Bs, key=lambda p: p[1])
phi1 = lambda Bp: np.arctan2(Bp[1] - D[1], Bp[0] - D[0])
print('  |AD| =', AD, 'xa', xa, 'h', hh, ' B up', Bup, 'phi1', d(phi1(Bup)), ' B down', Bdn, 'phi1', d(phi1(Bdn)))
phiC = lambda Bp: np.arctan2(Bp[1] - A[1], Bp[0] - A[0])
print('  coupler angle up', d(phiC(Bup)), 'down', d(phiC(Bdn)))
ans('qa-w04-2', 'AD', AD); ans('qa-w04-2', 'phi1', d(phi1(Bup))); ans('qa-w04-2', 'phi1_down', d(phi1(Bdn)))
# crank range
ADmin, ADmax = L0f - L1f, L0f + L1f
check('w4q2 crank turns fully: |L2-L3| <= AD range <= L2+L3', abs(L2f - L3f) <= ADmin and ADmax <= L2f + L3f)
print('  AD range', ADmin, ADmax, ' L2-L3', abs(L2f - L3f), 'L2+L3', L2f + L3f)
# non-Grashof variant for the explanation: crank 0.45
L1x = 0.45; ADmax_x = L0f + L1x
print('  with L1=0.45: AD max', ADmax_x, ' > L2+L3 =', L2f + L3f, '->', ADmax_x > L2f + L3f)
check('w4q2 L1=0.45 breaks assembly at theta1=0', ADmax_x > L2f + L3f)
# Q3 3-RPS / 3-SPS
M_rps = gk(6, 8, 9, 3 * (1 + 1 + 3)); M_sps = gk(6, 8, 9, 3 * (3 + 1 + 3))
check('w4q3 3-RPS plain 3, 3-SPS plain 9, true 6', M_rps == 3 and M_sps == 9 and gk(6, 8, 9, 21, 0, 3) == 6)
ans('qa-w04-3', 'rps', M_rps); ans('qa-w04-3', 'sps', M_sps); ans('qa-w04-3', 'sps_true', M_sps - 3)
# Q4 Stewart
bb = np.array([[np.cos(np.radians(t)), np.sin(np.radians(t)), 0] for t in (-10, 10, 110, 130, 230, 250)])
pp = 0.5 * np.array([[np.cos(np.radians(t)), np.sin(np.radians(t)), 0] for t in (-50, 50, 70, 170, 190, 290)])
home = np.linalg.norm(pp + np.array([0, 0, 1.0]) - bb, axis=1); close('w4q4 home legs 1.2182', home, [1.2182] * 6, tol=5e-5)
Rr = Rz(10 * deg)[:3, :3]; tt = np.array([0, 0, 0.9])
lg = np.linalg.norm(tt + pp @ Rr.T - bb, axis=1); print('  legs', lg)
ans('qa-w04-4', 'd2', lg[1]); ans('qa-w04-4', 'd5', lg[4]); ans('qa-w04-4', 'home1', home[0])
check('w4q4 stroke range', lg.max() - lg.min() > 0)
print('  leg1 vector', (tt + Rr @ pp[0] - bb[0]))
# Q6 simplest delta
check('w4q6 correct M=3', gk(6, 8, 9, 15) == 3)
check('w4q6 student: +1 typo gives 15, minus 6 gives 9', gk(6, 8, 9, 15) + 12 == 15 and 15 - 6 == 9)
ans('qa-w04-6', 'M', 3)

# =====================================================================  W5
print('--- W5')
t_, A0, A1_, A2_, A3_, Dl, Tt = sp.symbols('t a0 a1 a2 a3 Delta T')
qs = A0 + A1_ * t_ + A2_ * t_ ** 2 + A3_ * t_ ** 3
sol = sp.solve([qs.subs(t_, 0), A1_, sp.diff(qs, t_, 2).subs(t_, Tt), qs.subs(t_, Tt) - Dl], [A0, A1_, A2_, A3_], dict=True)[0]
print('  symbolic a2, a3:', sp.simplify(sol[A2_]), sp.simplify(sol[A3_]))
check('w5q1 a2 = 3D/(2T^2), a3 = -D/(2T^3)', sp.simplify(sol[A2_] - 3 * Dl / (2 * Tt ** 2)) == 0 and sp.simplify(sol[A3_] + Dl / (2 * Tt ** 3)) == 0)
D5, T5 = 30.0, 2.0
a2_, a3_ = 3 * D5 / (2 * T5 ** 2), -D5 / (2 * T5 ** 3)
vend = 2 * a2_ * T5 + 3 * a3_ * T5 ** 2
check('w5q1 numbers', np.isclose(a2_, 11.25) and np.isclose(a3_, -1.875) and np.isclose(vend, 22.5) and np.isclose(a2_ * T5 ** 2 + a3_ * T5 ** 3, 30))
check('w5q1 end accel zero', np.isclose(2 * a2_ + 6 * a3_ * T5, 0))
ts = np.linspace(0, T5, 2001); vv = 2 * a2_ * ts + 3 * a3_ * ts ** 2
check('w5q1 velocity increases monotonically to the end', np.all(np.diff(vv) >= -1e-12) and np.isclose(vv[-1], 22.5))
ans('qa-w05-1', 'a2', a2_); ans('qa-w05-1', 'a3', a3_); ans('qa-w05-1', 'vend', vend)
ans('qa-w05-1', 'wrong_a2', 3 * D5 / T5 ** 2)  # rest-to-rest value 22.5
print('  rest-to-rest a2 would be', 3 * D5 / T5 ** 2, ' a3', -2 * D5 / T5 ** 3)
# Q2 time-optimal trapezoid
vm, am = 60.0, 120.0
def tmin(Dq):
    if Dq >= vm ** 2 / am: return Dq / vm + vm / am, vm
    return 2 * np.sqrt(Dq / am), np.sqrt(Dq * am)
T90, pk90 = tmin(90.0); T20, pk20 = tmin(20.0)
print('  T90', T90, pk90, ' T20', T20, pk20, ' v^2/a', vm ** 2 / am)
check('w5q2 T90=2.0', np.isclose(T90, 2.0)); check('w5q2 T20', np.isclose(T20, 0.816497, atol=1e-5))
# cross check with LSPB formula: for T=2, a=120, D=90 -> v = 60
tB = 2 / 2 - np.sqrt(am ** 2 * 4 - 4 * am * 90) / (2 * am); check('w5q2 LSPB cross-check tB=0.5, v=60', np.isclose(tB, 0.5) and np.isclose(am * tB, 60))
ans('qa-w05-2', 'T90', T90); ans('qa-w05-2', 'T20', T20); ans('qa-w05-2', 'pk20', pk20)
print('  naive Delta/v for 20 deg =', 20 / vm + vm / am)
ans('qa-w05-2', 'wrong_T20', 20 / vm + vm / am)
# Q3 via rule
tt5 = np.array([0, 2, 5, 9, 12.]); qq5 = np.array([10, 30, 70, 80, 60.])
s = np.diff(qq5) / np.diff(tt5); v = np.zeros(5)
for i in range(1, 4): v[i] = 0 if s[i - 1] * s[i] < 0 else (s[i - 1] + s[i]) / 2
print('  slopes', s, 'v', v)
k = 2; Tk = tt5[k + 1] - tt5[k]; Dk = qq5[k + 1] - qq5[k]
a2k = (3 * Dk - (2 * v[k] + v[k + 1]) * Tk) / Tk ** 2; a3k = (-2 * Dk + (v[k] + v[k + 1]) * Tk) / Tk ** 3
print('  segment 5->9: a0', qq5[k], 'a1', v[k], 'a2', a2k, 'a3', a3k)
check('w5q3 segment ends at 80 with v=0', np.isclose(qq5[k] + v[k] * Tk + a2k * Tk ** 2 + a3k * Tk ** 3, 80) and np.isclose(v[k] + 2 * a2k * Tk + 3 * a3k * Tk ** 2, 0))
check('w5q3 v at t=9 is zero (sign change)', v[3] == 0 and s[2] * s[3] < 0)
ans('qa-w05-3', 'v5', v[2]); ans('qa-w05-3', 'a2', a2k); ans('qa-w05-3', 'a3', a3k); ans('qa-w05-3', 'v2', v[1])
print('  naive average of the two times-uneven slopes using index average (13.333+2.5)/2 =', (s[1] + s[2]) / 2, ' vs mean of times-ignoring?', (30 - 10) / 2)
# acceleration at t=5: end of seg 2->... compute jump
k1 = 1; T1 = tt5[2] - tt5[1]; D1 = qq5[2] - qq5[1]
a2p = (3 * D1 - (2 * v[1] + v[2]) * T1) / T1 ** 2; a3p = (-2 * D1 + (v[1] + v[2]) * T1) / T1 ** 3
acc_end = 2 * a2p + 6 * a3p * T1; acc_start = 2 * a2k
print('  accel jump at t=5:', acc_end, acc_start)
ans('qa-w05-3', 'jump', acc_start - acc_end)
# Q4 LSPB
Dq, Tf, aa = 75.0, 1.5, 200.0
amin = 4 * Dq / Tf ** 2; disc = aa ** 2 * Tf ** 2 - 4 * aa * Dq
tb = Tf / 2 - np.sqrt(disc) / (2 * aa); vq = aa * tb
print('  a_min', amin, 'disc', disc, 'tB', tb, 'v', vq, 'coast', Tf - 2 * tb)
check('w5q4 distance', np.isclose(aa * tb ** 2 + vq * (Tf - 2 * tb), Dq))
check('w5q4 a=130 is below minimum', 130 < amin and 130 ** 2 * Tf ** 2 - 4 * 130 * Dq < 0)
tbp = Tf / 2 + np.sqrt(disc) / (2 * aa); print('  + root', tbp, 'v', aa * tbp)
ans('qa-w05-4', 'tB', tb); ans('qa-w05-4', 'v', vq); ans('qa-w05-4', 'amin', amin); ans('qa-w05-4', 'wrong_tB', tbp)
# Q5 peak speeds
D45, T45 = 45.0, 1.5
pc = 1.5 * D45 / T45; pq = 1.875 * D45 / T45
# verify with quintic coefficient
a3q, a4q, a5q = 10 * D45 / T45 ** 3, -15 * D45 / T45 ** 4, 6 * D45 / T45 ** 5
tz_ = np.linspace(0, T45, 20001); vq5 = 3 * a3q * tz_ ** 2 + 4 * a4q * tz_ ** 3 + 5 * a5q * tz_ ** 4
check('w5q5 quintic peak = 1.875 D/T', np.isclose(vq5.max(), pq, rtol=1e-6) and np.isclose(pq, 56.25))
check('w5q5 cubic peak = 45', np.isclose(pc, 45))
aq = 6 * D45 / T45 ** 2 ; print('  cubic end accel', aq, ' quintic peak accel', 10 * D45 / (np.sqrt(3) * T45 ** 2))
ans('qa-w05-5', 'cubic', pc); ans('qa-w05-5', 'quintic', pq); ans('qa-w05-5', 'cubic_acc', aq)
print('  LSPB with a = 2*amin: tB', None)
am5 = 4 * D45 / T45 ** 2; a_l = 2 * am5; tb5 = T45 / 2 - np.sqrt(a_l ** 2 * T45 ** 2 - 4 * a_l * D45) / (2 * a_l); print('  amin', am5, 'a=2amin', a_l, 'v', a_l * tb5)
# Q6 spot
qa0, qav, qaf, Tq = 0.0, 40.0, 60.0, 2.0
wv = 3 * (qaf - qa0) / (4 * Tq); wbad = 3 * (qaf - qa0) / (4 * (2 * Tq))
close('w5q6 via velocity 22.5', wv, 22.5)
# check against the general formula and two cubics
T1_ = T2_ = Tq; d1 = qav - qa0; d2 = qaf - qav
wgen = 3 * (T1_ ** 2 * d2 + T2_ ** 2 * d1) / (2 * T1_ * T2_ * (T1_ + T2_)); close('w5q6 general formula', wgen, wv)
acc1 = (-6 * d1 + 4 * wv * T1_) / T1_ ** 2; acc2 = (6 * d2 - 4 * wv * T2_) / T2_ ** 2
check('w5q6 accelerations match at via point', np.isclose(acc1, acc2)); print('  via accel', acc1, 'wrong w', wbad)
ans('qa-w05-6', 'w', wv); ans('qa-w05-6', 'wrong_w', wbad); ans('qa-w05-6', 'acc', acc1)

# =====================================================================  W6
print('--- W6')
# Q1 planar 3R
Lc = np.array([0.4, 0.3, 0.2]); qc = np.array([20, 40, -30]) * deg
def fk3(q):
    c = np.cumsum(q); return np.array([np.sum(Lc * np.cos(c)), np.sum(Lc * np.sin(c))])
def J3(q):
    c = np.cumsum(q); J = np.zeros((2, 3))
    for j in range(3):
        J[0, j] = -np.sum(Lc[j:] * np.sin(c[j:])); J[1, j] = np.sum(Lc[j:] * np.cos(c[j:]))
    return J
Jn3 = J3(qc); eps = 1e-6
Jnum = np.column_stack([(fk3(qc + eps * np.eye(3)[j]) - fk3(qc - eps * np.eye(3)[j])) / (2 * eps) for j in range(3)])
close('w6q1 J vs finite differences', Jn3, Jnum, tol=1e-6)
print('  J =', Jn3.round(4).tolist())
qd = np.array([0.3, -0.2, 0.5]); vlin = Jn3 @ qd
print('  v', vlin, ' wz', qd.sum(), ' FK', fk3(qc))
ans('qa-w06-1', 'vx', vlin[0]); ans('qa-w06-1', 'vy', vlin[1]); ans('qa-w06-1', 'wz', qd.sum())
# wrong: using absolute angle only for first column
# min-norm joint rates for v = (0.05, 0.02)? keep for solution
vdes = np.array([0.05, 0.02]); qmin = np.linalg.pinv(Jn3) @ vdes
check('w6q1 min norm solves J qd = v', np.allclose(Jn3 @ qmin, vdes))
print('  min-norm rates', qmin, 'norm', np.linalg.norm(qmin))
ans('qa-w06-1', 'qmin_norm', np.linalg.norm(qmin))
# Q2 SCARA Jacobian
def fk_scara(q):
    q1_, q2_, d3_, q4_ = q
    T = nchain(tz(H0), Rz(q1_), tx(A1), Rz(q2_), tx(A2), tz(-d3_), Rz(q4_)); return np.array([T[0, 3], T[1, 3], T[2, 3], q1_ + q2_ + q4_])
qs0 = np.array([40 * deg, -70 * deg, 0.12, 25 * deg])
c12 = np.cos(qs0[0] + qs0[1]); s12 = np.sin(qs0[0] + qs0[1])
xx, yy = fk_scara(qs0)[:2]
Jsc = np.array([[-yy, -A2 * s12, 0, 0], [xx, A2 * c12, 0, 0], [0, 0, -1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 1]])
Jsn = np.column_stack([(fk_scara(qs0 + eps * np.eye(4)[j])[:3] - fk_scara(qs0 - eps * np.eye(4)[j])[:3]) / (2 * eps) for j in range(4)])
close('w6q2 linear rows vs finite differences', Jsc[:3], Jsn, tol=1e-6)
qd2 = np.array([0.5, 0.2, 0.05])
vs = Jsc[:3, :3] @ qd2; detv = np.linalg.det(Jsc[:3, :3])
print('  tip', xx, yy, ' c1', Jsc[:3, 0], 'c2', Jsc[:3, 1], ' v', vs, ' det', detv, ' L1L2 sin q2', A1 * A2 * np.sin(qs0[1]))
check('w6q2 det = -L1 L2 sin q2', np.isclose(detv, -A1 * A2 * np.sin(qs0[1])))
print('  sign', np.sign(detv), np.sign(A1 * A2 * np.sin(qs0[1])))
ans('qa-w06-2', 'vx', vs[0]); ans('qa-w06-2', 'vy', vs[1]); ans('qa-w06-2', 'vz', vs[2]); ans('qa-w06-2', 'det', detv)
ans('qa-w06-2', 'J12', Jsc[0, 1]); ans('qa-w06-2', 'J22', Jsc[1, 1])
# Q3 statics
l1t, l2t = 0.6, 0.35; q1t, q2t = 50 * deg, -40 * deg; F = np.array([-12.0, 5.0])
c1_, s1_, c12_, s12_ = np.cos(q1t), np.sin(q1t), np.cos(q1t + q2t), np.sin(q1t + q2t)
Jt = np.array([[-l1t * s1_ - l2t * s12_, -l2t * s12_], [l1t * c1_ + l2t * c12_, l2t * c12_]])
tau = Jt.T @ F; wrongt = Jt @ F
print('  J', Jt.round(4).tolist(), 'tau', tau, 'wrong J F', wrongt)
# virtual work check
qdt = np.array([0.3, -0.7]); close('w6q3 power balance', tau @ qdt, F @ (Jt @ qdt))
ans('qa-w06-3', 'tau1', tau[0]); ans('qa-w06-3', 'tau2', tau[1]); ans('qa-w06-3', 'wrong1', wrongt[0])
# Q4 manipulability
Le = 0.5; q2m = 110 * deg
Jm = np.array([[-Le * np.sin(0.3) - Le * np.sin(0.3 + q2m), -Le * np.sin(0.3 + q2m)], [Le * np.cos(0.3) + Le * np.cos(0.3 + q2m), Le * np.cos(0.3 + q2m)]])
sv = np.linalg.svd(Jm, compute_uv=False)
lam = np.linalg.eigvalsh(Jm @ Jm.T)
print('  sigma', sv, 'lambda', lam, 'm', sv[1] / sv[0], 'w', sv[0] * sv[1], '|det|', abs(np.linalg.det(Jm)), 'L^2 sin', Le ** 2 * np.sin(q2m))
trc = np.trace(Jm @ Jm.T); dt = np.linalg.det(Jm @ Jm.T)
lam_hand = np.array([(trc - np.sqrt(trc ** 2 - 4 * dt)) / 2, (trc + np.sqrt(trc ** 2 - 4 * dt)) / 2]); close('w6q4 lambda by trace/det', lam_hand, lam)
# independence of q1
Jm2 = np.array([[-Le * np.sin(1.1) - Le * np.sin(1.1 + q2m), -Le * np.sin(1.1 + q2m)], [Le * np.cos(1.1) + Le * np.cos(1.1 + q2m), Le * np.cos(1.1 + q2m)]])
close('w6q4 sigma independent of q1', np.linalg.svd(Jm2, compute_uv=False), sv)
# closed form check: JJ^T in tool frame = [[L1^2 s2^2, L1^2 s2 c2 + ...]] - just compare tr, det
ans('qa-w06-4', 'sigma1', sv[0]); ans('qa-w06-4', 'sigma2', sv[1]); ans('qa-w06-4', 'm', sv[1] / sv[0]); ans('qa-w06-4', 'w', sv[0] * sv[1])
ans('qa-w06-4', 'wrong_m', lam[0] / lam[1])
# Q5 growth factor
fac = np.sin(np.radians(20)) / np.sin(np.radians(4)); print('  factor', fac)
ans('qa-w06-5', 'factor', fac)
# damped least squares example: J near singular
# Q6 spot the error
l1e, l2e, q1e, q2e = 0.5, 0.3, 30 * deg, 60 * deg
J11 = -l1e * np.sin(q1e) - l2e * np.sin(q1e + q2e); J21 = l1e * np.cos(q1e) + l2e * np.cos(q1e + q2e)
J12 = -l2e * np.sin(q1e + q2e); J22 = l2e * np.cos(q1e + q2e)
J11b = -l1e * np.sin(q1e) - l2e * np.sin(q2e); J21b = l1e * np.cos(q1e) + l2e * np.cos(q2e)
print('  correct J11,J21,J12,J22', J11, J21, J12, J22, ' student J11,J21', J11b, J21b, ' det', J11 * J22 - J12 * J21, l1e * l2e * np.sin(q2e))
close('w6q6 det = L1L2 sin q2', J11 * J22 - J12 * J21, l1e * l2e * np.sin(q2e))
ans('qa-w06-6', 'J11', J11); ans('qa-w06-6', 'J21', J21); ans('qa-w06-6', 'wrong_J11', J11b); ans('qa-w06-6', 'det', J11 * J22 - J12 * J21)


# =====================================================================  extra checks for numbers quoted in the worked solutions
print('--- solution intermediates')
rd = lambda v, n=4: round(float(v), n)
check('w1q1 forgot ground gives -2', gk(3, 5, 7, 7) == -2)
ans('qa-w02-1', 'yaw', d(yaw))
close('w2q1 matrix rotation block', T[:2, :2], [[0.9962, 0.0872], [-0.0872, 0.9962]], tol=5e-5)
close('w2q2 wrong z if alpha=0 (arm stays horizontal)', nchain(DH_prox(0, 0, qq[0], L0), DH_prox(0, 0, qq[1], 0), DH_prox(0, B2, qq[2], 0), tx(B3))[2, 3], 0.45)
close('w2q2 intermediate r, z', [rd(r), rd(zz)], [0.5390, 0.8590])
close('w2q4 frame-1 origin of T12 (a, -sin(alpha) d, cos(alpha) d)', T12[:3, 3], [0.2, -0.05, 0.0], tol=1e-12)
close('w2q4 hand arithmetic', [0.2 * np.cos(30 * deg) + 0.05 * np.sin(30 * deg), 0.2 * np.sin(30 * deg) - 0.05 * np.cos(30 * deg)], [T02[0, 3], T02[1, 3]])
close('w3q1 atan2(0.25,0.30) and mu', [rd(d(np.arctan2(0.25, 0.30)), 3), rd(d(np.arctan2(0.25 * np.sin(q2p), 0.35 + 0.25 * np.cos(q2p))), 3)], [39.806, 38.980], tol=2e-3)
close('w3q1 mu components', [rd(0.25 * np.sin(q2p)), rd(0.35 + 0.25 * np.cos(q2p))], [0.2457, 0.3036])
close('w3q1 other branch q4', 60 - REG[('qa-w03-1', 'wrong_q1')] + REG[('qa-w03-1', 'q2')], 81.917, tol=2e-3)
close('w3q2 rW, sigma, mu', [rd(rW), rd(d(sg), 2), rd(d(mu), 2)], [0.4986, 31.03, -24.34], tol=6e-3)
close('w3q2 numerator', rd(rW ** 2 + (pW[2] - Hs) ** 2 - Fs ** 2 - Gs ** 2), 0.1261)
close('w3q3 atan2(-0.3,0.7) and mu', [rd(d(np.arctan2(-0.3, 0.7)), 2), rd(d(np.arctan2(l2 * np.sin(q2a), l1 + l2 * np.cos(q2a))), 2)], [-23.20, 36.20], tol=6e-3)
close('w3q3 cos q2 numerator 0.0175', rd(0.7 ** 2 + 0.3 ** 2 - l1 ** 2 - l2 ** 2), 0.0175)
close('w3q4 partials', [rd(fk2(q0)[0]), rd(fk2(q0)[1]), rd(e[0]), rd(e[1]), rd(Jn[0, 0]), rd(Jn[0, 1]), rd(Jn[1, 0]), rd(Jn[1, 1]), rd(np.linalg.det(Jn))], [0.7773, 0.5784, -0.0773, -0.2784, -0.5784, -0.4010, 0.7773, 0.2041, 0.1937])
c2s, so = planar_ik(xs, ys, l1s, l2s)
close('w3q6 other branch q1 = 191.30', d(so[1][0]), 191.30, tol=6e-3)
close('w3q6 mu', d(mu_s), 44.989, tol=2e-3)
close('w4q1 platform points and vectors', [rd(P[0, 0]), rd(P[0, 1]), rd(P[1, 0]), rd(P[1, 1]), rd(P[2, 0]), rd(P[2, 1])], [0.6970, 0.3847, 0.3714, 0.5032, 0.4316, 0.1621])
close('w4q1 leg vectors', (P - b).round(4), [[0.6970, 0.3847], [-0.6286, 0.5032], [-0.0684, -0.7379]])
close('w4q2 triangle numbers', [rd(AD ** 2), rd(xa), rd(hh), rd(Bup[0]), rd(Bup[1]), rd(A[0]), rd(A[1])], [0.5025, 0.4320, 0.4164, 0.6635, 0.4810, 0.1250, 0.2165])
close('w4q2 |AD|^2 by cosine rule', L0f ** 2 + L1f ** 2 - 2 * L0f * L1f * np.cos(th1), AD ** 2)
close('w4q4 leg vectors', [(tt + Rr @ pp[1] - bb[1]).round(4), (tt + Rr @ pp[4] - bb[4]).round(4)], [[-0.7348, 0.2594, 0.9], [0.1729, 0.5950, 0.9]], tol=6e-5)
close('w4q4 platform point 2, 5 after rotation', [(Rr @ pp[1]).round(4)[:2], (Rr @ pp[4]).round(4)[:2]], [[0.25, 0.4330], [-0.4698, -0.1710]], tol=6e-5)
close('w4q4 leg2 squared', rd(0.7348 ** 2 + 0.2594 ** 2 + 0.81), 1.4173, tol=2e-4)
close('w5q1 rest-to-rest end accel is -45', 2 * 22.5 + 6 * (-7.5) * 2, -45.0)
# W5 Q3 earlier segment 2->5
Ts2 = 3.0; Ds2 = 40.0; v0s2, vfs2 = v[1], v[2]
c2_ = (3 * Ds2 - (2 * v0s2 + vfs2) * Ts2) / Ts2 ** 2; c3_ = (-2 * Ds2 + (v0s2 + vfs2) * Ts2) / Ts2 ** 3
close('w5q3 segment 2->5 coefficients', [rd(c2_), rd(c3_), rd(2 * c2_ + 6 * c3_ * Ts2), rd(2 * a2k)], [2.9167, -0.7870, -8.3333, -4.1667])
check('w5q3 q(4) = 80 pieces', np.isclose(70 + 7.9167 * 4 - 2.0833 * 16 + 0.18229 * 64, 80, atol=2e-3))
check('w5q4 a=130 discriminant -975', np.isclose(130 ** 2 * 1.5 ** 2 - 4 * 130 * 75, -975))
close('w5q4 parts', [rd(np.sqrt(disc), 2), rd(0.5 * aa * tb ** 2 * 2, 2), rd(vq * (Tf - 2 * tb), 2)], [173.21, 20.10, 54.90], tol=6e-3)
close('w5q5 LSPB a=160', [rd(a_l), rd(tb5, 4), rd(a_l * tb5, 2), rd(np.sqrt(a_l ** 2 * T45 ** 2 - 4 * a_l * D45), 2)], [160.0, 0.2197, 35.15, 169.71], tol=6e-3)
ans('qa-w05-5', 'lspb_v', a_l * tb5)
# W5 Q6 segments with via velocity 22.5
s1a2 = (3 * 40 - (0 + 22.5) * 2) / 4; s1a3 = (-2 * 40 + (0 + 22.5) * 2) / 8
s2a2 = (3 * 20 - (2 * 22.5 + 0) * 2) / 4; s2a3 = (-2 * 20 + (22.5 + 0) * 2) / 8
close('w5q6 segments', [s1a2, s1a3, s2a2, s2a3], [18.75, -4.375, -7.5, 0.625])
close('w5q6 checks', [s1a2 * 4 + s1a3 * 8, 2 * s1a2 * 2 + 3 * s1a3 * 4, 2 * s1a2 + 6 * s1a3 * 2, 40 + 22.5 * 2 + s2a2 * 4 + s2a3 * 8, 2 * s2a2], [40, 22.5, -15, 60, -15])
# W6
close('w6q1 J entries', Jn3.round(4), [[-0.4966, -0.3598, -0.1], [0.6991, 0.3232, 0.1732]], tol=6e-5)
close('w6q1 min-norm rates', qmin.round(4), [0.2069, -0.4664, 0.1507], tol=6e-5)
close('w6q2 columns', [Jsc[0, 0], Jsc[1, 0], Jsc[0, 1], Jsc[1, 1]], [-0.09998, 0.48462, 0.125, 0.21651], tol=6e-5)
close('w6q3 J entries', Jt.round(4), [[-0.5204, -0.0608], [0.7304, 0.3447]], tol=6e-5)
close('w6q3 products', [rd(-0.5204 * -12 + 0.7304 * 5), rd(-0.0608 * -12 + 0.3447 * 5)], [9.8968, 2.4531], tol=2e-4)
JJt = Jm @ Jm.T
close('w6q4 JJt in tool frame', [rd(0.5 * np.sin(q2m)), rd(0.5 * np.cos(q2m) + 0.5), rd(trc), rd(dt), rd(np.sqrt(trc ** 2 - 4 * dt))], [0.4698, 0.3290, 0.5790, 0.0552, 0.3383], tol=6e-4)
close('w6q4 eigenvalues', lam.round(4), [0.1203, 0.4587], tol=6e-5)
close('w6q4 JJt entries', [rd((0.5 * np.sin(q2m)) ** 2, 4), rd(0.5 * np.sin(q2m) * (0.5 * np.cos(q2m) + 0.5), 4), rd((0.5 * np.cos(q2m) + 0.5) ** 2 + 0.25, 4)], [0.2208, 0.1546, 0.3582], tol=6e-5)
check('w6q5 sin20 / sin4', np.isclose(np.sin(np.radians(20)) / np.sin(np.radians(4)), 4.9031, atol=1e-4))
close('w6q6 J12, J22', [J12, J22], [-0.3, 0.0], tol=1e-9)

# =====================================================================  optional HTML cross-check
def html_check():
    src = open(os.path.join(HERE, '..', 'frag', 'qa', 'part1.html')).read()
    gps = re.split(r'<div class="gp"', src)[1:]
    n_in = 0
    for g in gps:
        m = re.search(r'data-id="(qa-w\d\d-\d)"', g)
        qid = m.group(1)
        for im in re.finditer(r'<input data-ans="([^"]+)" data-tol="([^"]+)" data-label="([^"]+)"', g):
            val, tol, lab = float(im.group(1)), float(im.group(2)), im.group(3)
            key = (qid, lab)
            ok = key in REG and abs(REG[key] - val) <= tol * 0.9 + 1e-12
            check(f'html {qid}:{lab} data-ans={val} vs computed {REG.get(key)}', ok); n_in += 1
        for wm in re.finditer(r'data-wrong=\'(\[.*?\])\'', g):
            pass
    print(f'  {len(gps)} questions, {n_in} answer boxes cross-checked')

if '--html' in sys.argv:
    html_check()
report()
