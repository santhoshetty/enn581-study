"""W3 Inverse kinematics — verifies every number on the W3 tab. Run: python3 site/src/checks/w03.py"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *

out = {}
def fk2(L1, L2, q1, q2):
    return np.array([L1*np.cos(q1) + L2*np.cos(q1+q2), L1*np.sin(q1) + L2*np.sin(q1+q2)])

def ik2(L1, L2, x, y):
    """algebraic IK (slides pp.25-30): both elbow branches; signed CCW angles"""
    c2 = (x*x + y*y - L1*L1 - L2*L2) / (2*L1*L2)
    if abs(c2) > 1: return []
    sols = []
    for s in (+1, -1):
        q2 = np.arctan2(s*np.sqrt(1 - c2*c2), c2)
        q1 = np.arctan2(y, x) - np.arctan2(L2*np.sin(q2), L1 + L2*np.cos(q2))
        sols.append((q1, q2))
    return sols

# ---- workspace of the slide's 2R (pp.14-16): L1 = 1, L2 = 0.7
check('2R workspace radii |L1-L2| = 0.3, L1+L2 = 1.7', np.isclose(abs(1 - 0.7), 0.3) and np.isclose(1 + 0.7, 1.7))

# ---- worked 2R IK example
L1, L2, x, y = 1.0, 0.7, 1.2, 0.8
c2 = (x*x + y*y - L1*L1 - L2*L2) / (2*L1*L2)
sols = ik2(L1, L2, x, y)
print(f'2R: cos q2 = {c2:.6f}')
for q1, q2 in sols: print(f'   q1 = {q1/deg:.3f}°, q2 = {q2/deg:.3f}°')
check('2R: both branches reproduce (1.2, 0.8)', all(np.allclose(fk2(L1, L2, *s), [x, y]) for s in sols))
# geometric route (p.31): sigma = atan2(y,x); mu from sine rule; q1 = sigma - mu (mu signed with q2)
sig = np.arctan2(y, x)
for q1, q2 in sols:
    mu = np.arctan2(L2*np.sin(q2), L1 + L2*np.cos(q2))
    r_ = np.hypot(x, y)
    check(f'2R geometric: sin(mu)/L2 = sin(q2)/r for branch q2={q2/deg:.2f}°', np.isclose(np.sin(mu)/L2, np.sin(q2)/r_))
    check(f'2R geometric: q1 = sigma - mu (branch {q2/deg:.2f}°)', np.isclose(q1, sig - mu))
out['ik2'] = dict(L=[L1, L2], P=[x, y], c2=r(c2, 6), sols_deg=[[r(a/deg, 3), r(b/deg, 3)] for a, b in sols], sigma_deg=r(sig/deg, 3))

# ---- atan vs atan2 trap
check('atan2(-1,-1) = -135°, atan(1) = 45°', np.isclose(np.arctan2(-1, -1)/deg, -135) and np.isclose(np.arctan(1)/deg, 45))

# ---- planar RP (p.24)
xr, yr = 0.3, 0.4
check('RP: S = 0.5, q1 = 53.130°', np.isclose(np.hypot(xr, yr), 0.5) and np.isclose(np.arctan2(yr, xr)/deg, 53.13010235))

# ---- unreachable target
check('2R: (2.0, 0.5) unreachable (|cos q2| > 1)', ik2(1.0, 0.7, 2.0, 0.5) == [])

# ---- planar RRR (notebook Q10): given (x, y, phi) of the tip
L = (1.0, 0.7, 0.3); xt, yt, ph = 1.2, 0.9, 60*deg
xw, yw = xt - L[2]*np.cos(ph), yt - L[2]*np.sin(ph)
s3 = ik2(L[0], L[1], xw, yw)
rrr = [(a, b, ph - a - b) for a, b in s3]
def fk3(q):
    a = np.cumsum(q); return np.array([sum(L[i]*np.cos(a[i]) for i in range(3)), sum(L[i]*np.sin(a[i]) for i in range(3)), a[-1]])
check('RRR: both solutions reproduce (x, y, phi)', all(np.allclose(fk3(q)[:2], [xt, yt]) and np.isclose(np.arctan2(np.sin(fk3(q)[2]), np.cos(fk3(q)[2])), ph) for q in rrr))
print(f'RRR: wrist point = ({xw:.4f}, {yw:.4f})')
for q in rrr: print('   q (deg) =', np.round(np.array(q)/deg, 3))
out['rrr'] = dict(L=list(L), target=[xt, yt, 60], wrist=[r(xw), r(yw)], sols_deg=[[r(v/deg, 3) for v in q] for q in rrr])

# ---- spherical-wrist arm, position part (pp.35-37) with H, G, F; signed CCW angles in the (r, z) plane
H, G, F = 0.5, 0.4, 0.3
def fk_pos(q):
    rr = G*np.cos(q[1]) + F*np.cos(q[1] + q[2])
    return np.array([np.cos(q[0])*rr, np.sin(q[0])*rr, H + G*np.sin(q[1]) + F*np.sin(q[1] + q[2])])
qtrue = np.array([40, 30, -50])*deg
pw = fk_pos(qtrue)
rw = np.hypot(pw[0], pw[1]); zz = pw[2] - H
c3 = (rw**2 + zz**2 - F**2 - G**2) / (2*F*G)
check('wrist arm: cosine rule gives cos q3 (relative angle) = RHS of slide p.36', np.isclose(c3, np.cos(qtrue[2])))
check('wrist arm: slide p.36 LHS cos(pi - q3) = -RHS, so the printed sign is wrong for a relative q3', np.isclose(np.cos(np.pi - qtrue[2]), -c3))
sol = []
for s in (+1, -1):
    q3 = np.arctan2(s*np.sqrt(1 - c3**2), c3)
    sigma = np.arctan2(zz, rw); mu = np.arctan2(F*np.sin(q3), G + F*np.cos(q3))
    sol.append((np.arctan2(pw[1], pw[0]), sigma - mu, q3))
check('wrist arm: q2 = sigma - mu(q3) with signed q3 recovers (40, 30, -50)°', any(np.allclose(s_, qtrue) for s_ in sol))
check('wrist arm: every branch reproduces the wrist centre', all(np.allclose(fk_pos(np.array(s_)), pw) for s_ in sol))
# slide form q2 = sigma + mu with mu built from |q3| (elbow-up branch, q3 drawn clockwise)
mu_mag = np.arctan2(F*np.sin(abs(qtrue[2])), G + F*np.cos(abs(qtrue[2])))
check('wrist arm: slide p.37 q2 = sigma + mu holds when q3 is measured clockwise (magnitude)', np.isclose(qtrue[1], np.arctan2(zz, rw) + mu_mag))
print(f'wrist arm: P_W = {np.round(pw, 4)}, r_W = {rw:.4f}, cos q3 = {c3:.4f}')
for s_ in sol: print('   (q1,q2,q3) deg =', np.round(np.array(s_)/deg, 3))
out['wrist'] = dict(H=H, G=G, F=F, PW=np.round(pw, 4).tolist(), rW=r(rw), c3=r(c3), sols_deg=[[r(v/deg, 3) for v in s_] for s_ in sol])

# ---- wrist centre from the end-effector pose: p_W = p_E - d6 * (approach axis z_E)
R = (Rz(0.3) @ Ry(0.8) @ Rz(-0.4))[:3, :3]; pE = np.array([0.6, 0.2, 0.7]); d6 = 0.1
pW = pE - d6*R[:, 2]
close('wrist centre: p_W = p_E - d6 R[:,2]', pW, pE - d6*R @ np.array([0, 0, 1]), show=False)

# ---- wrist orientation: 3R6 = (0R3)^T 0R6, then Z-Y-Z Euler extraction (added; slide p.38 says "like Euler angles")
a_, b_, c_ = 0.7, 1.1, -0.4
Rw = (Rz(a_) @ Ry(b_) @ Rz(c_))[:3, :3]
b_e = np.arctan2(np.hypot(Rw[2, 0], Rw[2, 1]), Rw[2, 2]); a_e = np.arctan2(Rw[1, 2], Rw[0, 2]); c_e = np.arctan2(Rw[2, 1], -Rw[2, 0])
close('ZYZ extraction: beta=atan2(sqrt(r31²+r32²),r33), alpha=atan2(r23,r13), gamma=atan2(r32,-r31)', [a_e, b_e, c_e], [a_, b_, c_])
b2 = -b_e; a2 = np.arctan2(-Rw[1, 2], -Rw[0, 2]); c2_ = np.arctan2(-Rw[2, 1], Rw[2, 0])
close('ZYZ second solution (beta<0, "wrist flip") gives the same R', (Rz(a2) @ Ry(b2) @ Rz(c2_))[:3, :3], Rw, show=False)

# ---- Newton-Raphson step (slides pp.45-52) on the 2R arm
def J2(L1, L2, q1, q2):
    return np.array([[-L1*np.sin(q1) - L2*np.sin(q1+q2), -L2*np.sin(q1+q2)], [L1*np.cos(q1) + L2*np.cos(q1+q2), L2*np.cos(q1+q2)]])
q = np.array([0.5, 0.5]); tgt = np.array([1.2, 0.8])
e0 = tgt - fk2(1.0, 0.7, *q)
dq = np.linalg.solve(J2(1.0, 0.7, *q), e0)
q1n = q + dq; e1 = tgt - fk2(1.0, 0.7, *q1n)
print(f'NR: q0 = (0.5, 0.5) rad, e0 = {np.round(e0,4)}, J0 = {np.round(J2(1,0.7,*q),4).tolist()}, dq = {np.round(dq,4)}, q1 = {np.round(q1n,4)}, |e1| = {np.linalg.norm(e1):.4f}')
it = q.copy(); n = 0
while np.linalg.norm(tgt - fk2(1.0, 0.7, *it)) > 1e-10 and n < 50:
    it = it + np.linalg.solve(J2(1.0, 0.7, *it), tgt - fk2(1.0, 0.7, *it)); n += 1
check(f'NR converges in {n} iterations to one of the analytic solutions', any(np.allclose(np.arctan2(np.sin(it), np.cos(it)), s, atol=1e-8) for s in sols))
out['nr'] = dict(q0=[0.5, 0.5], e0=np.round(e0, 4).tolist(), J0=np.round(J2(1, 0.7, *q), 4).tolist(), dq=np.round(dq, 4).tolist(), q1=np.round(q1n, 4).tolist(), e1=r(np.linalg.norm(e1)), iters=n)
check('NR step 1 shrinks the error', np.linalg.norm(e1) < np.linalg.norm(e0))
# ---- pseudo-inverse shapes (p.51)
Jf = np.random.default_rng(0).normal(size=(2, 3)); Jt = Jf.T
close('fat J (n>m): J+ = J^T (J J^T)^-1', Jf.T @ np.linalg.inv(Jf @ Jf.T), np.linalg.pinv(Jf), show=False)
close('tall J (n<m): J+ = (J^T J)^-1 J^T', np.linalg.inv(Jt.T @ Jt) @ Jt.T, np.linalg.pinv(Jt), show=False)

# ---- notebook Q12: RRP-type arm count (slider axis offset L2 from the theta2 axis), d3 sign free vs d3 >= 0
# in the plane normal to the theta2 axis, tip = L2*n + d3*t  ->  d3 = ±sqrt(rho^2 - L2^2): 2 per theta1 branch, 2 theta1 branches
rho, L2o = 0.8, 0.2
d3s = [np.sqrt(rho**2 - L2o**2), -np.sqrt(rho**2 - L2o**2)]
check('Q12 model: two slider values per plane, |d3| = sqrt(rho^2 - L2^2)', np.isclose(d3s[0], np.sqrt(0.6)))
json.dump(out, open(os.path.join(os.path.dirname(__file__), 'w03_vectors.json'), 'w'), indent=1)
report()
