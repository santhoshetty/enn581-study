"""W2 Forward kinematics — verifies every number used on the W2 tab, booklet page and widget test vectors.
Run: python3 site/src/checks/w02.py  (prints PASS/FAIL; writes w02_vectors.json for the widget tests)"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import roboticstoolbox as rtb

out = {}
# ---------------- 1. Planar RRR (slides pp. 6-9) worked example ----------------
L1, L2, L3 = 0.5, 0.4, 0.2
q = np.array([30, 45, -60]) * deg
x = L1*np.cos(q[0]) + L2*np.cos(q[0]+q[1]) + L3*np.cos(q.sum())
y = L1*np.sin(q[0]) + L2*np.sin(q[0]+q[1]) + L3*np.sin(q.sum())
qE = q.sum()
# same thing as an ETS: Rz(q1) tx(L1) Rz(q2) tx(L2) Rz(q3) tx(L3)
T = nchain(Rz(q[0]), tx(L1), Rz(q[1]), tx(L2), Rz(q[2]), tx(L3))
close('RRR: trig formula == ETS product (x,y)', [x, y], T[:2, 3])
close('RRR: orientation qE = atan2(T21,T11)', np.arctan2(T[1, 0], T[0, 0]), qE)
print(f'   RRR example: x={x:.4f} m, y={y:.4f} m, qE={qE/deg:.1f} deg')
out['rrr'] = dict(L=[L1, L2, L3], q_deg=[30, 45, -60], x=r(x), y=r(y), qE_deg=r(qE/deg, 2))

# ---------------- 2. Homogeneous transform facts (p. 12) ----------------
th = 30*deg
TAB = Rz(th) @ Tr(1, 2, 0)  # an example pose
Rm, t = TAB[:3, :3], TAB[:3, 3]
Tinv = np.eye(4); Tinv[:3, :3] = Rm.T; Tinv[:3, 3] = -Rm.T @ t
close('inverse: [R^T, -R^T t] == numpy inv', Tinv, np.linalg.inv(TAB))
check('Rz(30) is a proper rotation', is_rotation(Rm))
# columns of R are the axes of {B} expressed in {A}
close('col 1 of Rz(30) = x_B in {A}', Rm[:, 0], [np.cos(th), np.sin(th), 0])
# point mapping: B_P = (1,0,0) -> A_P
BP = np.array([1, 0, 0, 1.0]); AP = TAB @ BP
out['tf'] = dict(theta_deg=30, t_BA_in_B=[1, 2, 0], AP=[r(v) for v in AP[:3]], AtB=[r(v) for v in t])
print(f'   T = Rz(30°) tr(1,2,0): origin of B in A = {np.round(t,4)}, point (1,0,0)_B -> {np.round(AP[:3],4)}_A')

# ---------------- 3. Compound transforms (p. 13): ATC = Rx(theta) tz(d) ----------------
th, d = 30*deg, 2.0
ATC = Rx(th) @ tz(d)
close('compound: origin of C in A = (0, -d sin th, d cos th)', ATC[:3, 3], [0, -d*np.sin(th), d*np.cos(th)])
# order matters: tz(d) Rx(th) puts the origin at (0,0,d)
close('order: tz(d)Rx(th) origin = (0,0,d)', (tz(d) @ Rx(th))[:3, 3], [0, 0, d])
out['compound'] = dict(theta_deg=30, d=2, origin=[r(v) for v in ATC[:3, 3]])

# ---------------- 4. Furuta pendulum: ETS (p. 19) vs proximal DH (p. 34) ----------------
q1, q2, L0s, L1s, L2s = sp.symbols('q1 q2 L0 L1 L2', real=True)
ETS = chain(stz(L0s), sRz(q1), stx(L1s), sRx(q2), stz(-L2s), sRy(sp.pi))
DH = chain(sDH_prox(0, 0, q1 + sp.pi/2, L0s), sDH_prox(sp.pi/2, 0, q2, L1s), sDH_prox(sp.pi/2, 0, 0, L2s))
fix = sRz(-sp.pi/2)  # 3T_E
diff = sp.simplify(DH * fix - ETS)
check('Furuta: DH table (p.34) * Rz(-90°) == ETS (p.19), symbolically', diff == sp.zeros(4, 4))
diff_nofix = sp.simplify(DH - ETS)
check('Furuta: without the fix-up the frames differ ({3} != {E})', diff_nofix != sp.zeros(4, 4))
print('   Furuta ETS position:', [sp.simplify(e) for e in ETS[:3, 3]])
vals = {L0s: 0.5, L1s: 0.4, L2s: 0.3, q1: sp.pi/3, q2: sp.pi/4}
Tf = np.array(ETS.subs(vals).evalf(), dtype=float)
print('   Furuta (L0=0.5,L1=0.4,L2=0.3, q=(60°,45°)) T =\n', np.round(Tf, 4))
# independent: rtb ETS
ets = rtb.ET.tz(0.5) * rtb.ET.Rz() * rtb.ET.tx(0.4) * rtb.ET.Rx() * rtb.ET.tz(-0.3) * rtb.ET.Ry(np.pi)
close('Furuta: rtb ETS fkine == sympy', ets.eval([np.pi/3, np.pi/4]), Tf, show=False)
check('Furuta: R is a rotation', is_rotation(Tf[:3, :3]))
out['furuta'] = dict(L=[0.5, 0.4, 0.3], q_deg=[60, 45], T=np.round(Tf, 4).tolist())
# notebook bug: tz(-30)
ets_bug = rtb.ET.tz(0.5) * rtb.ET.Rz() * rtb.ET.tx(0.4) * rtb.ET.Rx() * rtb.ET.tz(-30) * rtb.ET.Ry(np.pi)
close('Notebook output (-18.17, 10.95, -20.71) comes from the tz(-30) typo', ets_bug.eval([np.pi/3, np.pi/4])[:3, 3], [-18.17, 10.95, -20.71], tol=0.01, show=False)

# Furuta workspace: rho^2 + (z-L0)^2 = L1^2 + L2^2 (spherical band, |z-L0| <= L2)
rng0 = np.random.default_rng(5); okw = True
for _ in range(300):
    qq = rng0.uniform(-np.pi, np.pi, 2)
    p = ets.eval(qq)[:3, 3]
    okw &= abs(p[0]**2 + p[1]**2 + (p[2] - 0.5)**2 - (0.4**2 + 0.3**2)) < 1e-12 and abs(p[2] - 0.5) <= 0.3 + 1e-12
check('Furuta workspace: every tip point on sphere radius sqrt(L1²+L2²)=0.5 about (0,0,L0), |z-L0|<=L2', okw)

# ---------------- 5. Notebook Q3: two frames ----------------
# Figure: origin of B at (3,5,2) in A; x_B = -z_A, z_B = +y_A, so y_B = z_B x x_B = -x_A
R_true = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0.0]])
check('two frames: R columns form a right-handed frame', is_rotation(R_true))
T_ets = nchain(Tr(3, 5, 2), Ry(90*deg), Rx(-90*deg))
close('two frames: tr(3,5,2) Ry(90°) Rx(-90°) gives R_true', T_ets[:3, :3], R_true, show=False)
T_nb = nchain(ty(5), tx(3), Ry(90*deg), Rx(90*deg))
check('two frames: the notebook answer ty5 tx3 Ry90 Rx90 is WRONG', not np.allclose(T_nb, T_ets))
out['twoframes'] = dict(T=np.round(T_ets, 4).tolist(), ets='tx(3) ty(5) tz(2) Ry(90°) Rx(−90°)')

# ---------------- 6. Panda: Franka official proximal (modified) DH ----------------
a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]
dd = [0.333, 0, 0.316, 0, 0.384, 0, 0]
al = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2]
def panda_fk(qv):
    T = np.eye(4)
    for i in range(7):
        T = T @ DH_prox(al[i], a[i], qv[i], dd[i])
    return T @ tz(0.107)  # flange
links = [rtb.RevoluteMDH(a=a[i], d=dd[i], alpha=al[i]) for i in range(7)]
panda_rtb = rtb.DHRobot(links, tool=sm_tool if (sm_tool := None) else None)
import spatialmath as smath
panda_rtb.tool = smath.SE3.Tz(0.107)
rng = np.random.default_rng(0)
ok = True
for _ in range(20):
    qv = rng.uniform(-np.pi, np.pi, 7)
    ok &= np.allclose(panda_fk(qv), panda_rtb.fkine(qv).A, atol=1e-10)
check('Panda: my proximal-DH chain == rtb RevoluteMDH model (20 random poses)', ok)
# also matches rtb's built-in Panda (URDF-derived) for the flange position
p_builtin = rtb.models.DH.Panda()
p_builtin.tool = smath.SE3()  # built-in adds a 0.103 m gripper + Rz(-45°); remove it to compare at the flange
ok2 = True
for _ in range(10):
    qv = rng.uniform(-np.pi, np.pi, 7)
    ok2 &= np.allclose(panda_fk(qv)[:3, 3], p_builtin.fkine(qv).t, atol=1e-6)
check('Panda: flange position == rtb.models.DH.Panda (10 random poses)', ok2)
qz = np.zeros(7)
Tz = panda_fk(qz)
print('   Panda at q=0, flange position:', np.round(Tz[:3, 3], 4))
out['panda'] = dict(a=a, d=dd, alpha_deg=[r(v/deg, 1) for v in al], flange_d=0.107, p0=np.round(Tz[:3, 3], 4).tolist())
# ETS form written out from the DH rows (Rx(alpha) tx(a) Rz(q) tz(d) per row)
ets_panda = (rtb.ET.tz(0.333) * rtb.ET.Rz() * rtb.ET.Rx(-np.pi/2) * rtb.ET.Rz() * rtb.ET.Rx(np.pi/2) * rtb.ET.tz(0.316) * rtb.ET.Rz()
             * rtb.ET.Rx(np.pi/2) * rtb.ET.tx(0.0825) * rtb.ET.Rz() * rtb.ET.Rx(-np.pi/2) * rtb.ET.tx(-0.0825) * rtb.ET.tz(0.384) * rtb.ET.Rz()
             * rtb.ET.Rx(np.pi/2) * rtb.ET.Rz() * rtb.ET.Rx(np.pi/2) * rtb.ET.tx(0.088) * rtb.ET.Rz() * rtb.ET.tz(0.107))
ok3 = True
for _ in range(10):
    qv = rng.uniform(-np.pi, np.pi, 7)
    ok3 &= np.allclose(ets_panda.eval(qv), panda_fk(qv), atol=1e-10)
check('Panda: written-out ETS == DH chain', ok3)

# ---------------- 7. PUMA-like ETS (notebook Q5; own link values, no offsets) ----------------
# home pose: upper arm and forearm vertical; joints 2,3 horizontal (about y); wrist roll about forearm (z), bend about y, swivel about z
Lb, Lu, Lf = 1.0, 0.5, 0.3
def puma(qv):
    return nchain(tz(Lb), Rz(qv[0]), Ry(qv[1]), tz(Lu), Ry(qv[2]), tz(Lf), Rz(qv[3]), Ry(qv[4]), Rz(qv[5]))
close('PUMA-like: home pose position = (0,0,Lb+Lu+Lf)', puma(np.zeros(6))[:3, 3], [0, 0, Lb+Lu+Lf])
# wrist centre independent of q4..q6 (spherical wrist: three axes intersect)
qa = np.array([0.3, -0.5, 0.8, 0.0, 0.0, 0.0]); qb = qa.copy(); qb[3:] = [1.1, -0.7, 2.0]
close('PUMA-like: wrist centre does not move with q4..q6', puma(qa)[:3, 3], puma(qb)[:3, 3], show=False)
out['puma'] = dict(L=[Lb, Lu, Lf], ets='tz(Lb) Rz(q1) Ry(q2) tz(Lu) Ry(q3) tz(Lf) Rz(q4) Ry(q5) Rz(q6)')

# ---------------- 8. widget test vectors (DH explorer presets) ----------------
vec = []
for qv in [[0, 0, 0], [30, 45, -60], [90, -30, 10]]:
    qq = np.array(qv) * deg
    T = nchain(Rz(qq[0]), tx(L1), Rz(qq[1]), tx(L2), Rz(qq[2]), tx(L3))
    vec.append(dict(preset='planarRRR', q_deg=qv, T=np.round(T, 6).tolist()))
for qv in [[0, 0], [60, 45], [-30, 120]]:
    T = np.array(ETS.subs({L0s: 0.5, L1s: 0.4, L2s: 0.3, q1: qv[0]*sp.pi/180, q2: qv[1]*sp.pi/180}).evalf(), dtype=float)
    vec.append(dict(preset='furuta', q_deg=qv, T=np.round(T, 6).tolist()))
for qv in [[0]*7, [10, -20, 30, -90, 15, 60, 45]]:
    vec.append(dict(preset='panda', q_deg=qv, T=np.round(panda_fk(np.array(qv)*deg), 6).tolist()))
out['vectors'] = vec
json.dump(out, open(os.path.join(os.path.dirname(__file__), 'w02_vectors.json'), 'w'), indent=1)
report()
