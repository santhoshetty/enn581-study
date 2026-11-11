"""W7 Dynamics — verifies every number on the W7 tab (and the W8 practical sheet, which is about dynamics).
Run: python3 site/src/checks/w07.py"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from scipy.integrate import solve_ivp

out = {}
t = sp.symbols('t')
g = sp.symbols('g', positive=True)

def lagrange(qs, KE, PE):
    L = KE - PE
    qd = [sp.diff(q, t) for q in qs]
    tau = [sp.simplify(sp.diff(sp.diff(L, qd[i]), t) - sp.diff(L, qs[i])) for i in range(len(qs))]
    qdd = [sp.diff(q, t, 2) for q in qs]
    n = len(qs)
    M = sp.Matrix(n, n, lambda i, j: sp.simplify(sp.expand(tau[i]).coeff(qdd[j])))
    G = sp.Matrix([sp.simplify(tau[i].subs({**{x: 0 for x in qdd}}).subs({x: 0 for x in qd})) for i in range(n)])
    C = sp.simplify(sp.Matrix(tau) - M*sp.Matrix(qdd) - G)
    return tau, M, C, G

# ===== 2R point masses at link tips (slides pp.18-25) =====
m1, m2, L1, L2 = sp.symbols('m1 m2 L1 L2', positive=True)
q1, q2 = sp.Function('q1')(t), sp.Function('q2')(t)
x1, y1 = L1*sp.cos(q1), L1*sp.sin(q1)
x2, y2 = x1 + L2*sp.cos(q1 + q2), y1 + L2*sp.sin(q1 + q2)
K = sp.Rational(1, 2)*m1*(sp.diff(x1, t)**2 + sp.diff(y1, t)**2) + sp.Rational(1, 2)*m2*(sp.diff(x2, t)**2 + sp.diff(y2, t)**2)
P = m1*g*y1 + m2*g*y2
tau, M, C, G = lagrange([q1, q2], K, P)
d1, d2 = sp.diff(q1, t), sp.diff(q2, t)
c2, s2 = sp.cos(q2), sp.sin(q2)
K2_slide = sp.Rational(1, 2)*m2*((L1**2 + 2*L1*L2*c2 + L2**2)*d1**2 + 2*(L2**2 + L1*L2*c2)*d1*d2 + L2**2*d2**2)
check('2R: K2 equals slide p.20', sp.simplify(sp.Rational(1, 2)*m2*(sp.diff(x2, t)**2 + sp.diff(y2, t)**2) - K2_slide) == 0)
M_slide = sp.Matrix([[m1*L1**2 + m2*(L1**2 + 2*L1*L2*c2 + L2**2), m2*(L1*L2*c2 + L2**2)], [m2*(L1*L2*c2 + L2**2), m2*L2**2]])
C_slide = sp.Matrix([-m2*L1*L2*s2*(2*d1*d2 + d2**2), m2*L1*L2*d1**2*s2])
G_slide = sp.Matrix([(m1 + m2)*L1*g*sp.cos(q1) + m2*g*L2*sp.cos(q1 + q2), m2*g*L2*sp.cos(q1 + q2)])
check('2R: M equals slide p.25', sp.simplify(M - M_slide) == sp.zeros(2, 2))
check('2R: C equals slide p.25', sp.simplify(C - C_slide) == sp.zeros(2, 1))
check('2R: g equals slide p.25', sp.simplify(G - G_slide) == sp.zeros(2, 1))
h = m2*L1*L2*s2
check('2R: C = centripetal [[0,-h],[h,0]][qd1²,qd2²] + Coriolis [-2h;0] qd1 qd2 (p.30)',
      sp.simplify(C - (sp.Matrix([[0, -h], [h, 0]])*sp.Matrix([d1**2, d2**2]) + sp.Matrix([-2*h, 0])*d1*d2)) == sp.zeros(2, 1))
# numeric 2R used on the tab and in the widget
num = {m1: 2.0, m2: 1.0, L1: 1.0, L2: 0.8, g: 9.81}
qs_ = sp.symbols('a b'); qds_ = sp.symbols('ad bd'); qdds_ = sp.symbols('add bdd')
sub = {sp.diff(q1, t, 2): qdds_[0], sp.diff(q2, t, 2): qdds_[1], d1: qds_[0], d2: qds_[1], q1: qs_[0], q2: qs_[1]}
tau_f = sp.lambdify([qs_, qds_, qdds_], [sp.sympify(x).subs(sub).subs(num) for x in tau], 'numpy')
M_f = sp.lambdify([qs_], M.subs(sub).subs(num), 'numpy'); C_f = sp.lambdify([qs_, qds_], C.subs(sub).subs(num), 'numpy'); G_f = sp.lambdify([qs_], G.subs(sub).subs(num), 'numpy')
qa, qda, qdda = (30*deg, 45*deg), (1.0, -0.5), (0.5, 2.0)
tau_num = np.array(tau_f(qa, qda, qdda), float)
print('2R example (m=2,1 kg; L=1,0.8 m) q=(30°,45°), qd=(1,-0.5), qdd=(0.5,2):')
print('   M =', np.round(np.array(M_f(qa), float), 4).tolist(), ' C =', np.round(np.array(C_f(qa, qda), float).ravel(), 4).tolist(), ' g =', np.round(np.array(G_f(qa), float).ravel(), 4).tolist())
print('   tau =', np.round(tau_num, 4).tolist())
out['twoR'] = dict(m=[2, 1], L=[1, 0.8], q_deg=[30, 45], qd=list(qda), qdd=list(qdda), M=np.round(np.array(M_f(qa), float), 4).tolist(),
                   C=np.round(np.array(C_f(qa, qda), float).ravel(), 4).tolist(), G=np.round(np.array(G_f(qa), float).ravel(), 4).tolist(), tau=np.round(tau_num, 4).tolist())

# ===== Newton-Euler for the same 2R (homework p.46): outward then inward recursion, Craig style =====
def ne_2R(q, qd, qdd, m=(2.0, 1.0), L=(1.0, 0.8), grav=9.81):
    Z = np.array([0, 0, 1.0])
    Rs = [Rz(q[0])[:3, :3], Rz(q[1])[:3, :3]]           # 0R1, 1R2
    Ps = [np.zeros(3), np.array([L[0], 0, 0])]           # origin of {i} in {i-1}
    Pc = [np.array([L[0], 0, 0]), np.array([L[1], 0, 0])]  # point mass at link tip
    w = np.zeros(3); wd = np.zeros(3); vd = np.array([0, grav, 0.0])  # gravity trick: base accelerates up
    F, N, ws, wds, vds = [], [], [], [], []
    for i in range(2):
        R = Rs[i]
        vd = R.T @ (np.cross(wd, Ps[i]) + np.cross(w, np.cross(w, Ps[i])) + vd)
        wd = R.T @ wd + np.cross(R.T @ w, qd[i]*Z) + qdd[i]*Z
        w = R.T @ w + qd[i]*Z
        vc = np.cross(wd, Pc[i]) + np.cross(w, np.cross(w, Pc[i])) + vd
        F.append(m[i]*vc); N.append(np.zeros(3))         # point masses: no rotational inertia
    f = np.zeros(3); n = np.zeros(3); taus = [0, 0]
    for i in (1, 0):
        R_next = Rs[i + 1] if i + 1 < 2 else np.eye(3); P_next = Ps[i + 1] if i + 1 < 2 else np.zeros(3)
        f_new = R_next @ f + F[i]
        n = N[i] + R_next @ n + np.cross(Pc[i], F[i]) + np.cross(P_next, R_next @ f)
        f = f_new; taus[i] = n @ Z
    return np.array(taus)
close('Newton-Euler 2R == Lagrange 2R at the example state (slide p.46 homework)', ne_2R(qa, qda, qdda), tau_num)
rng = np.random.default_rng(2); okne = True
for _ in range(50):
    a_, b_, c_ = rng.uniform(-2, 2, 2), rng.uniform(-2, 2, 2), rng.uniform(-2, 2, 2)
    okne &= np.allclose(ne_2R(a_, b_, c_), np.array(tau_f(a_, b_, c_), float), atol=1e-9)
check('Newton-Euler == Lagrange at 50 random states', okne)

# free swing energy conservation (widget test vector)
def rhs(_, s):
    qq, vv = s[:2], s[2:]
    acc = np.linalg.solve(np.array(M_f(qq), float), -np.array(C_f(qq, vv), float).ravel() - np.array(G_f(qq), float).ravel())
    return np.concatenate([vv, acc])
E_f = sp.lambdify([qs_, qds_], (K + P).subs(sub).subs(num), 'numpy')
sol = solve_ivp(rhs, [0, 2], [30*deg, 0, 0, 0], rtol=1e-11, atol=1e-11, dense_output=True)
E = [E_f(sol.y[:2, k], sol.y[2:, k]) for k in range(sol.y.shape[1])]
check('2R free swing from (30°, 0°) at rest conserves energy over 2 s', np.ptp(E) < 1e-7, f'drift {np.ptp(E):.1e}')
s2end = sol.sol(2.0)
print(f'   free swing state at t=2 s: q=({s2end[0]/deg:.3f}°, {s2end[1]/deg:.3f}°)')
out['swing'] = dict(q0_deg=[30, 0], T=2.0, qT_deg=[r(s2end[0]/deg, 3), r(s2end[1]/deg, 3)], E0=r(E[0], 6))

# ===== computational cost (p.47) =====
ne_m = lambda n: 104*n - 77; ne_a = lambda n: 92*n - 70
la_m = lambda n: 32*n**4 + 86*n**3 + 171*n**2 + 53*n - 128; la_a = lambda n: 25*n**4 + 66*n**3 + 129*n**2 + 42*n - 96
check('cost n=6: NE 547 mult, 482 add; Lagrange 66394 mult, 51456 add (p.47)', (ne_m(6), ne_a(6), la_m(6), la_a(6)) == (547, 482, 66394, 51456))

# ===== inertia tensor of a cuboid (pp.36-39) =====
W, Lc, H, m = sp.symbols('W L H m', positive=True)
x, y, z = sp.symbols('x y z'); rho = m/(W*Lc*H)
I = lambda f: sp.integrate(sp.integrate(sp.integrate(f*rho, (x, 0, W)), (y, 0, Lc)), (z, 0, H))
Icorner = sp.Matrix([[I(y**2 + z**2), -I(x*y), -I(x*z)], [-I(x*y), I(x**2 + z**2), -I(y*z)], [-I(x*z), -I(y*z), I(x**2 + y**2)]])
Icorner_slide = sp.Matrix([[m*(Lc**2 + H**2)/3, -m*W*Lc/4, -m*H*W/4], [-m*W*Lc/4, m*(W**2 + H**2)/3, -m*H*Lc/4], [-m*H*W/4, -m*H*Lc/4, m*(Lc**2 + W**2)/3]])
check('cuboid inertia at a corner == slide p.38', sp.simplify(Icorner - Icorner_slide) == sp.zeros(3, 3))
Ic = sp.diag(m*(Lc**2 + H**2)/12, m*(W**2 + H**2)/12, m*(Lc**2 + W**2)/12)
p = sp.Matrix([W/2, Lc/2, H/2])
check('parallel-axis: I_A = I_C + m(p^T p 1 - p p^T) reproduces the corner tensor (added)', sp.simplify(Ic + m*((p.T*p)[0]*sp.eye(3) - p*p.T) - Icorner) == sp.zeros(3, 3))
In = np.array(Icorner_slide.subs({m: 6, W: 0.2, Lc: 0.3, H: 0.1}), dtype=float)
print('   cuboid m=6 kg, W=0.2, L=0.3, H=0.1 m, corner tensor (kg m²):', np.round(In, 4).tolist())
out['cuboid'] = dict(m=6, W=0.2, L=0.3, H=0.1, I_corner=np.round(In, 4).tolist(), I_centroid=np.round(np.array(Ic.subs({m: 6, W: 0.2, Lc: 0.3, H: 0.1}), float), 4).tolist())

# ===== W8 practical Q1: single link, P_c = [3,0,0], cI = diag(1,2,3), q = b t + c t^2 =====
b, c = sp.symbols('b c')
qq = b*t + c*t**2
w = sp.Matrix([0, 0, sp.diff(qq, t)]); wd = sp.Matrix([0, 0, sp.diff(qq, t, 2)])
Pc = sp.Matrix([3, 0, 0])
vdc = wd.cross(Pc) + w.cross(w.cross(Pc))
check('prac Q1: angular acceleration = [0, 0, 2c]', sp.simplify(wd - sp.Matrix([0, 0, 2*c])) == sp.zeros(3, 1))
check('prac Q1: COM acceleration = [-3(b+2ct)², 6c, 0]', sp.simplify(vdc - sp.Matrix([-3*(b + 2*c*t)**2, 6*c, 0])) == sp.zeros(3, 1))
Icm = sp.diag(1, 2, 3); F = 1*vdc; N = Icm*wd + w.cross(Icm*w)
tau_z = (N + Pc.cross(F))[2]
check('prac Q1: joint torque about z = 24c = (Izz + m|Pc|²) qdd (parallel axis: 3 + 9 = 12)', sp.simplify(tau_z - 24*c) == 0)

# ===== W8 practical Q3: planar 3R, L = 3, 2, 1 m; (5,1,90°) -> (-3,3,180°) at rest; tip point masses 3, 2, 1 kg =====
Lp = (3.0, 2.0, 1.0)
def ik3(xt, yt, ph):
    xw, yw = xt - Lp[2]*np.cos(ph), yt - Lp[2]*np.sin(ph)
    cth = (xw**2 + yw**2 - Lp[0]**2 - Lp[1]**2)/(2*Lp[0]*Lp[1]); cth = np.clip(cth, -1, 1)
    sols = []
    for sgn in (+1, -1):
        t2 = np.arctan2(sgn*np.sqrt(1 - cth**2), cth); t1 = np.arctan2(yw, xw) - np.arctan2(Lp[1]*np.sin(t2), Lp[0] + Lp[1]*np.cos(t2))
        sols.append(np.array([t1, t2, ph - t1 - t2]))
    return sols
qs0 = ik3(5, 1, 90*deg); qsf = ik3(-3, 3, np.pi)
check('prac Q3: start pose is the fully stretched arm, q0 = (0°, 0°, 90°) (both branches coincide)', np.allclose(qs0[0], [0, 0, np.pi/2]) and np.allclose(qs0[1], qs0[0]))
print('   prac Q3 goal branches (deg):', [np.round(s/deg, 3).tolist() for s in qsf])
qf = qsf[0]
check('prac Q3: chosen goal q_f = (90°, 90°, 0°)', np.allclose(qf/deg, [90, 90, 0]))
T = 2.0
a2c = 3*(qf - qs0[0])/T**2; a3c = -2*(qf - qs0[0])/T**3
print('   cubic (T=2 s): a2 =', np.round(a2c/deg, 4).tolist(), 'deg/s²; a3 =', np.round(a3c/deg, 4).tolist(), 'deg/s³')
# dynamics of planar 3R point masses at tips, gravity -y (vertical plane, assumption)
mp_ = (3.0, 2.0, 1.0)
qq3 = [sp.Function(f'p{i}')(t) for i in (1, 2, 3)]
pts = []; xa = 0; ya = 0; ang = 0
for i in range(3):
    ang = ang + qq3[i]; xa = xa + Lp[i]*sp.cos(ang); ya = ya + Lp[i]*sp.sin(ang); pts.append((xa, ya))
K3 = sum(sp.Rational(1, 2)*mp_[i]*(sp.diff(pts[i][0], t)**2 + sp.diff(pts[i][1], t)**2) for i in range(3))
P3 = sum(mp_[i]*9.81*pts[i][1] for i in range(3))
tau3, M3, C3, G3 = lagrange(qq3, K3, P3)
S3 = sp.symbols('s1:4'); SD = sp.symbols('sd1:4'); SDD = sp.symbols('sdd1:4')
sub3 = {**{sp.diff(qq3[i], t, 2): SDD[i] for i in range(3)}, **{sp.diff(qq3[i], t): SD[i] for i in range(3)}, **{qq3[i]: S3[i] for i in range(3)}}
tau3_f = sp.lambdify([S3, SD, SDD], [sp.sympify(x_).subs(sub3) for x_ in tau3], 'numpy')
rows = []
for tt in (0.0, 1.0, 2.0):
    qv = qs0[0] + a2c*tt**2 + a3c*tt**3; qdv = 2*a2c*tt + 3*a3c*tt**2; qddv = 2*a2c + 6*a3c*tt
    tv = np.array(tau3_f(qv, qdv, qddv), float)
    rows.append(dict(t=tt, q_deg=np.round(qv/deg, 3).tolist(), qd=np.round(qdv, 4).tolist(), qdd=np.round(qddv, 4).tolist(), tau=np.round(tv, 3).tolist()))
    print(f'   t={tt}: q={np.round(qv/deg,2)} deg, qdd={np.round(qddv,4)}, tau={np.round(tv,3)} N·m')
G0 = np.array([sp.sympify(x_).subs(sub3).subs({**{SD[i]: 0 for i in range(3)}, **{SDD[i]: 0 for i in range(3)}}).subs({S3[i]: qs0[0][i] for i in range(3)}) for i in range(3) for x_ in [G3[i]]], float)
check('prac Q3: at t=0 the arm is at rest, so tau = M qdd + g (no velocity terms)', np.allclose(rows[0]['tau'], np.array(M3.subs(sub3).subs({S3[i]: qs0[0][i] for i in range(3)}), float) @ (2*a2c) + G0, atol=1e-3))
out['prac3'] = dict(L=list(Lp), m=list(mp_), T=T, q0_deg=[0, 0, 90], qf_deg=[90, 90, 0], goal_branches_deg=[np.round(s/deg, 3).tolist() for s in qsf],
                    a2_deg=np.round(a2c/deg, 4).tolist(), a3_deg=np.round(a3c/deg, 4).tolist(), rows=rows)

# ===== RP arm (Craig Ex. 6.5; the unit's rp_lagrangian_dynamics.py) =====
th, dd = sp.Function('th')(t), sp.Function('d')(t)
m1r, m2r, l1, Iz1, Iz2 = sp.symbols('m_1 m_2 l_1 I_zz1 I_zz2', positive=True)
Krp = sp.Rational(1, 2)*m1r*l1**2*sp.diff(th, t)**2 + sp.Rational(1, 2)*Iz1*sp.diff(th, t)**2 + sp.Rational(1, 2)*m2r*(dd**2*sp.diff(th, t)**2 + sp.diff(dd, t)**2) + sp.Rational(1, 2)*Iz2*sp.diff(th, t)**2
Prp = m1r*l1*g*sp.sin(th) + m2r*g*dd*sp.sin(th)
taurp, Mrp, Crp, Grp = lagrange([th, dd], Krp, Prp)
check('RP: M = diag(m1 l1² + Izz1 + Izz2 + m2 d², m2)', sp.simplify(Mrp - sp.diag(m1r*l1**2 + Iz1 + Iz2 + m2r*dd**2, m2r)) == sp.zeros(2, 2))
check('RP: V = [2 m2 d thd dd; -m2 d thd²]', sp.simplify(Crp - sp.Matrix([2*m2r*dd*sp.diff(th, t)*sp.diff(dd, t), -m2r*dd*sp.diff(th, t)**2])) == sp.zeros(2, 1))
check('RP: G = [(m1 l1 + m2 d) g cos th; m2 g sin th]', sp.simplify(Grp - sp.Matrix([(m1r*l1 + m2r*dd)*g*sp.cos(th), m2r*g*sp.sin(th)])) == sp.zeros(2, 1))
TH, D_, THD, DD_, THDD, DDD = sp.symbols('TH D THD DDot THDD DDD')
subrp = {sp.diff(th, t, 2): THDD, sp.diff(dd, t, 2): DDD, sp.diff(th, t): THD, sp.diff(dd, t): DD_, th: TH, dd: D_}
vals = {m1r: 2, m2r: 1, l1: 0.3, Iz1: 0.05, Iz2: 0.02, g: 9.81, TH: 0.7, D_: 0.6, THD: 1.3, DD_: -0.8, THDD: 2.0, DDD: 1.5}
trp = [float(sp.sympify(x_).subs(subrp).subs(vals)) for x_ in taurp]
print(f'   RP at (0.7 rad, 0.6 m; 1.3, -0.8; 2.0, 1.5): tau = ({trp[0]:.4f} N·m, {trp[1]:.4f} N)')
out['rp'] = dict(tau=[r(trp[0]), r(trp[1])])
json.dump(out, open(os.path.join(os.path.dirname(__file__), 'w07_vectors.json'), 'w'), indent=1)
report()
