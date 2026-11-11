"""Practice Q&A part 2 (W7-W12): every number used in frag/qa/part2.html is computed here."""
import heapq
import itertools
import math
import sympy as sp
import numpy as np
from common import check, close, report

g0 = 9.81
d2r = math.pi / 180


def show(name, val, nd=4):
    v = round(float(val), nd)
    print(f'  {name} = {v}')
    return v


# ============================ W7 ============================
print('--- W7')
# qa-w07-1: cart + pendulum (Lagrange, sympy) --------------------------------
t = sp.symbols('t')
Mc, mp, l, G = sp.symbols('Mc mp l G', positive=True)
x = sp.Function('x')(t)
ph = sp.Function('ph')(t)
K = sp.Rational(1, 2) * Mc * x.diff(t)**2 + sp.Rational(1, 2) * mp * ((x.diff(t) + l * sp.cos(ph) * ph.diff(t))**2 + (l * sp.sin(ph) * ph.diff(t))**2)
P = -mp * G * l * sp.cos(ph)
L = K - P
Fx = sp.simplify(sp.diff(sp.diff(L, x.diff(t)), t) - sp.diff(L, x))
Fp = sp.simplify(sp.diff(sp.diff(L, ph.diff(t)), t) - sp.diff(L, ph))
# claimed closed forms
xd, xdd, pd_, pdd, pp = sp.symbols('xd xdd pd pdd pp')
sub = {x.diff(t, 2): xdd, ph.diff(t, 2): pdd, x.diff(t): xd, ph.diff(t): pd_, ph: pp}
Fx_s = sp.simplify(Fx.subs(sub))
Fp_s = sp.simplify(Fp.subs(sub))
claim_x = (Mc + mp) * xdd + mp * l * pdd * sp.cos(pp) - mp * l * pd_**2 * sp.sin(pp)
claim_p = mp * l**2 * pdd + mp * l * xdd * sp.cos(pp) + mp * G * l * sp.sin(pp)
check('W7.1 F_x symbolic', sp.simplify(Fx_s - claim_x) == 0)
check('W7.1 tau_phi symbolic', sp.simplify(Fp_s - claim_p) == 0)
vals = {Mc: 2, mp: 1, l: 0.5, G: g0, pp: 30 * d2r, pd_: 2, pdd: 3, xdd: 1.5}
F_cart = show('F', float(claim_x.subs(vals)))
tau_ph = show('tau_phi', float(claim_p.subs(vals)))
F_wrong = show('F with +ml phid^2 sin', float((claim_x + 2 * mp * l * pd_**2 * sp.sin(pp)).subs(vals)))
mm = 1; ll = 0.5
check('W7.1 hand F', abs(3 * 1.5 + mm * ll * 3 * math.cos(30 * d2r) - mm * ll * 4 * math.sin(30 * d2r) - F_cart) < 1e-3)
print('  hand pieces: 3*1.5=4.5; m l phidd cos =', round(mm * ll * 3 * math.cos(30 * d2r), 4), '; m l phid^2 sin =', round(mm * ll * 4 * math.sin(30 * d2r), 4))
print('  tau pieces: m l^2 phidd =', mm * ll**2 * 3, '; m l xdd cos =', round(mm * ll * 1.5 * math.cos(30 * d2r), 4), '; m g l sin =', round(mm * g0 * ll * math.sin(30 * d2r), 4))

# qa-w07-2: cuboid inertia about the midpoint of the bottom edge (parallel axis) ---
m, W, Ly, H = 4, 0.3, 0.2, 0.5
IC = m / 12 * np.diag([Ly**2 + H**2, W**2 + H**2, Ly**2 + W**2])
p = np.array([0, 0.1, 0.25])
IA = IC + m * (p @ p * np.eye(3) - np.outer(p, p))
# brute-force integral check
rng = np.random.default_rng(1)
N = 400000
pts = rng.uniform([-W / 2, -Ly / 2, -H / 2], [W / 2, Ly / 2, H / 2], size=(N, 3)) + p
rho = m / (W * Ly * H); dv = W * Ly * H / N
Ixx_mc = rho * dv * np.sum(pts[:, 1]**2 + pts[:, 2]**2)
Iyz_mc = -rho * dv * np.sum(pts[:, 1] * pts[:, 2])
check('W7.2 Monte-Carlo Ixx', abs(Ixx_mc - IA[0, 0]) < 0.01 * IA[0, 0], f'{Ixx_mc:.4f} vs {IA[0,0]:.4f}')
check('W7.2 Monte-Carlo Iyz', abs(Iyz_mc - IA[1, 2]) < 0.01, f'{Iyz_mc:.4f} vs {IA[1,2]:.4f}')
print('  I_C diag =', np.round(np.diag(IC), 5).tolist())
Ixx_A = show('Ixx_A', IA[0, 0], 5)
Iyz_A = show('Iyz_A', IA[1, 2], 5)
show('Iyy_A', IA[1, 1], 5)
show('Izz_A', IA[2, 2], 5)
show('Ixx wrong (no parallel axis)', IC[0, 0], 5)
show('Iyz wrong (+)', -IA[1, 2], 5)

# qa-w07-3: 2R arm full Q, cross-checked against sympy Lagrange --------------
q1, q2, v1, v2, a1, a2 = sp.symbols('q1 q2 v1 v2 a1 a2')
m1_, m2_, L1_, L2_ = sp.symbols('m1 m2 L1 L2', positive=True)
Mm = sp.Matrix([[m1_ * L1_**2 + m2_ * (L1_**2 + 2 * L1_ * L2_ * sp.cos(q2) + L2_**2), m2_ * (L1_ * L2_ * sp.cos(q2) + L2_**2)],
                [m2_ * (L1_ * L2_ * sp.cos(q2) + L2_**2), m2_ * L2_**2]])
Cv = sp.Matrix([-m2_ * L1_ * L2_ * sp.sin(q2) * (2 * v1 * v2 + v2**2), m2_ * L1_ * L2_ * sp.sin(q2) * v1**2])
gv = sp.Matrix([(m1_ + m2_) * G * L1_ * sp.cos(q1) + m2_ * G * L2_ * sp.cos(q1 + q2), m2_ * G * L2_ * sp.cos(q1 + q2)])
Qform = Mm * sp.Matrix([a1, a2]) + Cv + gv
# independent Lagrange
T_ = sp.symbols('T')
Q1f, Q2f = sp.Function('Q1')(T_), sp.Function('Q2')(T_)
qa, qb = sp.Function('qa')(T_), sp.Function('qb')(T_)
x2 = L1_ * sp.cos(qa) + L2_ * sp.cos(qa + qb); y2 = L1_ * sp.sin(qa) + L2_ * sp.sin(qa + qb)
x1 = L1_ * sp.cos(qa); y1 = L1_ * sp.sin(qa)
Kk = sp.Rational(1, 2) * m1_ * (x1.diff(T_)**2 + y1.diff(T_)**2) + sp.Rational(1, 2) * m2_ * (x2.diff(T_)**2 + y2.diff(T_)**2)
Pp = G * (m1_ * y1 + m2_ * y2)
Ll = Kk - Pp
tq = [sp.diff(sp.diff(Ll, qq.diff(T_)), T_) - sp.diff(Ll, qq) for qq in (qa, qb)]
sub2 = {qa.diff(T_, 2): a1, qb.diff(T_, 2): a2, qa.diff(T_): v1, qb.diff(T_): v2, qa: q1, qb: q2}
tq = [sp.simplify(e.subs(sub2)) for e in tq]
for i in range(2):
    check(f'W7.3 Q{i+1} symbolic equals M,C,g form', sp.simplify(tq[i] - Qform[i]) == 0)
nv = {m1_: 1.5, m2_: 0.8, L1_: 0.9, L2_: 0.6, G: g0, q1: 60 * d2r, q2: -30 * d2r, v1: 0.8, v2: 1.2, a1: -1.0, a2: 2.5}
Qn = [float(e.subs(nv)) for e in Qform]
Mn = np.array(Mm.subs(nv).tolist(), dtype=float)
Cn = np.array(Cv.subs(nv).tolist(), dtype=float).ravel()
gn = np.array(gv.subs(nv).tolist(), dtype=float).ravel()
print('  M =', np.round(Mn, 4).tolist(), ' C =', np.round(Cn, 4).tolist(), ' g =', np.round(gn, 4).tolist())
show('M*qdd', 0)
print('  M qdd =', np.round(Mn @ np.array([-1.0, 2.5]), 4).tolist())
Q1n = show('tau1', Qn[0]); Q2n = show('tau2', Qn[1])
show('tau1 without C', Qn[0] - Cn[0]); show('tau2 without C', Qn[1] - Cn[1])
show('tau2 with sin in g', Qn[1] - gn[1] + float(m2_.subs(nv) * g0 * 0.6 * math.sin(30 * d2r)))

# qa-w07-4: Q = g + J^T w at rest -----------------------------------------------
mA, LA = (2, 1), (1.0, 0.8)
qA = np.array([30, 45]) * d2r
c1, s1, c12, s12 = math.cos(qA[0]), math.sin(qA[0]), math.cos(qA.sum()), math.sin(qA.sum())
gA = np.array([(mA[0] + mA[1]) * g0 * LA[0] * c1 + mA[1] * g0 * LA[1] * c12, mA[1] * g0 * LA[1] * c12])
J = np.array([[-LA[0] * s1 - LA[1] * s12, -LA[1] * s12], [LA[0] * c1 + LA[1] * c12, LA[1] * c12]])
w = np.array([-5.0, 12.0])
QA = gA + J.T @ w
print('  g =', np.round(gA, 4).tolist(), ' J =', np.round(J, 4).tolist(), ' J^T w =', np.round(J.T @ w, 4).tolist())
QA1 = show('tau1 static', QA[0]); QA2 = show('tau2 static', QA[1])
show('tau1 with J w (no transpose)', gA[0] + (J @ w)[0]); show('tau1 g only', gA[0])
# virtual-work check on joint 1: moment of tip force about joint 1
tip = np.array([LA[0] * c1 + LA[1] * c12, LA[0] * s1 + LA[1] * s12])
check('W7.4 moment check joint 1', abs((tip[0] * w[1] - tip[1] * w[0]) - (J.T @ w)[0]) < 1e-9)

# qa-w07-5: operation counts ---------------------------------------------------
def lag_mul(n): return 32 * n**4 + 86 * n**3 + 171 * n**2 + 53 * n - 128
def ne_mul(n): return 104 * n - 77
check('W7.5 slide n=6 values', lag_mul(6) == 66394 and ne_mul(6) == 547)
show('NE n=5', ne_mul(5), 0); show('Lag n=5', lag_mul(5), 0); show('ratio n=5', lag_mul(5) / ne_mul(5), 1)
show('NE n=12', ne_mul(12), 0)

# qa-w07-6: RP arm (point mass on prismatic link) -------------------------------
th, dd, thd, ddd, thdd, dddd = sp.symbols('th d thd dd thdd ddd')
tt = sp.symbols('tt')
thf, df = sp.Function('thf')(tt), sp.Function('df')(tt)
m2s = sp.symbols('m2s', positive=True)
K2 = sp.Rational(1, 2) * m2s * (df.diff(tt)**2 + df**2 * thf.diff(tt)**2)
P2 = m2s * G * df * sp.sin(thf)
L2f = K2 - P2
f2 = sp.diff(sp.diff(L2f, df.diff(tt)), tt) - sp.diff(L2f, df)
tau1p = sp.diff(sp.diff(L2f, thf.diff(tt)), tt) - sp.diff(L2f, thf)
s3 = {df.diff(tt, 2): dddd, thf.diff(tt, 2): thdd, df.diff(tt): ddd, thf.diff(tt): thd, df: dd, thf: th}
f2s = sp.simplify(f2.subs(s3)); tau1s = sp.simplify(tau1p.subs(s3))
check('W7.6 f2 symbolic', sp.simplify(f2s - m2s * (dddd - dd * thd**2) - m2s * G * sp.sin(th)) == 0)
check('W7.6 tau1 symbolic', sp.simplify(tau1s - (m2s * dd**2 * thdd + 2 * m2s * dd * ddd * thd + m2s * G * dd * sp.cos(th))) == 0)
nv6 = {m2s: 1.5, dd: 0.4, dddd: 0.5, th: 0.9, thd: 1.1, ddd: 0.3, thdd: 0.6, G: g0}
f2n = show('f2 correct', float(f2s.subs(nv6)))
f2w = show('f2 student (m d dd + m g cos)', 1.5 * 0.5 + 1.5 * g0 * math.cos(0.9))
show('tau1 correct', float(tau1s.subs(nv6)))
show('centrifugal term -m d thd^2', -1.5 * 0.4 * 1.1**2)

# ============================ W8 ============================
print('--- W8')
# qa-w08-1: model error in b only, tracking with feed-forward ---------------------
Es, Esd, Esdd, ft, fd_, xs, xsd, xdd_d, vd = sp.symbols('E Ed Edd ft fdst xs xsd xddd vd')
m, b, bh, k, kp, kv = sp.symbols('m b bh k kp kv', positive=True)
# plant: m xdd + b xd + k x = f - fd ; law f = m f' + bh xd + k x ; E = xd_ - x
xddot = sp.symbols('xddot')
fprime = xdd_d + kv * Esd + kp * Es
f_law = m * fprime + bh * xsd + k * xs
xdd_sol = sp.solve(sp.Eq(m * xddot + b * xsd + k * xs, f_law - fd_), xddot)[0]
Eddot = xdd_d - xdd_sol
rhs = sp.simplify(Eddot + kv * Esd + kp * Es)
claim = fd_ / m + (b - bh) * xsd / m
check('W8.1 error equation RHS', sp.simplify(rhs - claim) == 0)
Ess = (6 + (5 - 3) * 0.4) / (3 * 36)
show('E_ss tracking', Ess, 6)
show('E_ss if b-error ignored', 6 / (3 * 36), 6)
show('shifted k_v', 12 + (5 - 3) / 3, 4)
# qa-w08-2: spring-mass-damper, underdamped with nonzero x'(0) ----------------------
tq_ = sp.symbols('tq')
Xf = sp.Function('X')
sol = sp.dsolve(sp.Eq(1 * Xf(tq_).diff(tq_, 2) + 2 * Xf(tq_).diff(tq_) + 5 * Xf(tq_), 0), Xf(tq_), ics={Xf(0): 1, Xf(tq_).diff(tq_).subs(tq_, 0): 3})
xt = sp.lambdify(tq_, sol.rhs)
claimed = lambda tv: math.exp(-tv) * (math.cos(2 * tv) + 2 * math.sin(2 * tv))
check('W8.2 dsolve matches e^-t(cos2t+2 sin2t)', all(abs(xt(tv) - claimed(tv)) < 1e-9 for tv in (0, 0.3, 0.5, 1.7)))
A2 = show('A2', 2, 4)
x05 = show('x(0.5)', claimed(0.5), 4)
show('x(0.5) with A2=0', math.exp(-0.5) * math.cos(1.0), 4)
disc = 2**2 - 4 * 1 * 5
print('  discriminant =', disc)
tpk = math.atan(2.0) / 2  # first stationary point? x' = 0
xp = lambda tv: (claimed(tv + 1e-6) - claimed(tv - 1e-6)) / 2e-6
print('  x(1) =', round(claimed(1.0), 4), ' x(0.5)=', round(claimed(0.5), 4), " x'(0)=", round(xp(0.0), 4))
# qa-w08-3: partitioned control numbers -------------------------------------------
mq, bq, kq = 1.5, 4.0, 10.0
kpq, kvq = 49.0, 14.0
xq, xqd = 0.3, -0.2
xdq, xddq, xdddq = 0.5, 0.1, 0.8
Eq, Edq = xdq - xq, xddq - xqd
fp = xdddq + kvq * Edq + kpq * Eq
fq = mq * fp + bq * xqd + kq * xq
fwrong = mq * fp + bq * xddq + kq * xdq
show('E', Eq); show('Edot', Edq)
fpq = show("f'", fp); fqq = show('f', fq)
show('f wrong (beta with desired)', fwrong)
check('W8.3 plant gives xdd=f\'', abs((fq - bq * xqd - kq * xq) / mq - fp) < 1e-12)
check('W8.3 critical', kvq**2 == 4 * kpq)
# qa-w08-4: gear train ---------------------------------------------------------------
eta, Im, I, c, cm, k4 = 40, 3e-5, 0.05, 0.2, 2e-4, 0.05
qd_, qdd_ = 3.0, 8.0
Ieff, ceff = I + eta**2 * Im, c + eta**2 * cm
Qax = Ieff * qdd_ + ceff * qd_
Qm = Qax / eta; ia = Qm / k4
show('I_eff', Ieff); show('c_eff', ceff)
Qax_s = show('Q axle', Qax); ia_s = show('i_a', ia); show('Q_m', Qm, 5)
Iw, cw = I + eta * Im, c + eta * cm
show('i_a wrong (eta not eta^2)', (Iw * qdd_ + cw * qd_) / eta / k4)
show('motor speed', eta * qd_, 1)
# derivation cross-check: motor torque = Im*eta*qdd + cm*eta*qd + (I qdd + c qd)/eta
check('W8.4 motor torque by derivation', abs(Im * eta * qdd_ + cm * eta * qd_ + (I * qdd_ + c * qd_) / eta - Qm) < 1e-12)
# qa-w08-5: steady-state error and PID limit -----------------------------------------------
mE, fdE, kpE = 4, 12, 75
show('E_ss', fdE / (kpE * mE), 4)
show('kp needed (2 mm)', fdE / (mE * 0.002), 1)
show('kv critical for kp=1500', 2 * math.sqrt(1500), 2)
show('kv critical for kp=75', 2 * math.sqrt(75), 3)
show('ki limit = kv kp (kp=75)', 2 * math.sqrt(75) * 75, 1)
# stability check by roots
def stable(kv_, kp_, ki_): return max(np.roots([1, kv_, kp_, ki_]).real) < 0
kvc = 2 * math.sqrt(75)
check('W8.5 stable below limit', stable(kvc, 75, 0.9 * kvc * 75))
check('W8.5 unstable above limit', not stable(kvc, 75, 1.1 * kvc * 75))
# qa-w08-6: spot, critical damping -----------------------------------------------------------
show('kv (kp=36)', 2 * math.sqrt(36), 1)
show('wrong kv from kp^2=4kv', 36**2 / 4, 1)
print('  roots with kv=324,kp=36:', np.round(np.roots([1, 324, 36]), 4).tolist())
show('E_ss indep. of kv', 10 / (36 * 2.5), 4)

# ============================ W9 ============================
print('--- W9')
# qa-w09-1: partitioned single link with only gravity mass wrong ----------------
mA9, mh, L9, kp9 = 3.0, 2.5, 0.6, 81.0
q9 = 20 * d2r
Ess9 = (mA9 - mh) * g0 * math.cos(q9) / (mA9 * L9 * kp9)
# symbolic check of error equation
qs, qds, qdds, qdd_d, Es9, Eds9, kv9, kp9s = sp.symbols('qs qds qdds qddd Es9 Eds9 kv9 kp9s')
m9, mh9, L9s = sp.symbols('m9 mh9 L9s', positive=True)
alpha = m9 * L9s**2
beta_hat = mh9 * G * L9s * sp.cos(qs)
tau_law = alpha * (qdd_d + kv9 * Eds9 + kp9s * Es9) + beta_hat
qdd_true = sp.solve(sp.Eq(m9 * L9s**2 * qdds + m9 * G * L9s * sp.cos(qs), tau_law), qdds)[0]
rhs9 = sp.simplify((qdd_d - qdd_true) + kv9 * Eds9 + kp9s * Es9)
check('W9.1 error equation RHS', sp.simplify(rhs9 - (m9 - mh9) * G * sp.cos(qs) / (m9 * L9s)) == 0)
show('E_ss', Ess9, 6); show('E_ss deg', Ess9 / d2r, 4)
show('forcing', (mA9 - mh) * g0 * math.cos(q9) / (mA9 * L9), 4)
show('E_ss at 90 deg', (mA9 - mh) * g0 * math.cos(math.pi / 2) / (mA9 * L9 * kp9), 6)
# qa-w09-2: Q = J^T F
L1_9, L2_9 = 0.5, 0.4
q9b = np.array([60, -30]) * d2r
c1, s1 = math.cos(q9b[0]), math.sin(q9b[0]); c12, s12 = math.cos(q9b.sum()), math.sin(q9b.sum())
J9 = np.array([[-L1_9 * s1 - L2_9 * s12, -L2_9 * s12], [L1_9 * c1 + L2_9 * c12, L2_9 * c12]])
F9 = np.array([-2.0, 6.0])
Q9 = J9.T @ F9
tip9 = np.array([L1_9 * c1 + L2_9 * c12, L1_9 * s1 + L2_9 * s12])
p2 = np.array([L1_9 * c1, L1_9 * s1])
check('W9.2 moment about joint 1', abs(tip9[0] * F9[1] - tip9[1] * F9[0] - Q9[0]) < 1e-12)
r2 = tip9 - p2
check('W9.2 moment about joint 2', abs(r2[0] * F9[1] - r2[1] * F9[0] - Q9[1]) < 1e-12)
print('  J =', np.round(J9, 4).tolist(), ' tip =', np.round(tip9, 4).tolist())
Q9a = show('Q1', Q9[0]); Q9b_ = show('Q2', Q9[1])
show('Q1 wrong (J F)', (J9 @ F9)[0]); show('Q2 wrong (J F)', (J9 @ F9)[1])
# virtual work: small joint motion gives dx = J dq
dq = np.array([1e-6, -2e-6]); fk = lambda q: np.array([L1_9 * math.cos(q[0]) + L2_9 * math.cos(q.sum()), L1_9 * math.sin(q[0]) + L2_9 * math.sin(q.sum())])
dx = fk(q9b + dq) - fk(q9b)
check('W9.2 dx = J dq', np.allclose(dx, J9 @ dq, atol=1e-10))
check('W9.2 virtual work', abs(Q9 @ dq - F9 @ dx) < 1e-10)
show('det J (singular at q2=0)', np.linalg.det(J9), 4)
show('L1 L2 sin q2', L1_9 * L2_9 * math.sin(q9b[1]), 4)
# qa-w09-3: single link with Coulomb + viscous friction -----------------------------------
m3, L3, bv, bc = 1.2, 0.7, 0.4, 0.3
q3, qd3 = 50 * d2r, -0.8
qddd3, e3, ed3, kp3, kv3 = 1.5, 0.04, 0.2, 100.0, 20.0
al = m3 * L3**2
taup = qddd3 + kv3 * ed3 + kp3 * e3
beta = bv * qd3 + bc * np.sign(qd3) + m3 * g0 * L3 * math.cos(q3)
tau3 = al * taup + beta
show('alpha', al, 4); tp_s = show("tau'", taup, 4)
show('beta', beta, 4); tau3_s = show('tau', tau3, 4)
beta_w = bv * qd3 + bc * 1 + m3 * g0 * L3 * math.cos(q3)
tau3_w = show('tau wrong (sgn=+1)', al * taup + beta_w, 4)
check('W9.3 plant gives qdd = tau\'', abs((tau3 - beta) / al - taup) < 1e-12)
show('gravity term', m3 * g0 * L3 * math.cos(q3), 4)
# qa-w09-4: force control, steady-state errors ------------------------------------------------
m4, fd4, kp4, ke4 = 1.5, 8.0, 64.0, 300.0
xx = m4 / ke4 * kp4
E1 = fd4 / xx; E2 = fd4 / (1 + xx)
show('m/ke', m4 / ke4, 4); show('x', xx, 4); show('E1', E1, 4); E2s = show('E2', E2, 4)
show('E2/E1', E2 / E1, 4)
# symbolic: option 2 closed loop steady state
al4, kp4s, kv4s, fdist, Ef, fdes, fe = sp.symbols('al4 kp4s kv4s fdist Ef fdes fe')
# alpha*(Edd+kv Ed+kp E)= fdist + fe - fdes ; E = fdes - fe -> fe = fdes - E
rhs_o2 = fdist + (fdes - Ef) - fdes
check('W9.4 option-2 RHS is fdist - E', sp.simplify(rhs_o2 - (fdist - Ef)) == 0)
Ess_o2 = sp.solve(sp.Eq(al4 * kp4s * Ef, fdist - Ef), Ef)[0]
check('W9.4 option-2 E_ss', sp.simplify(Ess_o2 - fdist / (1 + al4 * kp4s)) == 0)
# qa-w09-5: hybrid
Fpos = np.array([2.0, -1.0, 6.0]); Fforce = np.array([0.5, 0.3, 12.0])
S = np.diag([1, 1, 0]); Sp = np.eye(3) - S
Fh = S @ Fpos + Sp @ Fforce
show('Fx', Fh[0]); show('Fy', Fh[1]); show('Fz', Fh[2])
check('W9.5 S + S\' = I', np.allclose(S + Sp, np.eye(3)))
Fsw = Sp @ Fpos + S @ Fforce
show('Fz if S,S\' swapped', Fsw[2]); show('Fx if swapped', Fsw[0])
# qa-w09-6: spot
m6, ke6, kp6, fd6 = 2.0, 800.0, 100.0, 6.0
x6 = m6 / ke6 * kp6
show('x6', x6, 4); show('E2 (spot)', fd6 / (1 + x6), 4); show('E1 (spot)', fd6 / x6, 4)

# ============================ W10 ============================
print('--- W10')
# qa-w10-1: 3 omni wheels, positions at 0, 120, 240 deg; tangential anticlockwise drive
d, r = 0.12, 0.04
u_, v_, w_ = 0.3, 0.1, 0.5
rows = []
for phi in (0, 120, 240):
    ph_ = phi * d2r; beta_ = ph_ + math.pi / 2
    xw, yw = d * math.cos(ph_), d * math.sin(ph_)
    row = (math.cos(beta_) * (u_ - w_ * yw) + math.sin(beta_) * (v_ + w_ * xw)) / r
    row2 = (-math.sin(ph_) * u_ + math.cos(ph_) * v_ + d * w_) / r
    check(f'W10.1 row phi={phi} matches closed form', abs(row - row2) < 1e-12)
    rows.append(row)
for i, rv in enumerate(rows): show(f'phidot{i+1}', rv, 4)
Wm = np.array([[-math.sin(p * d2r), math.cos(p * d2r), d] for p in (0, 120, 240)]) / r
print('  det of rows matrix * r^3 =', round(np.linalg.det(Wm * r), 5), ' expected 3*sqrt(3)*d/2 = ', round(3 * math.sqrt(3) * d / 2, 5))
show('phidot2 wrong (beta=-120 clockwise rows)', (-math.sin(120 * d2r) * u_ * -1 + math.cos(120 * d2r) * v_ - d * w_) / r, 3)
# qa-w10-2: exact arc update vs Euler
u2, w2, th0, T = 0.4, 0.8, 30 * d2r, 1.5
th1 = th0 + w2 * T
x2n = u2 / w2 * (math.sin(th1) - math.sin(th0)); y2n = -u2 / w2 * (math.cos(th1) - math.cos(th0))
# numeric integration check
n = 200000; xi = yi = 0.0; thc = th0
for _ in range(n):
    xi += u2 * math.cos(thc) * T / n; yi += u2 * math.sin(thc) * T / n; thc += w2 * T / n
check('W10.2 exact arc vs numeric integral', abs(xi - x2n) < 1e-4 and abs(yi - y2n) < 1e-4, f'{xi:.5f},{yi:.5f}')
show('x', x2n, 4); show('y', y2n, 4); show('theta1 deg', th1 / d2r, 3)
xe, ye = u2 * math.cos(th0) * T, u2 * math.sin(th0) * T
show('x Euler', xe, 4); show('y Euler', ye, 4)
show('R = u/w', u2 / w2, 3)
# qa-w10-3: differential drive, radius 1.2
u3, R3, a3, r3 = 0.5, 1.2, 0.22, 0.08
w3 = u3 / R3
pl = (u3 - a3 * w3) / r3; pr = (u3 + a3 * w3) / r3
show('thetadot', w3, 4); show('phi_L', pl, 3); show('phi_R', pr, 3)
check('W10.3 inverse recovers u,w', abs(r3 * (pl + pr) / 2 - u3) < 1e-12 and abs(r3 * (pr - pl) / (2 * a3) - w3) < 1e-12)
check('W10.3 ICR at R', abs(a3 * (pl + pr) / (pr - pl) - R3) < 1e-12)
# qa-w10-4: Ackermann
Lw, tk, Rr = 2.7, 1.5, 8.0
inner = math.atan(Lw / (Rr - tk / 2)) / d2r; outer = math.atan(Lw / (Rr + tk / 2)) / d2r
show('inner', inner, 3); show('outer', outer, 3); show('naive atan(L/R)', math.atan(Lw / Rr) / d2r, 3)
# every axle line passes through ICR: cot diff = track/wheelbase
check('W10.4 cot(out)... relation', abs(1 / math.tan(outer * d2r) - 1 / math.tan(inner * d2r) - tk / Lw) < 1e-12)
show('cot out - cot in', 1 / math.tan(outer * d2r) - 1 / math.tan(inner * d2r), 4)
# qa-w10-5: ICR
a5 = 0.25
show('yICR (3,7)', a5 * (3 + 7) / (7 - 3), 3)
show('yICR (7,3)', a5 * (7 + 3) / (3 - 7), 3)
dm = lambda d_: 3 * math.sqrt(3) * d_ / 2
show('det omni d=0.15', dm(0.15), 4)
# qa-w10-6: spot
r6 = 0.05
show('u correct', r6 * (8 + 12) / 2, 3); show('u student', r6 * (8 + 12), 3)
show('thetadot with a=0.2', r6 * (12 - 8) / (2 * 0.2), 3)
show('bicycle w u=1.5 b=25deg L=0.9', 1.5 * math.tan(25 * d2r) / 0.9, 4)

# ============================ W11 ============================
print('--- W11')
# qa-w11-1: M with offset COM
m1, IG, xG, yG = 8.0, 0.3, 0.05, 0.02
M = np.array([[m1, 0, -m1 * yG], [0, m1, m1 * xG], [-m1 * yG, m1 * xG, IG + m1 * (xG**2 + yG**2)]])
Qw = np.array([6.0, 0.0, 0.5])
acc = np.linalg.solve(M, Qw)
# independent derivation: forces and moment about G
ud, vd_, thdd_ = acc
fx_ = m1 * (ud - yG * thdd_); fy_ = m1 * (vd_ + xG * thdd_)
nz_ = IG * thdd_ - (yG * fx_ - xG * fy_)
check('W11.1 force balance', abs(fx_ - 6) < 1e-12 and abs(fy_ - 0) < 1e-12 and abs(nz_ - 0.5) < 1e-12)
print('  M =', np.round(M, 4).tolist())
show('M33', M[2, 2], 4); show('udot', ud, 4); show('vdot', vd_, 5); show('thetaddot', thdd_, 4)
Mdiag = np.diag([m1, m1, IG])
show('thetaddot if M33=I_G', np.linalg.solve(Mdiag, Qw)[2], 4)
# symbolic check of M structure
mx, Ix, xg, yg, u1, v1_, w1, ud_, vd_s, wd_ = sp.symbols('mx Ix xg yg u1 v1 w1 ud vds wd')
ax = ud_ - yg * wd_ - v1_ * w1 - xg * w1**2; ay = vd_s + xg * wd_ + u1 * w1 - yg * w1**2
fxs, fys = mx * ax, mx * ay
nzs = Ix * wd_ - (yg * fxs - xg * fys)
Msym = sp.Matrix([[mx, 0, -mx * yg], [0, mx, mx * xg], [-mx * yg, mx * xg, Ix + mx * (xg**2 + yg**2)]])
acc_vec = sp.Matrix([ud_, vd_s, wd_])
zero = {u1: 0, v1_: 0, w1: 0}
check('W11.1 M symbolic', sp.simplify(sp.Matrix([fxs, fys, nzs]).subs(zero) - Msym * acc_vec) == sp.zeros(3, 1))
# qa-w11-2: C Q = J^T O Q
th_ = 120 * d2r
Jr = np.array([[math.cos(th_), -math.sin(th_), 0], [math.sin(th_), math.cos(th_), 0], [0, 0, 1]])
OQ = np.array([4.0, -2.0, 0.7])
CQ = Jr.T @ OQ
show('fx', CQ[0], 4); show('fy', CQ[1], 4); show('nz', CQ[2], 4)
show('fx wrong (J)', (Jr @ OQ)[0], 4); show('fy wrong (J)', (Jr @ OQ)[1], 4)
check('W11.2 power invariance', abs(OQ @ (Jr @ np.array([0.5, 0.2, 0.3])) - CQ @ np.array([0.5, 0.2, 0.3])) < 1e-12)
check('W11.2 |f| preserved', abs(np.linalg.norm(CQ[:2]) - np.linalg.norm(OQ[:2])) < 1e-12)
show('|f|', np.linalg.norm(OQ[:2]), 4)
# qa-w11-3: differential drive torques
m3_, IG3, r3_, a3_, tL, tR = 12.0, 0.4, 0.08, 0.25, 0.5, 0.9
F1, F2 = tL / r3_, tR / r3_
fx3 = F1 + F2; nz3 = a3_ * (F2 - F1)
udot3, thdd3 = fx3 / m3_, nz3 / IG3
show('F1', F1, 3); show('F2', F2, 3); show('fx', fx3, 3); show('nz', nz3, 4)
show('udot', udot3, 4); show('thetaddot', thdd3, 4)
show('R = IG fx/(m nz)', IG3 * fx3 / (m3_ * nz3), 4)
show('thetaddot with lever 2a', 2 * a3_ * (F2 - F1) / IG3, 4)
check('W11.3 power', abs(tL * (0 / r3_) + tR * 0) == 0)
# qa-w11-4: acceleration in {C}
u4, v4, w4, ud4, vd4 = 1.5, 0.2, 0.8, 0.3, -0.1
ax4, ay4 = ud4 - v4 * w4, vd4 + u4 * w4
show('ax', ax4, 4); show('ay', ay4, 4); show('ay (slide typo v*w)', vd4 + v4 * w4, 4)
show('f_y needed (m=12)', 12 * ay4, 3)
# check by differentiating velocity in the world frame numerically
def vel_world(tm):
    uu = u4 + ud4 * tm; vv = v4 + vd4 * tm; thh = w4 * tm
    return np.array([uu * math.cos(thh) - vv * math.sin(thh), uu * math.sin(thh) + vv * math.cos(thh)])
h = 1e-6
aw = (vel_world(h) - vel_world(-h)) / (2 * h)
aC = np.array([aw[0], aw[1]])  # at t=0, theta=0, {C} aligned with {O}
check('W11.4 acceleration in {C} by finite difference', np.allclose(aC, [ax4, ay4], atol=1e-6), str(np.round(aC, 5).tolist()))
# qa-w11-5: explain
show('M33 for G 0.1 m ahead', 0.4 + 12 * 0.1**2, 4)
show('c for C=G (u=2, v=0, w=0.5, m=12): c_y', 12 * 2 * 0.5, 3)
# qa-w11-6: spot
show('nz correct', 0.2 * (7 - 3), 3); show('nz student', 2 * 0.2 * (7 - 3), 3)

# ============================ W12 ============================
print('--- W12')
# qa-w12-1: Steiner formula
Wr, Hr, rr = 3.0, 1.5, 0.4
A = Wr * Hr; Pm = 2 * (Wr + Hr)
Ag = A + Pm * rr + math.pi * rr**2
# Monte-Carlo check of Minkowski sum area
rng = np.random.default_rng(3)
N = 2_000_000
pts = rng.uniform([-rr, -rr], [Wr + rr, Hr + rr], size=(N, 2))
dx = np.maximum(np.maximum(-pts[:, 0], pts[:, 0] - Wr), 0); dy = np.maximum(np.maximum(-pts[:, 1], pts[:, 1] - Hr), 0)
inside = (dx**2 + dy**2) <= rr**2
mc = inside.mean() * (Wr + 2 * rr) * (Hr + 2 * rr)
check('W12.1 Steiner vs Monte-Carlo', abs(mc - Ag) < 0.03, f'{mc:.3f} vs {Ag:.3f}')
show('A', A, 3); show('P', Pm, 3); show('A_grown', Ag, 4)
show('square-margin area (wrong)', (Wr + 2 * rr) * (Hr + 2 * rr), 4)
show('P r', Pm * rr, 3); show('pi r^2', math.pi * rr**2, 4)
# qa-w12-2: sampling probability
rg = 0.8
pgoal = math.pi * rg**2 / 100
Nreq = math.ceil(math.log(0.05) / math.log(1 - pgoal))
show('p', pgoal, 6); show('1/p', 1 / pgoal, 2); show('N for 95%', Nreq, 0)
show('N exact', math.log(0.05) / math.log(1 - pgoal), 3)
check('W12.2 N meets 95%', 1 - (1 - pgoal)**Nreq >= 0.95 and 1 - (1 - pgoal)**(Nreq - 1) < 0.95)
show('prob at N=1/p', 1 - (1 - pgoal)**round(1 / pgoal), 4)
# qa-w12-3: A* vs Dijkstra
edges = {('S', 'A'): 4, ('S', 'B'): 2, ('B', 'A'): 1, ('A', 'C'): 5, ('B', 'D'): 7, ('C', 'G'): 3, ('D', 'G'): 3}
adj = {}
for (a, b), w in edges.items():
    adj.setdefault(a, []).append((b, w)); adj.setdefault(b, []).append((a, w))
hval = {'S': 9, 'A': 7, 'B': 8, 'C': 3, 'D': 3, 'G': 0}
def search(h):
    gcost = {'S': 0}; par = {}; openl = [(h['S'], h['S'], 'S')]; closed = []
    while openl:
        f, hh, n = heapq.heappop(openl)
        if n in closed: continue
        closed.append(n)
        if n == 'G': break
        for m_, w in sorted(adj[n]):
            if m_ in closed: continue
            ng = gcost[n] + w
            if m_ not in gcost or ng < gcost[m_]:
                gcost[m_] = ng; par[m_] = n
                heapq.heappush(openl, (ng + h[m_], h[m_], m_))
    path = ['G']
    while path[-1] != 'S': path.append(par[path[-1]])
    return closed, gcost['G'], path[::-1], gcost
cl_a, cost_a, path_a, gc_a = search(hval)
cl_d, cost_d, path_d, gc_d = search({k_: 0 for k_ in hval})
print('  A* order', cl_a, 'cost', cost_a, 'path', path_a, 'g', gc_a)
print('  Dijkstra order', cl_d, 'cost', cost_d, 'path', path_d)
# true cost-to-go for admissibility
dist = {}
for n0 in hval:
    _, c0, _, _ = (lambda s: None, 0, 0, 0) if False else (None, None, None, None)
def dij_from(src):
    dd_ = {src: 0}; pq = [(0, src)]
    while pq:
        c_, n_ = heapq.heappop(pq)
        if c_ > dd_[n_]: continue
        for m_, w in adj[n_]:
            if c_ + w < dd_.get(m_, 1e9): dd_[m_] = c_ + w; heapq.heappush(pq, (c_ + w, m_))
    return dd_
dG = dij_from('G')
print('  true cost-to-go', dG)
check('W12.3 admissible', all(hval[n] <= dG[n] for n in hval))
check('W12.3 consistent', all(hval[a] <= w + hval[b] and hval[b] <= w + hval[a] for (a, b), w in edges.items()))
show('A* cost', cost_a, 0); show('A* expansions (goal expanded, counted)', len(cl_a), 0); show('Dijkstra expansions', len(cl_d), 0)
# stopping at first discovery of G
first_g = None
gcost = {'S': 0}; openl = [(0, 'S')]; closed = []
while openl and first_g is None:
    c_, n_ = heapq.heappop(openl)
    if n_ in closed: continue
    closed.append(n_)
    for m_, w in sorted(adj[n_]):
        if m_ in closed: continue
        ng = c_ + w
        if m_ == 'G' and first_g is None: first_g = ng
        if m_ not in gcost or ng < gcost[m_]:
            gcost[m_] = ng; heapq.heappush(openl, (ng, m_))
show('cost when G first discovered', first_g, 0)
# overestimate h(B)=14 -> what happens
h_bad = dict(hval); h_bad['A'] = 30
cl_b, cost_b, path_b, _ = search(h_bad)
print('  h(A)=30 ->', cl_b, cost_b, path_b)
# qa-w12-4: grid counts
nodes5 = 50**5
show('nodes 50^5', nodes5, 0); show('2d', 10, 0); show('3^5-1', 3**5 - 1, 0)
mm_, nn_ = 40, 25
show('edges 40x25 4-conn', mm_ * (nn_ - 1) + nn_ * (mm_ - 1), 0); show('nodes 40x25', mm_ * nn_, 0)
# brute force edge count
cnt = 0
for i_, j_ in itertools.product(range(mm_), range(nn_)):
    if i_ + 1 < mm_: cnt += 1
    if j_ + 1 < nn_: cnt += 1
check('W12.4 edge count by enumeration', cnt == 1935)
dr, dc = 7, 5
show('Manhattan', dr + dc, 3); show('Euclid', math.hypot(dr, dc), 4)
oct_ = (max(dr, dc) - min(dr, dc)) + min(dr, dc) * math.sqrt(2)
show('true 8-conn cost', oct_, 4)
# 8-conn shortest path by Dijkstra on grid
W_, H_ = 12, 10
dgrid = {(0, 0): 0.0}; pq = [(0.0, (0, 0))]
while pq:
    c_, (i_, j_) = heapq.heappop(pq)
    if c_ > dgrid[(i_, j_)]: continue
    for di, dj in itertools.product((-1, 0, 1), repeat=2):
        if di == dj == 0: continue
        a_, b_ = i_ + di, j_ + dj
        if 0 <= a_ < W_ and 0 <= b_ < H_:
            nc = c_ + math.hypot(di, dj)
            if nc < dgrid.get((a_, b_), 1e9) - 1e-12: dgrid[(a_, b_)] = nc; heapq.heappush(pq, (nc, (a_, b_)))
check('W12.4 octile = Dijkstra on 8-grid', abs(dgrid[(7, 5)] - oct_) < 1e-9)
# qa-w12-5: corridor
show('free width r=0.5', 0.9 - 2 * 0.5, 3); show('free width r=0.4', 0.9 - 2 * 0.4, 3)
show('max radius', 0.9 / 2, 3)
# qa-w12-6: spot, discovery vs expansion
e6 = {('S', 'A'): 2, ('S', 'B'): 3, ('A', 'G'): 9, ('B', 'C'): 2, ('C', 'G'): 4}
adj6 = {}
for (a, b), w in e6.items():
    adj6.setdefault(a, []).append((b, w)); adj6.setdefault(b, []).append((a, w))
gc = {'S': 0}; pq = [(0, 'S')]; closed = []; first = None
while pq:
    c_, n_ = heapq.heappop(pq)
    if n_ in closed: continue
    closed.append(n_)
    if n_ == 'G': best6 = c_; break
    for m_, w in sorted(adj6[n_]):
        if m_ in closed: continue
        if m_ == 'G' and first is None: first = c_ + w
        if c_ + w < gc.get(m_, 1e9): gc[m_] = c_ + w; heapq.heappush(pq, (c_ + w, m_))
print('  order', closed)
show('first discovered G', first, 0); show('correct cost', best6, 0)
show('BFS fewest-edge path cost S-A-G', 2 + 9, 0)

# ---- extra intermediates quoted in solutions ----
print('--- extras')
show('student tau1 RP (m g d cos)', 1.5 * g0 * 0.4 * math.cos(0.9), 4)
show('tau1 RP pieces: m d^2 thdd', 1.5 * 0.16 * 0.6, 4); show('2 m d dd thd', 2 * 1.5 * 0.4 * 0.3 * 1.1, 4)
show('e^-0.5', math.exp(-0.5), 4); show('cos1', math.cos(1.0), 4); show('sin1', math.sin(1.0), 4)
show('E_ss forgetting m', 12 / 75, 4)
show('kp E pieces 49*0.2', 49 * 0.2, 2); show('kv Ed 14*0.3', 14 * 0.3, 2); show('m f\'', 1.5 * 14.8, 2)
show('wrong E_ss no cos (W9.1)', (3.0 - 2.5) * g0 / (3.0 * 0.6 * 81.0), 6)
show('W9.3 viscous', 0.4 * -0.8, 3); show('W9.3 kv ed', 20 * 0.2, 2); show('W9.3 kp e', 100 * 0.04, 2); show('W9.3 alpha tau\'', 0.588 * 9.5, 4)
show('W10.2 sin th1', math.sin(98.755 * d2r), 4); show('W10.2 cos th1', math.cos(98.755 * d2r), 4)
show('W10.2 cos th0', math.cos(30 * d2r), 4)
show('W10.2 w*T rad', 0.8 * 1.5, 3)
show('W10.4 R-t/2', 8 - 0.75, 3); show('W10.4 R+t/2', 8 + 0.75, 3)
show('W11.1 u dot via row1', 0.75 + 0.02 * thdd_, 4); show('W11.1 effective coeff', 0.3232 - 0.0032 - 0.02, 4); show('W11.1 theta dd', 0.62 / 0.3, 4)
show('W12.2 ln 0.05', math.log(0.05), 4); show('W12.2 ln(1-p)', math.log(1 - pgoal), 5)
show('W12.1 4.5+3.6', 4.5 + 3.6, 3)
show('W12.1 sum', 4.5 + 3.6 + 0.5027, 4)
show('W7.5 Lag terms 32*625', 32 * 625, 0); show('86*125', 86 * 125, 0); show('171*25', 171 * 25, 0); show('53*5', 53 * 5, 0)
report()
