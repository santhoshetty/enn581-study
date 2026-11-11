"""W9 Control part 2 — verifies every number on the W9 tab and the `ctc` widget test vectors.
Run: python3 site/src/checks/w09.py   (prints PASS/FAIL, writes w09_vectors.json, ends with ALL CHECKS PASSED)

Conventions (same as the tab): single link, q measured from the horizontal, gravity torque m g L cos q, point mass m at the tip;
E = desired - actual; gains k_p [1/s^2], k_v [1/s] act on acceleration-level error (alpha-scaled servo law).
"""
import json, os, re, shutil, subprocess, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
G = 9.81
out = {}

# =====================================================================================
# 1. Cubic-spring SISO example (slides pp. 9-11)
# =====================================================================================
m, b, k = 2.0, 4.0, 50.0
x, xd_, = 0.10, 0.30                    # actual x [m], xdot [m/s]
xD, xDd, xDdd = 0.12, 0.40, 0.50        # desired x, xdot, xddot
kp, kv = 100.0, 20.0
E, Ed = xD - x, xDd - xd_
fp = xDdd + kv*Ed + kp*E                # f' (commanded acceleration)
beta = b*xd_ + k*x**3
f = m*fp + beta
xdd = (f - b*xd_ - k*x**3)/m            # plant
close('cubic spring: E, Edot', [E, Ed], [0.02, 0.10])
close('cubic spring: f prime = 4.5', fp, 4.5)
close('cubic spring: beta = 1.25', beta, 1.25)
close('cubic spring: f = 10.25', f, 10.25)
close('cubic spring: plant xddot = 4.5', xdd, 4.5)
close('cubic spring: eddot = -4', xDdd - xdd, -4.0)
close('cubic spring: eddot + kv Edot + kp E = 0', (xDdd - xdd) + kv*Ed + kp*E, 0.0)
out['cubic'] = dict(fprime=fp, beta=beta, f=f, xdd=xdd)

# =====================================================================================
# 2. Single link: Lagrange derivation and worked partitioned law (slides pp. 12-14)
# =====================================================================================
q_, qd_, qdd_ = sp.symbols('q qd qdd')
m_s, L_s, g_s, bv_s, bc_s = sp.symbols('m L g b_v b_c', positive=True)
t = sp.symbols('t')
qt = sp.Function('q')(t)
xx, yy = L_s*sp.cos(qt), L_s*sp.sin(qt)
K_ = sp.Rational(1, 2)*m_s*(sp.diff(xx, t)**2 + sp.diff(yy, t)**2)
P_ = m_s*g_s*yy
Lag = K_ - P_
Qeq = sp.diff(sp.diff(Lag, sp.diff(qt, t)), t) - sp.diff(Lag, qt)
target = m_s*L_s**2*sp.diff(qt, t, 2) + m_s*g_s*L_s*sp.cos(qt)
check('single link Lagrange: Q = m L^2 qddot + m g L cos q', sp.simplify(Qeq - target) == 0)

# worked numbers: m=2 kg, L=0.5 m, bv=0.3, bc=0.2, q=30 deg, qdot=1, qdd_d=2, e=0.05, edot=-0.1, kp=64, kv=16
m1w, L1w, bvw, bcw = 2.0, 0.5, 0.3, 0.2
qw, qdw = 30*deg, 1.0
qddD, ew, edw, kpw, kvw = 2.0, 0.05, -0.1, 64.0, 16.0
alpha_w = m1w*L1w**2
taup = qddD + kvw*edw + kpw*ew
beta_w = bvw*qdw + bcw*np.sign(qdw) + m1w*G*L1w*np.cos(qw)
tau_w = alpha_w*taup + beta_w
close('single link worked: alpha = 0.5', alpha_w, 0.5)
close('single link worked: tau prime = 3.6', taup, 3.6)
close('single link worked: beta = 8.9957', beta_w, 8.9957, tol=5e-5)
close('single link worked: tau = 10.7957', tau_w, 10.7957, tol=5e-5)
qdd_plant = (tau_w - beta_w)/alpha_w
close('single link worked: plant qddot = tau prime', qdd_plant, taup)
out['link_worked'] = dict(alpha=alpha_w, taup=taup, beta=r(beta_w), tau=r(tau_w))

# =====================================================================================
# 3. Closed-loop algebra: perfect and imprecise model (slides pp. 21, 28), MIMO, random numeric check
# =====================================================================================
rng = np.random.default_rng(9)
def two_link(L1, L2, m1, m2):
    # point masses at the distal ends; returns M(q), C(q,qd), g(q) functions
    def M(q):
        c2 = np.cos(q[1])
        a = m2*L2**2 + m1*0
        M11 = (m1+m2)*L1**2 + m2*L2**2 + 2*m2*L1*L2*c2
        M12 = m2*L2**2 + m2*L1*L2*c2
        return np.array([[M11, M12], [M12, m2*L2**2]])
    def C(q, v):
        s2 = np.sin(q[1]); h = m2*L1*L2*s2
        return np.array([-h*(2*v[0]*v[1] + v[1]**2), h*v[0]**2])
    def g(q):
        return np.array([(m1+m2)*G*L1*np.cos(q[0]) + m2*G*L2*np.cos(q[0]+q[1]), m2*G*L2*np.cos(q[0]+q[1])])
    return M, C, g

Mt, Ct, gt = two_link(1.0, 0.8, 2.0, 1.5)       # true
Mh, Ch, gh = two_link(1.0, 0.8, 1.7, 1.2)       # imprecise (hat) parameters
Kp = np.diag([80., 120.]); Kv = np.diag([18., 22.])
ok = True; ok2 = True
for _ in range(20):
    q = rng.uniform(-2, 2, 2); v = rng.uniform(-2, 2, 2)
    qD, vD, aD = rng.uniform(-2, 2, 2), rng.uniform(-2, 2, 2), rng.uniform(-2, 2, 2)
    e, ed = qD - q, vD - v
    Qp = aD + Kv@ed + Kp@e
    # exact model
    Q = Mt(q)@Qp + Ct(q, v) + gt(q)
    qdd = np.linalg.solve(Mt(q), Q - Ct(q, v) - gt(q))
    edd = aD - qdd
    ok &= np.allclose(edd + Kv@ed + Kp@e, 0, atol=1e-9)
    # imprecise model: slide p. 28
    Q = Mh(q)@Qp + Ch(q, v) + gh(q)
    qdd = np.linalg.solve(Mt(q), Q - Ct(q, v) - gt(q))
    edd = aD - qdd
    lhs = edd + Kv@ed + Kp@e
    rhs = np.linalg.solve(Mh(q), (Mt(q)-Mh(q))@qdd + (Ct(q, v)-Ch(q, v)) + (gt(q)-gh(q)))
    ok2 &= np.allclose(lhs, rhs, atol=1e-9)
check('MIMO exact model: Edd + Kv Ed + Kp E = 0 (20 random states)', ok)
check('MIMO imprecise model: p. 28 error equation holds (20 random states)', ok2)

# scalar single link, m_hat = k m: forcing = ((m - m_hat)/m_hat) (qdd + (g/L) cos q)
mT, LT = 3.0, 3.0
ok3 = True
for _ in range(20):
    q = rng.uniform(-1.5, 1.5); v = rng.uniform(-2, 2); qD, vD, aD = rng.uniform(-1.5, 1.5), rng.uniform(-2, 2), rng.uniform(-2, 2)
    mh = rng.uniform(1, 5)
    e, ed = qD-q, vD-v
    tau = mh*LT**2*(aD + 16*ed + 64*e) + mh*G*LT*np.cos(q)
    qdd = (tau - mT*G*LT*np.cos(q))/(mT*LT**2)
    lhs = (aD-qdd) + 16*ed + 64*e
    rhs = (mT-mh)/mh*(qdd + G/LT*np.cos(q))
    ok3 &= abs(lhs-rhs) < 1e-9
check('scalar link, m_hat = k m: forcing = ((m-m_hat)/m_hat)(qddot + (g/L) cos q)', ok3)

# =====================================================================================
# 4. The 3-link practical sheet
# =====================================================================================
L1, L2, L3 = 3.0, 2.0, 1.0
mm = (3.0, 2.0, 1.0)
def fk3(q):
    a, b_, c = q
    x = L1*np.cos(a) + L2*np.cos(a+b_) + L3*np.cos(a+b_+c)
    y = L1*np.sin(a) + L2*np.sin(a+b_) + L3*np.sin(a+b_+c)
    return np.array([x, y, a+b_+c])
def ik3(x, y, phi, elbow=+1):
    wx, wy = x - L3*np.cos(phi), y - L3*np.sin(phi)
    c2 = (wx**2 + wy**2 - L1**2 - L2**2)/(2*L1*L2)
    th2 = elbow*np.arccos(np.clip(c2, -1, 1))
    th1 = np.arctan2(wy, wx) - np.arctan2(L2*np.sin(th2), L1 + L2*np.cos(th2))
    return np.array([th1, th2, phi - th1 - th2]), c2, (wx, wy)
qs, c2s, ws = ik3(5, 1, 90*deg)
qg, c2g, wg = ik3(-3, 3, 180*deg, +1)
qg2, _, _ = ik3(-3, 3, 180*deg, -1)
close('start wrist = (5, 0), cos q2 = 1 (fully stretched)', [*ws, c2s], [5, 0, 1])
close('start joints = (0, 0, 90 deg)', qs/deg, [0, 0, 90], tol=1e-6)
close('goal wrist = (-2, 3), cos q2 = 0', [*wg, c2g], [-2, 3, 0], tol=1e-9)
close('goal joints (elbow +) = (90, 90, 0) deg', qg/deg, [90, 90, 0], tol=1e-6)
close('goal joints (elbow -) = (157.38, -90, 112.62) deg', qg2/deg, [157.3801, -90, 112.6199], tol=1e-3)
close('FK start', fk3(qs)[:2], [5, 1], tol=1e-9)
close('FK goal (elbow +)', fk3(qg)[:2], [-3, 3], tol=1e-9)
close('FK goal (elbow -)', fk3(qg2)[:2], [-3, 3], tol=1e-9)
out['ik'] = dict(q_start_deg=(qs/deg).round(4).tolist(), q_goal_deg=(qg/deg).round(4).tolist(), q_goal_alt_deg=(qg2/deg).round(4).tolist())

# cubic, rest to rest, assumed T = 2 s
T = 2.0
dq = qg - qs
a2 = 3*dq/T**2; a3 = -2*dq/T**3
close('cubic a2 (rad/s^2), T=2', a2, [1.1781, 1.1781, -1.1781], tol=5e-5)
close('cubic a3 (rad/s^3), T=2', a3, [-0.3927, -0.3927, 0.3927], tol=5e-5)
vpk = 1.5*dq/T; apk = 6*dq/T**2
close('peak speed = 1.5 dq/T at T/2', vpk, [1.1781, 1.1781, -1.1781], tol=5e-5)
close('peak accel = 6 dq/T^2 at t=0', apk, [2.3562, 2.3562, -2.3562], tol=5e-5)
def cubic(tt):
    s = tt/T
    return qs + dq*(3*s**2 - 2*s**3), dq*(6*s - 6*s**2)/T, dq*(6 - 12*s)/T**2
q_mid = cubic(1.0)[0]
close('cubic mid-time value is the midpoint', q_mid, (qs+qg)/2)
# peak speed numerically
ts = np.linspace(0, T, 2001)
close('numeric peak speed joint 1', max(abs(cubic(tt)[1][0]) for tt in ts), vpk[0], tol=1e-6)
# the cubic joint path is not a straight Cartesian line: midpoint pose
mid_pose = fk3(q_mid)
print('   mid-time end-effector pose (x, y, phi deg):', np.round([mid_pose[0], mid_pose[1], mid_pose[2]/deg], 4))
out['mid_pose'] = [r(mid_pose[0]), r(mid_pose[1]), r(mid_pose[2]/deg)]
# the straight line mid between start and goal would be (1, 2): check the path does not pass it
check('joint-space path is not the Cartesian straight line (mid-time tip != (1,2))', np.hypot(mid_pose[0]-1, mid_pose[1]-2) > 0.5)

# --- Lagrangian model with sympy ---
q1s, q2s, q3s, d1, d2, d3 = sp.symbols('q1 q2 q3 d1 d2 d3')
qS = [q1s, q2s, q3s]; dS = [d1, d2, d3]
x1 = L1*sp.cos(q1s); y1 = L1*sp.sin(q1s)
x2 = x1 + L2*sp.cos(q1s+q2s); y2 = y1 + L2*sp.sin(q1s+q2s)
x3 = x2 + L3*sp.cos(q1s+q2s+q3s); y3 = y2 + L3*sp.sin(q1s+q2s+q3s)
def vel(e): return sum(sp.diff(e, qS[i])*dS[i] for i in range(3))
Ksym = sum(mi*(vel(xx_)**2 + vel(yy_)**2)/2 for mi, xx_, yy_ in zip(mm, (x1, x2, x3), (y1, y2, y3)))
gS = sp.Float(G)
Psym = gS*(mm[0]*y1 + mm[1]*y2 + mm[2]*y3)
Msym = sp.Matrix(3, 3, lambda i, j: sp.diff(Ksym, dS[i], dS[j]))
Csym = sp.Matrix([sum((sp.diff(Msym[i, j], qS[kk]) - sp.Rational(1, 2)*sp.diff(Msym[j, kk], qS[i]))*dS[j]*dS[kk] for j in range(3) for kk in range(3)) for i in range(3)])
gsym = sp.Matrix([sp.diff(Psym, qq) for qq in qS])
Mf = sp.lambdify(qS, Msym, 'numpy'); Cf = sp.lambdify(qS + dS, Csym, 'numpy'); gf = sp.lambdify(qS, gsym, 'numpy')
def M3(q): return np.array(Mf(*q), dtype=float)
def C3(q, v): return np.array(Cf(*q, *v), dtype=float).reshape(3)
def g3(q): return np.array(gf(*q), dtype=float).reshape(3)

# closed forms printed on the tab
def M3_closed(q):
    c2, c3, c23 = np.cos(q[1]), np.cos(q[2]), np.cos(q[1]+q[2])
    return np.array([[67 + 36*c2 + 6*c23 + 4*c3, 13 + 18*c2 + 3*c23 + 4*c3, 1 + 3*c23 + 2*c3],
                     [13 + 18*c2 + 3*c23 + 4*c3, 13 + 4*c3, 1 + 2*c3],
                     [1 + 3*c23 + 2*c3, 1 + 2*c3, 1.0]])
def C3_closed(q, v):
    A_, B_, D_ = 3*3*2, 1*3*1, 1*2*1      # (m2+m3)L1L2, m3 L1 L3, m3 L2 L3 -> 3*3*2? see below
    A_ = (mm[1]+mm[2])*L1*L2; B_ = mm[2]*L1*L3; D_ = mm[2]*L2*L3
    s2, s23, s3 = np.sin(q[1]), np.sin(q[1]+q[2]), np.sin(q[2])
    v12 = v[0]+v[1]
    return np.array([-A_*s2*(2*v[0]*v[1] + v[1]**2) - B_*s23*(2*v[0]*(v[1]+v[2]) + (v[1]+v[2])**2) - D_*s3*(2*v12*v[2] + v[2]**2),
                     A_*s2*v[0]**2 + B_*s23*v[0]**2 - D_*s3*(2*v12*v[2] + v[2]**2),
                     B_*s23*v[0]**2 + D_*s3*v12**2])
def g3_closed(q):
    c1, c12, c123 = np.cos(q[0]), np.cos(q[0]+q[1]), np.cos(q[0]+q[1]+q[2])
    return G*np.array([6*3*c1 + 3*2*c12 + 1*1*c123, 3*2*c12 + 1*1*c123, 1*1*c123])
okM = okC = okg = True
for _ in range(20):
    q = rng.uniform(-3, 3, 3); v = rng.uniform(-3, 3, 3)
    okM &= np.allclose(M3(q), M3_closed(q)); okC &= np.allclose(C3(q, v), C3_closed(q, v)); okg &= np.allclose(g3(q), g3_closed(q))
check('3-link M(q): sympy Lagrange == closed form printed on the tab', okM)
check('3-link C(q,qdot): Christoffel == closed form printed on the tab', okC)
check('3-link g(q): dP/dq == closed form printed on the tab', okg)
# M symmetric positive definite, C consistent with Mdot - 2C skew property (Christoffel form)
q = rng.uniform(-3, 3, 3)
check('M symmetric positive definite', np.allclose(M3(q), M3(q).T) and np.all(np.linalg.eigvalsh(M3(q)) > 0))

# (c) initial configuration, from rest
M0 = M3(qs); C0 = C3(qs, np.zeros(3)); g0 = g3(qs)
close('M(q0) = [[103,31,1],[31,13,1],[1,1,1]]', M0, [[103, 31, 1], [31, 13, 1], [1, 1, 1]], tol=1e-9)
close('C(q0, 0) = 0', C0, [0, 0, 0], tol=1e-12)
close('g(q0) = (235.44, 58.86, 0) N m', g0, [235.44, 58.86, 0.0], tol=1e-9)
close('g1 = 9.81*(18+6+0)', g0[0], 9.81*24)
out['init'] = dict(M=M0.tolist(), g=g0.tolist())
# values at the goal for the tab (elbow +, q = (90,90,0) deg)
Mg_ = M3(qg); gg_ = g3(qg)
print('   M(q_goal) =', np.round(Mg_, 4).tolist(), ' g(q_goal) =', np.round(gg_, 4).tolist())
out['goal'] = dict(M=np.round(Mg_, 6).tolist(), g=np.round(gg_, 6).tolist())

# inverse-dynamics feed-forward at t = 0 (cubic, T = 2 s), for the 'why computed torque needs a good M' remark
Qff0 = M0@apk + g0
print('   inverse dynamics Q(t=0) =', np.round(Qff0, 3).tolist())
out['Qff0'] = Qff0.round(4).tolist()

# (e) partitioned law for link 1
# reading A: isolated link 1 (m1 = 3 kg, L1 = 3 m, no friction)
alphaA = mm[0]*L1**2; betaA0 = mm[0]*G*L1*np.cos(qs[0])
close('reading A: alpha = m1 L1^2 = 27', alphaA, 27.0)
close('reading A: beta(q1=0) = 88.29', betaA0, 88.29)
tauA0 = alphaA*apk[0] + betaA0
close('reading A: tau(0) = 27*2.3562 + 88.29 = 151.907', tauA0, 151.907, tol=5e-4)
close('reading A: alpha*qdd_d(0) = 63.617', alphaA*apk[0], 63.617, tol=5e-4)
qdd_back = (tauA0 - betaA0)/alphaA
close('reading A: simulator returns qddot = qdd_d(0)', qdd_back, apk[0])
# reading B: joint 1 of the full arm (other joints held), alpha = M11, beta = g1
close('reading B: alpha = M11 = 103, beta = g1 = 235.44', [M0[0, 0], g0[0]], [103, 235.44])
# imprecise m1_hat = 2.8 in reading B
Mh11 = M0[0, 0] - 0.2*L1**2
gh1 = g0[0] - 0.2*G*L1
close('reading B: M11_hat = 101.2', Mh11, 101.2)
close('reading B: g1_hat = 229.554', gh1, 229.554)
close('reading B: static forcing (g1-g1_hat)/M11_hat = 0.05816', (g0[0]-gh1)/Mh11, 0.058162, tol=1e-6)
out['readingB'] = dict(Mh11=Mh11, gh1=gh1, forcing_static=(g0[0]-gh1)/Mh11)

# (f) forward dynamics simulator (reading A), RK4 with dt, ZOH of tau at controller period Ts
def sim_link1(mhat, kp, kv, Ts, dt=1e-4, T_end=4.0, hold=None, e0=0.0):
    """Link 1 alone (m=3 kg, L=3 m, no friction) tracking the cubic. Ts = dt means the law is evaluated inside every RK4 stage
    (continuous control); Ts > dt means zero-order hold of the torque for Ts."""
    mt, Lt = mm[0], L1
    N = int(round(Ts/dt)); steps = int(round(T_end/dt))
    def ref(tt):
        if hold is not None: return hold, 0.0, 0.0
        if tt >= T: return cubic(T)[0][0], 0.0, 0.0
        c = cubic(tt); return c[0][0], c[1][0], c[2][0]
    q = ref(0)[0] - e0; v = 0.0
    def law(tt, q, v):
        qD, vD, aD = ref(tt)
        return mhat*Lt**2*(aD + kv*(vD - v) + kp*(qD - q)) + mhat*G*Lt*np.cos(q)
    def acc(q, tau): return (tau - mt*G*Lt*np.cos(q))/(mt*Lt**2)
    ts_, es = [], []; tau = 0.0
    for n in range(steps+1):
        tt = n*dt
        ts_.append(tt); es.append(ref(tt)[0] - q)
        if n == steps: break
        if N == 1:
            k1q, k1v = v, acc(q, law(tt, q, v))
            k2q, k2v = v + dt/2*k1v, acc(q + dt/2*k1q, law(tt + dt/2, q + dt/2*k1q, v + dt/2*k1v))
            k3q, k3v = v + dt/2*k2v, acc(q + dt/2*k2q, law(tt + dt/2, q + dt/2*k2q, v + dt/2*k2v))
            k4q, k4v = v + dt*k3v, acc(q + dt*k3q, law(tt + dt, q + dt*k3q, v + dt*k3v))
        else:
            if n % N == 0: tau = law(tt, q, v)
            k1q, k1v = v, acc(q, tau)
            k2q, k2v = v + dt/2*k1v, acc(q + dt/2*k1q, tau)
            k3q, k3v = v + dt/2*k2v, acc(q + dt/2*k2q, tau)
            k4q, k4v = v + dt*k3v, acc(q + dt*k3q, tau)
        q += dt/6*(k1q + 2*k2q + 2*k3q + k4q); v += dt/6*(k1v + 2*k2v + 2*k3v + k4v)
    return np.array(ts_), np.array(es)

# (g) exact model: error stays at (numerically) zero; discretisation leaves a small residual
res = {}
for Ts in (1e-4, 1e-2, 5e-2):
    tt, ee = sim_link1(3.0, 100.0, 20.0, Ts)
    res[Ts] = float(np.max(np.abs(ee)))
    print(f'   exact model, k_p=100, k_v=20, Ts={Ts*1000:g} ms: max|e| = {res[Ts]:.3e} rad')
tt_, ee_ = sim_link1(3.0, 100.0, 20.0, 1e-4)
pre = float(np.max(np.abs(ee_[tt_ < 1.99])))
check('exact model, continuous law, t < 1.99 s: max |e| < 1e-10 rad', pre < 1e-10, f'{pre:.2e}')
check('exact model, continuous law, whole run: max |e| < 5e-6 rad (only the step in qdd_d at t = T disturbs the RK4)', res[1e-4] < 5e-6, f'{res[1e-4]:.2e}')
check('slower servo rate gives a larger error (10 ms < 50 ms)', res[1e-4] < res[1e-2] < res[5e-2])
out['disc'] = {f'{int(round(k*1000))}ms': float(f'{v:.4g}') for k, v in res.items()}
# exact model, initial error 0.1 rad: critically damped decay (kp=100, kv=20, w=10)
tt, ee = sim_link1(3.0, 100.0, 20.0, 1e-4, e0=0.1, T_end=1.0)
w = 10.0
for tq in (0.1, 0.25, 0.5):
    i = int(round(tq/1e-4)); ana = 0.1*(1 + w*tq)*np.exp(-w*tq)
    close(f'exact model, e0=0.1: e({tq}) = e0 (1+w t) e^(-w t)', ee[i], ana, tol=1e-7)
# (h) imprecise mass 2.8 kg
tt, ee = sim_link1(2.8, 100.0, 20.0, 1e-4)
imax = int(np.argmax(np.abs(ee)))
print(f'   m_hat = 2.8: max|e| = {abs(ee[imax]):.5f} rad ({abs(ee[imax])/deg:.4f} deg) at t = {tt[imax]:.3f} s; e(T=2) = {ee[int(2/1e-4)]:.6f}; e(4 s) = {ee[-1]:.3e}')
out['h'] = dict(max_e=float(abs(ee[imax])), t_max=float(tt[imax]), e_T=float(ee[int(2/1e-4)]))
# predicted initial forcing
close('forcing ratio (m - m_hat)/m_hat = 1/14', (3.0-2.8)/2.8, 1/14)
close('forcing at t=0: (1/14)(2.3562 + 3.27 cos 0) = 0.4019', (1/14)*(apk[0] + G/L1), 0.4019, tol=5e-5)
close('e_ss at q=0 hold, k_p=100: (1/14)(g/L)/k_p = 2.336e-3 rad', (1/14)*(G/L1)/100, 2.3357e-3, tol=1e-7)
# hold at q_d = 0 : simulate and compare to the fixed point
tt, ee = sim_link1(2.8, 100.0, 20.0, 1e-4, hold=0.0, T_end=3.0, e0=0.0)
e_inf = ee[-1]
# exact static solution: kp e = ((m-mh)/mh)(g/L) cos(qd - e)
ee_fix = 0.0
for _ in range(60): ee_fix = (1/14)*(G/L1)*np.cos(0.0 - ee_fix)/100
close('hold sim reaches static error predicted by p. 28 equation', e_inf, ee_fix, tol=1e-7)
print(f'   e_ss(hold q=0) = {e_inf:.6e} rad = {e_inf/deg:.4f} deg')
out['ess_hold'] = float(e_inf)
# at q_d = 90 deg the gravity term vanishes
ee_fix90 = 0.0
for _ in range(60): ee_fix90 = (1/14)*(G/L1)*np.cos(np.pi/2 - ee_fix90)/100
print(f'   e_ss(hold q=90 deg) = {ee_fix90:.3e} rad (cos(q) ~ 0, only a tiny sag)')
check('hold at 90 deg: static error is under 3e-5 rad (gravity torque has no lever arm)', ee_fix90 < 3e-5)

# (i) extension: coupling forcing if alpha were diag(M) only; and why alpha is not diagonal
coupling1 = (M0[0, 1]*a2[1]*0 + M0[0, 1]*apk[1] + M0[0, 2]*apk[2])/M0[0, 0]
close('ignoring off-diagonals of M: forcing on joint 1 at t=0 = (31*2.3562 - 2.3562)/103 = 30*2.3562/103 = 0.6863', coupling1, 0.68627, tol=5e-5)
out['coupling'] = float(coupling1)
# full law check on the 3-link arm: exact model gives decoupled error equation
Kp3 = np.diag([100., 100., 100.]); Kv3 = np.diag([20., 20., 20.])
ok = True
for _ in range(20):
    q = rng.uniform(-3, 3, 3); v = rng.uniform(-3, 3, 3); qD, vD, aD = rng.uniform(-3, 3, 3), rng.uniform(-3, 3, 3), rng.uniform(-3, 3, 3)
    Qc = M3(q)@(aD + Kv3@(vD-v) + Kp3@(qD-q)) + C3(q, v) + g3(q)
    qdd = np.linalg.solve(M3(q), Qc - C3(q, v) - g3(q))
    ok &= np.allclose((aD-qdd) + Kv3@(vD-v) + Kp3@(qD-q), 0, atol=1e-8)
check('3-link computed torque with exact model: decoupled error equation (20 random states)', ok)

# =====================================================================================
# 5. Cartesian control (pp. 29-33)
# =====================================================================================
l1, l2 = 1.0, 1.0
def Jv(q):
    s1, c1, s12, c12 = np.sin(q[0]), np.cos(q[0]), np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
    return np.array([[-l1*s1 - l2*s12, -l2*s12], [l1*c1 + l2*c12, l2*c12]])
q = np.array([0.0, 90*deg])
J = Jv(q)
close('2-link J at (0, 90 deg)', J, [[-1, -1], [1, 0]], tol=1e-12)
F = np.array([0.0, 10.0])
Qj = J.T@F
close('Q = J^T F, F = (0, 10) N -> (10, 0) N m', Qj, [10, 0], tol=1e-12)
tip = np.array([l1*np.cos(q[0]) + l2*np.cos(q[0]+q[1]), l1*np.sin(q[0]) + l2*np.sin(q[0]+q[1])])
joint2 = np.array([l1*np.cos(q[0]), l1*np.sin(q[0])])
close('torque about joint 1 = r x F (statics check)', (tip[0]*F[1] - tip[1]*F[0]), 10.0)
close('torque about joint 2 = r x F', ((tip-joint2)[0]*F[1] - (tip-joint2)[1]*F[0]), 0.0)
# second example, 45 deg posture, tip force (3, -4) N
q = np.array([30*deg, 60*deg]); J = Jv(q); F = np.array([3.0, -4.0]); Qj = J.T@F
tip = np.array([np.cos(q[0]) + np.cos(q[0]+q[1]), np.sin(q[0]) + np.sin(q[0]+q[1])])
j2 = np.array([np.cos(q[0]), np.sin(q[0])])
close('statics cross-check Q1 for F=(3,-4) at q=(30,60) deg', Qj[0], tip[0]*F[1]-tip[1]*F[0])
close('statics cross-check Q2', Qj[1], (tip-j2)[0]*F[1]-(tip-j2)[1]*F[0])
print('   Q = J^T F at q=(30,60) deg, F=(3,-4) N:', np.round(Qj, 4).tolist())
out['JtF'] = dict(J=np.round(J, 4).tolist(), Q=np.round(Qj, 4).tolist(), tip=np.round(tip, 4).tolist())

# determinant of J for 2-link: l1 l2 sin(q2) -> singular at q2 = 0 or pi (3-link start pose is stretched, q2=0)
close('det J = l1 l2 sin q2', np.linalg.det(Jv(np.array([0.3, 0.7]))), l1*l2*np.sin(0.7))

# d/dt (J^-1) = - J^-1 Jdot J^-1  (reading of the slide's "Jdot^-1" on p. 31)
qq = np.array([0.4, 1.1]); qd_v = np.array([0.7, -0.5]); h = 1e-6
Jn = Jv(qq); Jdot = (Jv(qq + h*qd_v) - Jv(qq - h*qd_v))/(2*h)
dJinv = (np.linalg.inv(Jv(qq + h*qd_v)) - np.linalg.inv(Jv(qq - h*qd_v)))/(2*h)
close('d/dt(J^-1) = -J^-1 Jdot J^-1 (finite difference)', dJinv, -np.linalg.inv(Jn)@Jdot@np.linalg.inv(Jn), tol=1e-6, show=False)
# qddot = J^-1 (xddot - Jdot qdot) = d/dt(J^-1) xdot + J^-1 xddot
xdot = Jn@qd_v; xddot = np.array([0.3, -0.2])
close('qddot = J^-1 xddot + d/dt(J^-1) xdot  == J^-1 (xddot - Jdot qdot)',
      np.linalg.inv(Jn)@xddot + (-np.linalg.inv(Jn)@Jdot@np.linalg.inv(Jn))@xdot,
      np.linalg.inv(Jn)@(xddot - Jdot@qd_v), tol=1e-6, show=False)

# Cartesian dynamics: M_x = J^-T M J^-1, c_x = J^-T (C - M J^-1 Jdot qdot), g_x = J^-T g ; F = M_x xddot + c_x + g_x -> Q = J^T F equals joint dynamics
Mt2, Ct2, gt2 = two_link(1.0, 1.0, 2.0, 1.5)
Ji = np.linalg.inv(Jn)
Mx = Ji.T@Mt2(qq)@Ji
cx = Ji.T@(Ct2(qq, qd_v) - Mt2(qq)@Ji@Jdot@qd_v)
gx = Ji.T@gt2(qq)
qddot_ = Ji@(xddot - Jdot@qd_v)
Qjoint = Mt2(qq)@qddot_ + Ct2(qq, qd_v) + gt2(qq)
close('Q = J^T (M_x xddot + c_x + g_x) equals M qddot + C + g', Jn.T@(Mx@xddot + cx + gx), Qjoint, tol=1e-5, show=False)

# =====================================================================================
# 6. Force control (pp. 34-44)
# =====================================================================================
fe_, fd_, fdist, mm_, ke, kp_, kv_ = sp.symbols('f_e f_d f_dist m k_e k_p k_v', positive=True)
al = mm_/ke
# Option 1: f = al (fdd_d + kv Ed + kp E) + fe ; plant f = al fe'' + fdist + fe   (E = fd - fe)
# -> al (E'' + kv E' + kp E) = fdist   [with fd'' etc: al(fd'' - fe'') + ... ]
# derive: al*(fdd + kv*Ed + kp*E) + fe = al*fedd + fdist + fe  ->  al*(fdd - fedd) + al kv Ed + al kp E = fdist
E_, Ed_, Edd_ = sp.symbols('E Ed Edd')
opt1 = sp.Eq(al*(Edd_ + kv_*Ed_ + kp_*E_), fdist)
E_ss1 = sp.solve(opt1.subs({Edd_: 0, Ed_: 0}), E_)[0]
check('Option 1 steady state = f_dist/(m k_e^-1 k_p)', sp.simplify(E_ss1 - fdist/(al*kp_)) == 0)
# Option 2: f = al (...) + fd ; -> al(E'' + kv E' + kp E) = fdist + fe - fd = fdist - E
opt2 = sp.Eq(al*(Edd_ + kv_*Ed_ + kp_*E_), fdist - E_)
E_ss2 = sp.solve(opt2.subs({Edd_: 0, Ed_: 0}), E_)[0]
check('Option 2 steady state = f_dist/(1 + m k_e^-1 k_p)', sp.simplify(E_ss2 - fdist/(1 + al*kp_)) == 0)
# Option 2 closed loop as printed: E'' + kv E' + (1/al + kp) E = fdist/al
lhs2 = sp.expand(sp.solve(opt2, Edd_)[0]*0 + (Edd_ + kv_*Ed_ + (1/al + kp_)*E_ - fdist/al))
chk2 = sp.simplify(sp.expand((opt2.lhs - opt2.rhs)/al) - lhs2)
check('Option 2 closed loop = slide p. 39 form', chk2 == 0)
# numerical example
mv, kev, kpv, fdv = 2.0, 1000.0, 100.0, 5.0
a_ = mv/kev
e1 = fdv/(a_*kpv); e2 = fdv/(1 + a_*kpv)
close('example: m k_e^-1 = 0.002, m k_e^-1 k_p = 0.2', [a_, a_*kpv], [0.002, 0.2])
close('example Option 1: E = 25 N', e1, 25.0)
close('example Option 2: E = 4.1667 N', e2, 4.1667, tol=5e-5)
close('ratio E2/E1 = x/(1+x), x = m k_e^-1 k_p = 0.1667', e2/e1, 0.2/1.2, tol=1e-12)
# soft environment, same gain
kev2 = 20.0; a2_ = mv/kev2
e1b, e2b = fdv/(a2_*kpv), fdv/(1 + a2_*kpv)
print(f'   soft (k_e = 20 N/m): m k_e^-1 k_p = {a2_*kpv:g}; E1 = {e1b:.4f}, E2 = {e2b:.4f} N')
close('soft env: x = 10, E1 = 0.5 N', e1b, 0.5); close('soft env: E2 = 5/11 = 0.4545 N', e2b, 5/11, tol=1e-9)
# large gain needed: k_p for 1% of f_dist in Option 2 (x = 99) -> k_p = 99/(m/ke)
kp99 = 99/a_
close('Option 2: 1% of f_dist needs k_p = 99 / (m k_e^-1) = 49500 s^-2', kp99, 49500.0)
out['force'] = dict(e1=e1, e2=e2, e1_soft=e1b, e2_soft=e2b, kp99=kp99)
# simulation check of Option 1 and Option 2 steady states (linear plant: m xdd + fdist + ke x = f)
def force_sim(option, mv=2.0, kev=1000.0, kpv=100.0, kvv=20.0, fdv=10.0, fdist_v=5.0, dt=1e-4, T_end=6.0):
    x_, v_ = fdv/kev, 0.0           # start in equilibrium for the desired force with no disturbance
    for n in range(int(T_end/dt)):
        fe = kev*x_; E = fdv - fe; Ed = 0 - kev*v_
        base = (mv/kev)*(0 + kvv*Ed + kpv*E)
        f = base + (fe if option == 1 else fdv)
        acc = (f - fdist_v - kev*x_)/mv
        v_ += acc*dt; x_ += v_*dt      # semi-implicit Euler, tiny dt
    return fdv - kev*x_
e1s, e2s = force_sim(1), force_sim(2)
close('force-control sim, Option 1: steady E = 25 N', e1s, 25.0, tol=0.05)
close('force-control sim, Option 2: steady E = 4.1667 N', e2s, 4.1667, tol=5e-3)
# practical law p. 43: f = m [ke^-1 kp E - kv xdot] + fd == Option 2 with fd_dot = 0, Ed = -ke xdot
kvv = 20.0; x_ = 0.004; v_ = 0.02; fe = kev*x_; Ev = 10.0 - fe
lawA = (mv/kev)*(0 + kvv*(0 - kev*v_) + kpv*Ev) + 10.0
lawB = mv*((1/kev)*kpv*Ev - kvv*v_) + 10.0
close('p. 43 law == Option 2 law when f_d is constant (E-dot = -k_e xdot)', lawA, lawB)

# =====================================================================================
# 7. Hybrid control example (p. 47)
# =====================================================================================
S = np.diag([0, 1, 1]); Sp = np.eye(3) - S
Fpos = np.array([5.0, 3.0, -2.0]); Fforce = np.array([8.0, 40.0, 7.0])
Fcmd = S@Fpos + Sp@Fforce
close('hybrid: F = S F_pos + S\' F_force = (8, 3, -2)', Fcmd, [8, 3, -2])
check('S + S\' = I and S S\' = 0 (complementary diagonals)', np.allclose(S + Sp, np.eye(3)) and np.allclose(S@Sp, 0))

# =====================================================================================
# 7b. numbers quoted on the tab that are not covered above
# =====================================================================================
close('quiz Q2: alpha = 0.24', 1.5*0.4**2, 0.24)
tq2 = 0.24*(0 + 10*0 + 25*0.1) + 1.5*G*0.4*np.cos(60*deg)
close('quiz Q2: tau = 3.543', tq2, 3.543, tol=5e-4)
close('quiz Q2: beta = 2.943', 1.5*G*0.4*np.cos(60*deg), 2.943, tol=5e-4)
check('P9.5: ZOH 10 ms peak error is 0.26 mrad (0.25..0.27)', 2.5e-4 < res[1e-2] < 2.7e-4, f'{res[1e-2]:.3e}')
check('P9.5: ZOH 50 ms peak error is 1.13 mrad (1.12..1.14)', 1.12e-3 < res[5e-2] < 1.14e-3, f'{res[5e-2]:.3e}')
tt_h, ee_h = sim_link1(2.8, 100.0, 20.0, 1e-4)
ih = int(np.argmax(np.abs(ee_h)))
close('P9.6: peak |e| = 3.35 mrad', abs(ee_h[ih])*1000, 3.35, tol=0.01)
close('P9.6: peak at t = 0.47 s', tt_h[ih], 0.47, tol=0.005)
close('P9.6: peak = 0.19 deg', abs(ee_h[ih])/deg, 0.19, tol=0.005)
close('P9.6: e(2 s) = -1.21 mrad', ee_h[int(2/1e-4)]*1000, -1.21, tol=0.01)
check('P9.6: error positive at the peak (link sags below reference)', ee_h[ih] > 0)
check('P9.6: e(4 s) below 1e-9 rad', abs(ee_h[-1]) < 1e-9)
close('P9.6: g/L = 3.27', G/L1, 3.27)
close('P9.6: forcing at q=0 hold = 0.2336', (1/14)*(G/L1), 0.2336, tol=5e-5)
close('P9.6: wrong answer 2.18 mrad = (0.2/3)(3.27)/100', (0.2/3)*(G/L1)/100*1000, 2.18, tol=0.005)
close('P9.7: coupling 30*2.3562 = 70.69', M0[0,1]*apk[1] + M0[0,2]*apk[2], 70.69, tol=5e-3)
close('P9.7: 0.6863/100 = 6.9 mrad; 0.6863/2.3562 = 29 %', [coupling1/100*1000, coupling1/apk[0]*100], [6.9, 29.1], tol=0.1)
close('P9.1: pi/2 / 2 = 0.7854 (average speed)', dq[0]/T, 0.7854, tol=5e-5)
close('P9.2: g1 lever arms 3,5,5 m', G*(3*3 + 2*5 + 1*5), 235.44)
close('P9.2: Q(0) = M qdd_d + g', Qff0, [548.8, 160.2, 2.4], tol=0.06)
close('P9.2: A, B, D coefficients', [(mm[1]+mm[2])*L1*L2, mm[2]*L1*L3, mm[2]*L2*L3], [18, 3, 2])
close('P9.3: wrong answers 63.62 and 88.29', [alphaA*apk[0], betaA0], [63.62, 88.29], tol=5e-3)
close('P9.9: x = 0.2, E2 = 4.17, wrong E2 (no mass) = 4.5455', [a_*kpv, e2, 5/(1+kpv/kev)], [0.2, 4.1667, 4.5455], tol=5e-4)
close('P9.10: wrong Q1 (sign slip) = -1.0359', -4.5 + 3.4641, -1.0359, tol=5e-5)
close('P9.5: e(0.25), e(0.5) analytic', [0.1*(1+10*0.25)*np.exp(-2.5), 0.1*(1+5)*np.exp(-5)], [0.0287, 0.00404], tol=5e-5)
close('P9.5: wrong answer 0.0368', 0.1*np.exp(-1), 0.0368, tol=5e-5)
# hybrid example for P9.11
S2 = np.diag([1, 1, 0]); Sp2 = np.eye(3) - S2
close('P9.11: F = (12, -3, 10)', S2@np.array([12., -3, 9]) + Sp2@np.array([1., 2, 10]), [12, -3, 10])
# =====================================================================================
# 8. ctc widget: simulation (JS mirrors this exactly) + test vectors
# =====================================================================================
WP = dict(m=1.0, L=1.0, g=G, bv=0.6, bc=1.0, eps=0.01, q0=0.4, A=0.8, dt=0.001, Tend=8.0)
def S_(v): return np.tanh(v/WP['eps'])

def ctc_sim(ctrl, k=1.0, kf=1.0, wn=8.0, wr=2.0, e0=0.1, Ts=1):
    """Ts = 1 ms (= dt): the law is evaluated inside every RK4 stage (continuous control). Ts > 1: torque held for Ts ms (ZOH)."""
    p = WP; dt = p['dt']; n_steps = int(round(p['Tend']/dt)); N = int(Ts)
    kp_, kv_ = wn*wn, 2*wn
    mh = k*p['m']; ah = mh*p['L']**2
    def ref(t):
        s, c = np.sin(wr*t), np.cos(wr*t)
        return p['q0'] + p['A']*s, p['A']*wr*c, -p['A']*wr*wr*s
    def law(t, q, v):
        qD, vD, aD = ref(t); e = qD - q; ed = vD - v
        if ctrl == 'a': return ah*(kp_*e + kv_*ed)
        if ctrl == 'b': return ah*(kp_*e + kv_*ed) + mh*p['g']*p['L']*np.cos(q)
        return ah*(aD + kv_*ed + kp_*e) + kf*p['bv']*v + kf*p['bc']*S_(v) + mh*p['g']*p['L']*np.cos(q)
    def acc(q, v, tau):
        return (tau - p['bv']*v - p['bc']*S_(v) - p['m']*p['g']*p['L']*np.cos(q))/(p['m']*p['L']**2)
    q = ref(0)[0] - e0; v = ref(0)[1]
    es = np.empty(n_steps+1); tau = 0.0
    for n in range(n_steps+1):
        t = n*dt
        es[n] = ref(t)[0] - q
        if n == n_steps: break
        if N == 1:
            k1q, k1v = v, acc(q, v, law(t, q, v))
            q2, v2 = q + dt/2*k1q, v + dt/2*k1v; k2q, k2v = v2, acc(q2, v2, law(t + dt/2, q2, v2))
            q3, v3 = q + dt/2*k2q, v + dt/2*k2v; k3q, k3v = v3, acc(q3, v3, law(t + dt/2, q3, v3))
            q4, v4 = q + dt*k3q, v + dt*k3v; k4q, k4v = v4, acc(q4, v4, law(t + dt, q4, v4))
        else:
            if n % N == 0: tau = law(t, q, v)
            k1q, k1v = v, acc(q, v, tau)
            k2q, k2v = v + dt/2*k1v, acc(q + dt/2*k1q, v + dt/2*k1v, tau)
            k3q, k3v = v + dt/2*k2v, acc(q + dt/2*k2q, v + dt/2*k2v, tau)
            k4q, k4v = v + dt*k3v, acc(q + dt*k3q, v + dt*k3v, tau)
        q += dt/6*(k1q + 2*k2q + 2*k3q + k4q); v += dt/6*(k1v + 2*k2v + 2*k3v + k4v)
    return es

def stats(es):
    return dict(rms=float(np.sqrt(np.mean(es**2))), max=float(np.max(np.abs(es))), final=float(es[-1]), e_1s=float(es[1000]), e_4s=float(es[4000]))

cases = [
    dict(name='defaults', k=1.0, kf=1.0, wn=8.0, wr=2.0, e0=0.1, Ts=1),
    dict(name='mass 2.8/3', k=2.8/3, kf=1.0, wn=8.0, wr=2.0, e0=0.1, Ts=1),
    dict(name='mass +30%, friction half', k=1.3, kf=0.5, wn=8.0, wr=2.0, e0=0.0, Ts=1),
    dict(name='slow servo 20 ms', k=1.0, kf=1.0, wn=15.0, wr=3.0, e0=0.0, Ts=20),
    dict(name='stiff gains, fast ref', k=0.8, kf=0.0, wn=20.0, wr=4.0, e0=0.05, Ts=5),
    dict(name='exact model, no initial error', k=1.0, kf=1.0, wn=8.0, wr=2.0, e0=0.0, Ts=1),
    dict(name='mass 2.8/3, no initial error', k=2.8/3, kf=1.0, wn=8.0, wr=2.0, e0=0.0, Ts=1),
    dict(name='exact model, 50 ms servo', k=1.0, kf=1.0, wn=8.0, wr=2.0, e0=0.0, Ts=50),
    dict(name='friction estimate zero only', k=1.0, kf=0.0, wn=8.0, wr=2.0, e0=0.0, Ts=1),
]
vec = []
for c in cases:
    row = dict(c); res_ = {}
    for ctl in 'abc':
        res_[ctl] = stats(ctc_sim(ctl, c['k'], c['kf'], c['wn'], c['wr'], c['e0'], c['Ts']))
    row['res'] = res_
    vec.append(row)
    print(f"   widget '{c['name']}': RMS a/b/c = " + ' / '.join(f"{res_[x]['rms']:.6f}" for x in 'abc'))
byname = {c['name']: c['res'] for c in vec}
r0 = byname['exact model, no initial error']
close('Try: exact model, e0=0: RMS a = 124.8, b = 39.3 mrad, c = 0', [r0['a']['rms']*1e3, r0['b']['rms']*1e3, r0['c']['rms']*1e3], [124.8, 39.3, 0.0], tol=0.06)
close('Try: e(0.5 s) with e0=0.1, wn=8 is 9.158 mrad', ctc_sim('c', 1.0, 1.0, 8.0, 2.0, 0.1, 1)[500]*1e3, 9.158, tol=0.0006)
r1 = byname['mass 2.8/3, no initial error']
close('Try: k = 0.933, e0=0: RMS c = 8.78 mrad; still below b', [r1['c']['rms']*1e3], [8.78], tol=0.006)
check('Try: ... and below (b)', r1['c']['rms'] < r1['b']['rms'])
r2 = byname['friction estimate zero only']
close('Try: friction factor 0: RMS c = 22.6 mrad (worse than 8.78)', r2['c']['rms']*1e3, 22.6, tol=0.06)
r3 = byname['exact model, 50 ms servo']
close('Try: 50 ms period, exact model: RMS c = 3.29 mrad', r3['c']['rms']*1e3, 3.29, tol=0.006)
check('widget: with exact model, CTC beats PD+gravity beats PD', vec[0]['res']['c']['rms'] < vec[0]['res']['b']['rms'] < vec[0]['res']['a']['rms'])

# exact model CTC: e(t) obeys e'' + kv e' + kp e = 0  (critically damped solution, zeta = 1, e(0) = e0, e'(0) = 0)
es = ctc_sim('c', 1.0, 1.0, 8.0, 2.0, 0.1, 1)
wn_ = 8.0
for tq in (0.1, 0.25, 0.5, 1.0):
    ana = 0.1*(1 + wn_*tq)*np.exp(-wn_*tq)
    close(f'widget CTC exact model: e({tq} s) = e0 (1 + wn t) e^(-wn t)', es[int(round(tq/0.001))], ana, tol=1e-6)
# residual of the ODE from the sampled trace (central differences)
dt = 0.001; ts = np.arange(len(es))*dt
ed = np.gradient(es, dt); edd = np.gradient(ed, dt)
resid = edd + 16*ed + 64*es
check('widget CTC exact model: |e\'\' + kv e\' + kp e| < 5e-2 along the trace (finite differences)', np.max(np.abs(resid[5:-5])) < 5e-2, f'max {np.max(np.abs(resid[5:-5])):.2e}')
# exact model, e0 = 0, Ts = 1: error is numerically ~0
es0 = ctc_sim('c', 1.0, 1.0, 8.0, 2.0, 0.0, 1)
check('widget CTC exact model, e0 = 0, continuous law: max |e| < 1e-6 rad', np.max(np.abs(es0)) < 1e-6, f'{np.max(np.abs(es0)):.2e}')
# imprecise-mass forcing reproduces slide p. 28 at sampled states (continuous law, algebra)
check('ratio for k = 2.8/3 is 1/14', abs((1-2.8/3)/(2.8/3) - 1/14) < 1e-12)
out['widget_ratio_k0.9333'] = (1-2.8/3)/(2.8/3)

json.dump(dict(params=WP, cases=vec, notes='rms/max/final in rad over t = 0..8 s inclusive (8001 samples); e_1s = e at t = 1 s, e_4s at t = 4 s'),
          open(os.path.join(HERE, 'w09_vectors.json'), 'w'), indent=1)

# =====================================================================================
# 9. cross-check: the JavaScript in the fragment reproduces the Python numbers (needs node)
# =====================================================================================
frag = os.path.join(HERE, '..', 'frag', '09-w09.html')
node = shutil.which('node')
if os.path.exists(frag) and node:
    src = open(frag, encoding='utf-8').read()
    mm_ = re.search(r'/\*SIM-BEGIN\*/(.*?)/\*SIM-END\*/', src, re.S)
    check('fragment contains the SIM block', bool(mm_))
    if mm_:
        js = mm_.group(1) + """
const vec = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8'));
const outp = vec.cases.map(c => { const r = ctcSim({k: c.k, kf: c.kf, wn: c.wn, wr: c.wr, e0: c.e0, Ts: c.Ts}); return {name: c.name, res: r.stats}; });
console.log(JSON.stringify(outp));
"""
        tmp = os.path.join(HERE, '_node_check.js')
        open(tmp, 'w').write(js)
        try:
            pr = subprocess.run([node, tmp, os.path.join(HERE, 'w09_vectors.json')], capture_output=True, text=True, timeout=120)
            if pr.returncode != 0:
                check('node ran the widget simulation', False, pr.stderr[:300])
            else:
                got = json.loads(pr.stdout)
                worst = 0.0
                for c, g_ in zip(vec, got):
                    for ctl in 'abc':
                        for key in ('rms', 'max', 'final', 'e_1s', 'e_4s'):
                            worst = max(worst, abs(c['res'][ctl][key] - g_['res'][ctl][key]))
                check('JS widget simulation == Python simulation on all test vectors (max abs diff < 1e-9)', worst < 1e-9, f'worst {worst:.2e}')
        finally:
            if os.path.exists(tmp): os.remove(tmp)
elif not node:
    print('   (node not found: JS cross-check skipped)')
else:
    print('   (fragment not written yet: JS cross-check skipped)')

report()
