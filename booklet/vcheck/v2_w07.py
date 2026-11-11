"""v2 booklet W7 (dynamics): every number added in w07.tex. Run: python3 v2_w07.py"""
import numpy as np, sympy as sp
from scipy.integrate import solve_ivp

deg = np.pi/180
t = sp.symbols('t')
g = sp.symbols('g', positive=True)
okall = True
def check(label, cond, extra=''):
    global okall
    okall &= bool(cond)
    print(('[PASS] ' if cond else '[FAIL] ') + label + ('  ' + extra if extra else ''))
def lab(s, v):
    print('   ' + s, np.round(np.array(v, dtype=float), 4).tolist() if hasattr(v, '__len__') else round(float(v), 4))

def lagrange(qs, KE, PE):
    L = KE - PE
    qd = [sp.diff(q, t) for q in qs]; qdd = [sp.diff(q, t, 2) for q in qs]
    tau = [sp.simplify(sp.diff(sp.diff(L, qd[i]), t) - sp.diff(L, qs[i])) for i in range(len(qs))]
    n = len(qs)
    M = sp.Matrix(n, n, lambda i, j: sp.simplify(sp.expand(tau[i]).coeff(qdd[j])))
    G = sp.Matrix([sp.simplify(tau[i].subs({x: 0 for x in qdd}).subs({x: 0 for x in qd})) for i in range(n)])
    C = sp.simplify(sp.Matrix(tau) - M*sp.Matrix(qdd) - G)
    return tau, M, C, G

def numfun(qs, tau, params):
    """lambdify tau(q,qd,qdd) numerically"""
    n = len(qs)
    a = sp.symbols('a0:%d' % n); ad = sp.symbols('ad0:%d' % n); add = sp.symbols('add0:%d' % n)
    sub = {}
    for i in range(n):
        sub[sp.diff(qs[i], t, 2)] = add[i]; sub[sp.diff(qs[i], t)] = ad[i]; sub[qs[i]] = a[i]
    f = sp.lambdify([a, ad, add], [sp.sympify(x).subs(sub).subs(params) for x in tau], 'numpy')
    return lambda q, qd, qdd: np.array(f(q, qd, qdd), float), sub, (a, ad, add)

# ============ 1. single link: pendulum, point mass and rod =============
th = sp.Function('th')(t)
m, l, Ic, lc = sp.symbols('m l I_c l_c', positive=True)
# point mass at distance l, angle from horizontal, gravity -y
Kp = sp.Rational(1, 2)*m*l**2*sp.diff(th, t)**2; Pp = m*g*l*sp.sin(th)
tau_p, Mp, Cp, Gp = lagrange([th], Kp, Pp)
check('pendulum point mass: Q = m l^2 qdd + m g l cos q', sp.simplify(tau_p[0] - (m*l**2*sp.diff(th, t, 2) + m*g*l*sp.cos(th))) == 0)
# rod: COM at l_c, inertia I_c about COM; K = 1/2 m lc^2 qd^2 + 1/2 Ic qd^2
Kr = sp.Rational(1, 2)*(m*lc**2 + Ic)*sp.diff(th, t)**2; Pr = m*g*lc*sp.sin(th)
tau_r, Mr, Cr, Gr = lagrange([th], Kr, Pr)
check('rod: Q = (I_c + m l_c^2) qdd + m g l_c cos q', sp.simplify(tau_r[0] - ((Ic + m*lc**2)*sp.diff(th, t, 2) + m*g*lc*sp.cos(th))) == 0)
# numbers
mm, ll = 2.0, 0.5
print('point mass m=2 kg, l=0.5 m, g=9.81:')
print('   hold at 30 deg: Q =', mm*9.81*ll*np.cos(30*deg))
print('   hold at 0 deg (horizontal): Q =', mm*9.81*ll, ' at 90 deg:', 0.0)
mr, lr = 2.0, 0.6; Ir = mr*lr**2/12; lcr = lr/2
print('rod m=2 kg, l=0.6 m: I_c =', Ir, ' I_end =', Ir + mr*lcr**2, ' m g l_c =', mr*9.81*lcr)
Qr = (Ir + mr*lcr**2)*4 + mr*9.81*lcr*np.cos(30*deg)
print('   rod at q=30deg, qdd=4: inertial', (Ir + mr*lcr**2)*4, ' gravity', mr*9.81*lcr*np.cos(30*deg), ' Q =', Qr)
# energy check: release from horizontal at rest, speed at the bottom q=-90deg
Ie = Ir + mr*lcr**2
qd_bottom = np.sqrt(2*mr*9.81*lcr/Ie)
def rhs(_, s): return [s[1], -(mr*9.81*lcr/Ie)*np.cos(s[0])]
ev = lambda _, s: s[0] + np.pi/2
ev.terminal = True
sol = solve_ivp(rhs, [0, 5], [0, 0], events=ev, rtol=1e-11, atol=1e-12)
print('   rod released from horizontal: qd at q=-90deg: energy', qd_bottom, ' ODE', abs(sol.y[1, -1]), ' time', sol.t[-1])
check('rod energy check qd_bottom = sqrt(2 m g l_c / I_end) agrees with ODE', abs(abs(sol.y[1, -1]) - qd_bottom) < 1e-6)
print('   rod 1/2 I qd^2 at bottom =', 0.5*Ie*qd_bottom**2, ' m g l_c =', mr*9.81*lcr)

# ============ 2. inertia: rod, cylinder, parallel axis =============
x, y, z, r_, h_, M_ = sp.symbols('x y z r h M', positive=True)
Lr = sp.symbols('Lr', positive=True)
Irod_c = sp.integrate(M_/Lr*x**2, (x, -Lr/2, Lr/2)); Irod_e = sp.integrate(M_/Lr*x**2, (x, 0, Lr))
check('thin rod: centre M L^2/12, end M L^2/3', sp.simplify(Irod_c - M_*Lr**2/12) == 0 and sp.simplify(Irod_e - M_*Lr**2/3) == 0)
rho = M_/(sp.pi*r_**2*h_)
rr_, ph_ = sp.symbols('rr ph', positive=True)
Izz_c = sp.integrate(sp.integrate(sp.integrate(rho*rr_**2*rr_, (rr_, 0, r_)), (ph_, 0, 2*sp.pi)), (z, -h_/2, h_/2))
Ixx_c = sp.integrate(sp.integrate(sp.integrate(rho*(rr_**2*sp.sin(ph_)**2 + z**2)*rr_, (rr_, 0, r_)), (ph_, 0, 2*sp.pi)), (z, -h_/2, h_/2))
check('solid cylinder: Izz = M r^2/2; Ixx = Iyy = M(3 r^2 + h^2)/12', sp.simplify(Izz_c - M_*r_**2/2) == 0 and sp.simplify(Ixx_c - M_*(3*r_**2 + h_**2)/12) == 0)
W, Lc, H, mb = sp.symbols('W L H m', positive=True)
rho_b = mb/(W*Lc*H)
Ib = lambda f: sp.integrate(f*rho_b, (x, 0, W), (y, 0, Lc), (z, 0, H))
Icorner = sp.Matrix([[Ib(y**2 + z**2), -Ib(x*y), -Ib(x*z)], [-Ib(x*y), Ib(x**2 + z**2), -Ib(y*z)], [-Ib(x*z), -Ib(y*z), Ib(x**2 + y**2)]])
Ic_ = sp.diag(mb*(Lc**2 + H**2)/12, mb*(W**2 + H**2)/12, mb*(Lc**2 + W**2)/12)
p = sp.Matrix([W/2, Lc/2, H/2])
check('parallel axis: corner = centroid + m(p.p 1 - p p^T)', sp.simplify(Ic_ + mb*((p.T*p)[0]*sp.eye(3) - p*p.T) - Icorner) == sp.zeros(3, 3))
sub_b = {mb: 6, W: 0.2, Lc: 0.3, H: 0.1}
print('box 6 kg W=.2 L=.3 H=.1: centroid', np.array(Ic_.subs(sub_b), float).tolist(), ' corner', np.array(Icorner.subs(sub_b), float).tolist())
# a second parallel-axis numeric: tensor of the same box about the frame at the centre of the face z=0 ... use shift along x only
pshift = sp.Matrix([0.5, 0, 0])
Ish = np.array((Ic_.subs(sub_b) + 6*((pshift.T*pshift)[0]*sp.eye(3) - pshift*pshift.T)), float)
print('   box centroid tensor shifted by p=(0.5,0,0): ', Ish.tolist())
# principal axes example
A = np.array([[2, 1, 0], [1, 2, 0], [0, 0, 3.0]])
w_, V_ = np.linalg.eigh(A)
print('principal-axes example I=[[2,1,0],[1,2,0],[0,0,3]]: eigenvalues', w_.tolist(), ' eigenvectors (columns)', np.round(V_, 4).tolist())
# omega x I omega for omega not along principal axis
om = np.array([1.0, 0, 0]); print('   w=(1,0,0): I w =', (A@om).tolist(), ' w x I w =', np.cross(om, A@om).tolist())
om2 = np.array([1.0, 1.0, 0]); print('   w=(1,1,0): I w =', (A@om2).tolist(), ' w x I w =', np.cross(om2, A@om2).tolist())

# ============ 3. planar RR point masses (slides pp18-25) =============
m1, m2, L1, L2 = sp.symbols('m1 m2 L1 L2', positive=True)
q1, q2 = sp.Function('q1')(t), sp.Function('q2')(t)
d1, d2 = sp.diff(q1, t), sp.diff(q2, t)
x1, y1 = L1*sp.cos(q1), L1*sp.sin(q1)
x2, y2 = x1 + L2*sp.cos(q1 + q2), y1 + L2*sp.sin(q1 + q2)
K1 = sp.Rational(1, 2)*m1*(sp.diff(x1, t)**2 + sp.diff(y1, t)**2)
K2 = sp.Rational(1, 2)*m2*(sp.diff(x2, t)**2 + sp.diff(y2, t)**2)
P = m1*g*y1 + m2*g*y2
tau2R, M2, C2, G2 = lagrange([q1, q2], K1 + K2, P)
c2, s2 = sp.cos(q2), sp.sin(q2)
A_ = m1*L1**2 + m2*(L1**2 + 2*L1*L2*c2 + L2**2); B_ = m2*(L1*L2*c2 + L2**2); D_ = m2*L2**2
check('RR: K = 1/2 A qd1^2 + B qd1 qd2 + 1/2 D qd2^2', sp.simplify(K1 + K2 - (A_*d1**2/2 + B_*d1*d2 + D_*d2**2/2)) == 0)
hh = m2*L1*L2*s2
check('RR: M', sp.simplify(M2 - sp.Matrix([[A_, B_], [B_, D_]])) == sp.zeros(2, 2))
check('RR: C', sp.simplify(C2 - sp.Matrix([-hh*(2*d1*d2 + d2**2), hh*d1**2])) == sp.zeros(2, 1))
check('RR: g', sp.simplify(G2 - sp.Matrix([(m1 + m2)*g*L1*sp.cos(q1) + m2*g*L2*sp.cos(q1 + q2), m2*g*L2*sp.cos(q1 + q2)])) == sp.zeros(2, 1))
# derivative steps shown in the text
check('RR step: dA/dt = -2 m2 L1 L2 s2 qd2, dB/dt = -m2 L1 L2 s2 qd2',
      sp.simplify(sp.diff(A_, t) + 2*hh*d2) == 0 and sp.simplify(sp.diff(B_, t) + hh*d2) == 0)
detM = sp.simplify(M2.det())
check('RR: det M = m2 L1^2 L2^2 (m1 + m2 sin^2 q2) > 0', sp.simplify(detM - m2*L1**2*L2**2*(m1 + m2*s2**2)) == 0)
# C-matrix (Christoffel) and skew property
Cmat = sp.Matrix([[-hh*d2, -hh*(d1 + d2)], [hh*d1, 0]])
check('RR: Cmat(q,qd) qd = C(q,qd)', sp.simplify(Cmat*sp.Matrix([d1, d2]) - C2) == sp.zeros(2, 1))
Sk = sp.simplify(sp.diff(M2, t) - 2*Cmat)
check('RR: Mdot - 2 Cmat is skew-symmetric', sp.simplify(Sk + Sk.T) == sp.zeros(2, 2))
print('   Mdot - 2Cmat =', sp.simplify(Sk))
params = {m1: 2.0, m2: 1.0, L1: 1.0, L2: 0.8, g: 9.81}
f2R, sub2R, (aa, aad, aadd) = numfun([q1, q2], tau2R, params)
qa, qda, qdda = np.array([30*deg, 45*deg]), np.array([1.0, -0.5]), np.array([0.5, 2.0])
Mf = sp.lambdify([aa], M2.subs(sub2R).subs(params), 'numpy'); Cf = sp.lambdify([aa, aad], C2.subs(sub2R).subs(params), 'numpy'); Gf = sp.lambdify([aa], G2.subs(sub2R).subs(params), 'numpy')
Mn = np.array(Mf(qa), float); Cn = np.array(Cf(qa, qda), float).ravel(); Gn = np.array(Gf(qa), float).ravel()
Qn = f2R(qa, qda, qdda)
print('RR example m=(2,1) L=(1,0.8) q=(30,45)deg qd=(1,-.5) qdd=(.5,2):')
lab('M', Mn); lab('C', Cn); lab('g', Gn); lab('Q', Qn)
print('   by hand: A c2 =', np.cos(45*deg), ' A =', 2 + (1 + 2*0.8*np.cos(45*deg) + 0.64), ' B =', 0.8*np.cos(45*deg) + 0.64, ' h=', 0.8*np.sin(45*deg))
print('   Q1 pieces: M11 qdd1 =', Mn[0, 0]*0.5, ' M12 qdd2 =', Mn[0, 1]*2, ' C1 =', Cn[0], ' g1 =', Gn[0])
print('   Q2 pieces: M21 qdd1 =', Mn[1, 0]*0.5, ' M22 qdd2 =', Mn[1, 1]*2, ' C2 =', Cn[1], ' g2 =', Gn[1])
print('   det M =', np.linalg.det(Mn), ' formula', 0.64*1*(2 + 1*np.sin(45*deg)**2))
# power balance q^T Q = dK/dt + dP/dt
Mdot = sp.diff(M2, t).subs(sub2R).subs(params)
Mdotf = sp.lambdify([aa, aad], Mdot, 'numpy'); Mdn = np.array(Mdotf(qa, qda), float)
Kdot = qda@Mn@qdda + 0.5*qda@Mdn@qda; Pdot = qda@Gn
print('   power: qd.Q =', qda@Qn, ' dK/dt =', Kdot, ' dP/dt =', Pdot, ' sum =', Kdot + Pdot, ' qd.C =', qda@Cn, ' 1/2 qd.Mdot.qd =', 0.5*qda@Mdn@qda)
check('RR power balance qd.Q = dK/dt + dP/dt', abs(qda@Qn - (Kdot + Pdot)) < 1e-9)
check('RR qd.C = 1/2 qd.Mdot.qd', abs(qda@Cn - 0.5*qda@Mdn@qda) < 1e-9)
# hold still, accelerate from rest, forward dynamics
print('   hold still q=(30,45): Q = g =', Gn)
Gh = np.array(Gf(np.array([0.0, 0.0])), float).ravel(); print('   hold at (0,0): g =', Gh)
Qacc = Mn@qdda + Gn; print('   from rest at q=(30,45), qdd=(.5,2): Q = M qdd + g =', Qacc, ' (M qdd =', Mn@qdda, ')')
qdd_fall = np.linalg.solve(Mn, -Gn); print('   forward dynamics: Q=0, qd=0 at (30,45): qdd =', qdd_fall)
qdd_f2 = np.linalg.solve(Mn, np.array([10.0, 0.0]) - Cn - Gn); print('   forward dyn. Q=(10,0), qd=(1,-.5): qdd =', qdd_f2)
# energy: free swing from rest
Ef = lambda qq, vv: 0.5*vv@np.array(Mf(qq), float)@vv + (params[m1]+params[m2])*9.81*params[L1]*np.sin(qq[0]) + params[m2]*9.81*params[L2]*np.sin(qq[0]+qq[1]) - 0
def rhs2(_, s):
    qq, vv = s[:2], s[2:]
    return np.concatenate([vv, np.linalg.solve(np.array(Mf(qq), float), -np.array(Cf(qq, vv), float).ravel() - np.array(Gf(qq), float).ravel())])
s0 = [30*deg, 0, 0, 0]; sol = solve_ivp(rhs2, [0, 2], s0, rtol=1e-11, atol=1e-11, dense_output=True)
E0 = Ef(np.array(s0[:2]), np.array(s0[2:])); E1 = Ef(sol.y[:2, -1], sol.y[2:, -1])
print('   free swing from (30,0) at rest: E(0) =', E0, ' E(2s) =', E1, ' q(2s) deg =', (sol.y[:2, -1]/deg).tolist(), ' qd =', sol.y[2:, -1].tolist())
check('RR free-swing energy constant', abs(E0 - E1) < 1e-7)
# static energy example: both links horizontal released (0,0): speed check not needed.

# ============ 4. NE for 2R: symbolic = Lagrange; numeric vectors =============
def ne_2R_numeric(q, qd, qdd, m=(2.0, 1.0), L=(1.0, 0.8), grav=9.81, verbose=False):
    Z = np.array([0, 0, 1.0])
    Rz = lambda a: np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rs = [Rz(q[0]), Rz(q[1])]
    Ps = [np.zeros(3), np.array([L[0], 0, 0])]
    Pc = [np.array([L[0], 0, 0]), np.array([L[1], 0, 0])]
    w = np.zeros(3); wd = np.zeros(3); vd = np.array([0, grav, 0.0])
    F = []; log = {}
    for i in range(2):
        R = Rs[i]
        vd = R.T @ (np.cross(wd, Ps[i]) + np.cross(w, np.cross(w, Ps[i])) + vd)
        wd = R.T @ wd + np.cross(R.T @ w, qd[i]*Z) + qdd[i]*Z
        w = R.T @ w + qd[i]*Z
        vc = np.cross(wd, Pc[i]) + np.cross(w, np.cross(w, Pc[i])) + vd
        F.append(m[i]*vc)
        log[i+1] = dict(w=w.copy(), wd=wd.copy(), vd=vd.copy(), vc=vc.copy(), F=F[-1].copy())
    f = np.zeros(3); n = np.zeros(3); taus = [0, 0]
    for i in (1, 0):
        Rn = Rs[i+1] if i+1 < 2 else np.eye(3); Pn = Ps[i+1] if i+1 < 2 else np.zeros(3)
        fnew = Rn @ f + F[i]
        n = Rn @ n + np.cross(Pc[i], F[i]) + np.cross(Pn, Rn @ f)
        f = fnew; taus[i] = n @ Z
        log[i+1].update(f=f.copy(), n=n.copy())
    return np.array(taus), log
tau_ne, log = ne_2R_numeric(qa, qda, qdda)
print('NE 2R at the example state:')
for i in (1, 2):
    for k, v in log[i].items(): print('   link', i, k, np.round(v, 4).tolist())
print('   tau NE =', tau_ne)
check('NE == Lagrange at example', np.allclose(tau_ne, Qn, atol=1e-9))
# NE symbolic
a_ = -L1*d1**2 + g*sp.sin(q1); b_ = L1*sp.diff(q1, t, 2) + g*sp.cos(q1)
F2y = m2*(L2*(sp.diff(q1, t, 2) + sp.diff(q2, t, 2)) + (-sp.sin(q2)*a_ + sp.cos(q2)*b_))
F2x = m2*(-L2*(d1 + d2)**2 + (sp.cos(q2)*a_ + sp.sin(q2)*b_))
F1y = m1*b_
tau2_sym = L2*F2y
tau1_sym = tau2_sym + L1*F1y + L1*(sp.sin(q2)*F2x + sp.cos(q2)*F2y)
check('NE symbolic tau2 == Lagrange Q2', sp.simplify(tau2_sym - tau2R[1]) == 0)
check('NE symbolic tau1 == Lagrange Q1', sp.simplify(tau1_sym - tau2R[0]) == 0)
check('NE: (-s2 a + c2 b) = L1 s2 qd1^2 + L1 c2 qdd1 + g c12', sp.simplify((-sp.sin(q2)*a_ + sp.cos(q2)*b_) - (L1*sp.sin(q2)*d1**2 + L1*sp.cos(q2)*sp.diff(q1, t, 2) + g*sp.cos(q1 + q2))) == 0)
# single link prac Q1 numbers
b, c = sp.symbols('b c')
for (bb, cc, tt) in [(1.0, 0.5, 1.0)]:
    qd_ = bb + 2*cc*tt; qdd_ = 2*cc
    w_ = np.array([0, 0, qd_]); wd_ = np.array([0, 0, qdd_]); Pc_ = np.array([3.0, 0, 0]); Ic_m = np.diag([1.0, 2, 3])
    vdc = np.cross(wd_, Pc_) + np.cross(w_, np.cross(w_, Pc_)); F_ = 1.0*vdc; N_ = Ic_m@wd_ + np.cross(w_, Ic_m@w_)
    n_ = N_ + np.cross(Pc_, F_)
    print('single link b=1, c=0.5, t=1: qd =', qd_, ' qdd =', qdd_, ' vdotC =', vdc.tolist(), ' F =', F_.tolist(), ' N =', N_.tolist(), ' n =', n_.tolist(), ' tau =', n_[2])
    check('single link: tau = 24 c', abs(n_[2] - 24*cc) < 1e-12)
# NE for RP (prismatic outward rule), compare with Lagrange RP
th1, dd2 = sp.Function('th1')(t), sp.Function('d2')(t)
mA, mB, lA, IA, IB = sp.symbols('m_1 m_2 l_1 I_1 I_2', positive=True)
KRP = sp.Rational(1, 2)*(mA*lA**2 + IA)*sp.diff(th1, t)**2 + sp.Rational(1, 2)*mB*(dd2**2*sp.diff(th1, t)**2 + sp.diff(dd2, t)**2) + sp.Rational(1, 2)*IB*sp.diff(th1, t)**2
PRP = mA*g*lA*sp.sin(th1) + mB*g*dd2*sp.sin(th1)
tauRP, MRP, CRP, GRP = lagrange([th1, dd2], KRP, PRP)
tdot = sp.diff(th1, t); ddot = sp.diff(dd2, t)
check('RP: tau1', sp.simplify(tauRP[0] - ((mA*lA**2 + IA + IB + mB*dd2**2)*sp.diff(th1, t, 2) + 2*mB*dd2*tdot*ddot + (mA*lA + mB*dd2)*g*sp.cos(th1))) == 0)
check('RP: f2', sp.simplify(tauRP[1] - (mB*sp.diff(dd2, t, 2) - mB*dd2*tdot**2 + mB*g*sp.sin(th1))) == 0)
check('RP: M diagonal', sp.simplify(MRP - sp.diag(mA*lA**2 + IA + IB + mB*dd2**2, mB)) == sp.zeros(2, 2))
paramsRP = {mA: 2.0, mB: 1.0, lA: 0.3, IA: 0.05, IB: 0.02, g: 9.81}
fRP, subRP, (ra, rad, radd) = numfun([th1, dd2], tauRP, paramsRP)
qR, qdR, qddR = np.array([0.7, 0.6]), np.array([1.3, -0.8]), np.array([2.0, 1.5])
tauR = fRP(qR, qdR, qddR)
print('RP example: tau =', tauR)
print('   tau1 pieces: M11 =', 2*0.09 + 0.05 + 0.02 + 1*0.36, ' M11*qdd1 =', (2*0.09+0.05+0.02+0.36)*2.0, ' Coriolis 2 m2 d2 th1d d2d =', 2*1*0.6*1.3*(-0.8), ' gravity =', (2*0.3 + 0.6)*9.81*np.cos(0.7))
print('   f2 pieces: m2 d2dd =', 1.5, ' -m2 d2 th1d^2 =', -0.6*1.69, ' m2 g sin th1 =', 9.81*np.sin(0.7))
print('   RP hold still th=30deg, d2=0.6: tau1 =', (2*0.3 + 0.6)*9.81*np.cos(30*deg), ' f2 =', 9.81*np.sin(30*deg))
# RP power balance
MRPn = sp.lambdify([ra], MRP.subs(subRP).subs(paramsRP), 'numpy'); GRPn = sp.lambdify([ra], GRP.subs(subRP).subs(paramsRP), 'numpy')
MdRP = sp.lambdify([ra, rad], sp.diff(MRP, t).subs(subRP).subs(paramsRP), 'numpy')
Mr_ = np.array(MRPn(qR), float); Gr_ = np.array(GRPn(qR), float).ravel(); Mdr_ = np.array(MdRP(qR, qdR), float)
dE = qdR@Mr_@qddR + 0.5*qdR@Mdr_@qdR + qdR@Gr_
print('   RP power: qd.tau =', qdR@tauR, ' dK/dt+dP/dt =', dE)
check('RP power balance', abs(qdR@tauR - dE) < 1e-9)
# RP energy check: free fall from rest of the second mass? static: d2 slides freely at th=90deg -> f2 = m2 d2dd + m2 g = 0  => d2dd = -g
print('   RP: theta1 = 90deg fixed, no force: d2dd = -g (free fall):', -9.81)

# ============ 5. PR (cart + pendulum link), vertical prismatic =============
dd1 = sp.Function('d1')(t); ph = sp.Function('ph')(t)
M1c, M2c, Lp = sp.symbols('M1 M2 Lp', positive=True)
xx2 = dd1 + Lp*sp.cos(ph); yy2 = Lp*sp.sin(ph)
Kpr = sp.Rational(1, 2)*M1c*sp.diff(dd1, t)**2 + sp.Rational(1, 2)*M2c*(sp.diff(xx2, t)**2 + sp.diff(yy2, t)**2)
Ppr = M2c*g*yy2
tauPR, MPR, CPR, GPR = lagrange([dd1, ph], Kpr, Ppr)
cp_, sp_ = sp.cos(ph), sp.sin(ph)
check('PR: M', sp.simplify(MPR - sp.Matrix([[M1c + M2c, -M2c*Lp*sp_], [-M2c*Lp*sp_, M2c*Lp**2]])) == sp.zeros(2, 2))
check('PR: C = [-m2 L c qd^2; 0] (Coriolis cancels)', sp.simplify(CPR - sp.Matrix([-M2c*Lp*cp_*sp.diff(ph, t)**2, 0])) == sp.zeros(2, 1))
check('PR: g = [0; m2 g L c]', sp.simplify(GPR - sp.Matrix([0, M2c*g*Lp*cp_])) == sp.zeros(2, 1))
pPR = {M1c: 3.0, M2c: 1.0, Lp: 0.5, g: 9.81}
fPR, subPR, (pa, pad, padd) = numfun([dd1, ph], tauPR, pPR)
qP, qdP, qddP = np.array([0.2, 40*deg]), np.array([1.0, 2.0]), np.array([0.5, 1.0])
tauP = fPR(qP, qdP, qddP)
print('PR example m1=3 m2=1 L=0.5 phi=40deg d1d=1 phid=2 d1dd=.5 phidd=1: tau =', tauP)
print('   F1 pieces: (m1+m2) d1dd =', 4*0.5, ' -m2 L s phidd =', -0.5*np.sin(40*deg)*1, ' -m2 L c phid^2 =', -0.5*np.cos(40*deg)*4)
print('   tau pieces: -m2 L s d1dd =', -0.5*np.sin(40*deg)*0.5, ' m2 L^2 phidd =', 0.25, ' m2 g L c =', 9.81*0.5*np.cos(40*deg))
print('   PR hold still phi=40: F1 = 0, tau =', 9.81*0.5*np.cos(40*deg))
MPRn = sp.lambdify([pa], MPR.subs(subPR).subs(pPR), 'numpy'); GPRn = sp.lambdify([pa], GPR.subs(subPR).subs(pPR), 'numpy')
MdPR = sp.lambdify([pa, pad], sp.diff(MPR, t).subs(subPR).subs(pPR), 'numpy')
Mq = np.array(MPRn(qP), float); Gq = np.array(GPRn(qP), float).ravel(); Mdq = np.array(MdPR(qP, qdP), float)
dEp = qdP@Mq@qddP + 0.5*qdP@Mdq@qdP + qdP@Gq
print('   PR power: qd.tau =', qdP@tauP, ' dK/dt+dP/dt =', dEp)
check('PR power balance', abs(qdP@tauP - dEp) < 1e-9)
print('   PR forward dyn: F1=0 (free cart), tau=0, phi=40 at rest: qdd =', np.linalg.solve(Mq, -Gq))
print('   vertical prismatic: f = m (zdd + g): m=2, zdd=1.5 ->', 2*(1.5 + 9.81), ' hold ->', 2*9.81)

# ============ 6. 3D RRR (W13 Q4, W8 prac Q2): point masses and general slender links =============
T1, T2, T3 = [sp.Function('T%d' % i)(t) for i in (1, 2, 3)]
Hh, L3, L4, M2_, M3_, I1_, I2_, I3_, l2_, l3_ = sp.symbols('H L3 L4 m2 m3 I1 I2 I3 l2 l3', positive=True)
def rrr(l2, l3, Iflag):
    rho2 = l2*sp.cos(T2); h2 = Hh + l2*sp.sin(T2)
    rho3 = L3*sp.cos(T2) + l3*sp.cos(T2 + T3); h3 = Hh + L3*sp.sin(T2) + l3*sp.sin(T2 + T3)
    v2 = sp.diff(rho2, t)**2 + sp.diff(h2, t)**2 + rho2**2*sp.diff(T1, t)**2
    v3 = sp.diff(rho3, t)**2 + sp.diff(h3, t)**2 + rho3**2*sp.diff(T1, t)**2
    K = sp.Rational(1, 2)*M2_*v2 + sp.Rational(1, 2)*M3_*v3
    if Iflag:  # slender links: inertia I about axes perpendicular to the link, ~0 about its own axis
        K += sp.Rational(1, 2)*I1_*sp.diff(T1, t)**2
        K += sp.Rational(1, 2)*I2_*(sp.diff(T2, t)**2 + (sp.cos(T2)*sp.diff(T1, t))**2)
        K += sp.Rational(1, 2)*I3_*((sp.diff(T2, t) + sp.diff(T3, t))**2 + (sp.cos(T2 + T3)*sp.diff(T1, t))**2)
    P = M2_*g*h2 + M3_*g*h3
    return lagrange([T1, T2, T3], K, P), rho2, rho3
(tauA, MA, CA, GA), rho2, rho3 = rrr(L3, L4, False)
c2_, s2_, c3_, s3_, c23, s23 = sp.cos(T2), sp.sin(T2), sp.cos(T3), sp.sin(T3), sp.cos(T2 + T3), sp.sin(T2 + T3)
td1, td2, td3 = sp.diff(T1, t), sp.diff(T2, t), sp.diff(T3, t)
rho2s = L3*c2_; rho3s = L3*c2_ + L4*c23
Mf3 = sp.Matrix([[M2_*rho2s**2 + M3_*rho3s**2, 0, 0], [0, (M2_ + M3_)*L3**2 + M3_*L4**2 + 2*M3_*L3*L4*c3_, M3_*(L4**2 + L3*L4*c3_)], [0, M3_*(L4**2 + L3*L4*c3_), M3_*L4**2]])
Gf3 = sp.Matrix([0, g*((M2_ + M3_)*L3*c2_ + M3_*L4*c23), g*M3_*L4*c23])
V1 = -2*td1*(M2_*rho2s*L3*s2_*td2 + M3_*rho3s*(L3*s2_*td2 + L4*s23*(td2 + td3)))
V2 = td1**2*(M2_*rho2s*L3*s2_ + M3_*rho3s*(L3*s2_ + L4*s23)) - M3_*L3*L4*s3_*(2*td2*td3 + td3**2)
V3 = td1**2*M3_*rho3s*L4*s23 + M3_*L3*L4*s3_*td2**2
check('RRR point masses: M', sp.simplify(MA - Mf3) == sp.zeros(3, 3))
check('RRR point masses: g', sp.simplify(GA - Gf3) == sp.zeros(3, 1))
_dv = sp.simplify(sp.expand_trig(CA - sp.Matrix([V1, V2, V3])))
_rng = np.random.default_rng(7); _okv = True
for _ in range(20):
    _vals = {L3: _rng.uniform(.3, 1), L4: _rng.uniform(.3, 1), M2_: _rng.uniform(.5, 3), M3_: _rng.uniform(.5, 3)}
    for _f in (T1, T2, T3): _vals[_f] = _rng.uniform(-2, 2)
    for _f in (td1, td2, td3): _vals[_f] = _rng.uniform(-2, 2)
    _okv &= bool(np.allclose(np.array((CA - sp.Matrix([V1, V2, V3])).subs({sp.diff(T1,t,2):0,sp.diff(T2,t,2):0,sp.diff(T3,t,2):0}).subs(_vals).evalf(), float), 0, atol=1e-9))
check('RRR point masses: V (symbolic simplify or 20 random numeric states)', _dv == sp.zeros(3, 1) or _okv)
# general slender links (site W13 Q4 solution)
(tauB, MB, CB, GB), _, _ = rrr(l2_, l3_, True)
rho2g = l2_*c2_; rho3g = L3*c2_ + l3_*c23
MgenSite = sp.Matrix([[I1_ + M2_*rho2g**2 + M3_*rho3g**2 + I2_*c2_**2 + I3_*c23**2, 0, 0],
    [0, M2_*l2_**2 + I2_ + M3_*(L3**2 + l3_**2 + 2*L3*l3_*c3_) + I3_, M3_*(l3_**2 + L3*l3_*c3_) + I3_],
    [0, M3_*(l3_**2 + L3*l3_*c3_) + I3_, M3_*l3_**2 + I3_]])
GgenSite = sp.Matrix([0, g*((M2_*l2_ + M3_*L3)*c2_ + M3_*l3_*c23), g*M3_*l3_*c23])
V1g = -2*td1*(M2_*rho2g*l2_*s2_*td2 + M3_*rho3g*(L3*s2_*td2 + l3_*s23*(td2 + td3)) + I2_*c2_*s2_*td2 + I3_*c23*s23*(td2 + td3))
V2g = td1**2*(M2_*rho2g*l2_*s2_ + M3_*rho3g*(L3*s2_ + l3_*s23) + I2_*c2_*s2_ + I3_*c23*s23) - M3_*L3*l3_*s3_*(2*td2*td3 + td3**2)
V3g = td1**2*(M3_*rho3g*l3_*s23 + I3_*c23*s23) + M3_*L3*l3_*s3_*td2**2
check('RRR general: M', sp.simplify(MB - MgenSite) == sp.zeros(3, 3))
check('RRR general: g', sp.simplify(GB - GgenSite) == sp.zeros(3, 1))
check('RRR general: V', sp.simplify(CB - sp.Matrix([V1g, V2g, V3g])) == sp.zeros(3, 1))
pR = {M2_: 2.0, M3_: 1.0, L3: 0.6, L4: 0.4, Hh: 0.5, g: 9.81}
fR, subR, (ra3, rad3, radd3) = numfun([T1, T2, T3], tauA, pR)
qA, qdA, qddA = np.array([20*deg, 30*deg, -40*deg]), np.array([1.0, 0.5, -0.5]), np.array([0.2, 1.0, 0.5])
tauA_n = fR(qA, qdA, qddA)
MAn = np.array(sp.lambdify([ra3], MA.subs(subR).subs(pR), 'numpy')(qA), float)
CAn = np.array(sp.lambdify([ra3, rad3], CA.subs(subR).subs(pR), 'numpy')(qA, qdA), float).ravel()
GAn = np.array(sp.lambdify([ra3], GA.subs(subR).subs(pR), 'numpy')(qA), float).ravel()
print('RRR example m2=2 m3=1 L3=.6 L4=.4 q=(20,30,-40)deg qd=(1,.5,-.5) qdd=(.2,1,.5):')
lab('M', MAn); lab('C', CAn); lab('g', GAn); lab('Q', tauA_n)
r2n = 0.6*np.cos(30*deg); r3n = 0.6*np.cos(30*deg) + 0.4*np.cos(-10*deg)
print('   rho2 =', r2n, ' rho3 =', r3n, ' M11 =', 2*r2n**2 + r3n**2, ' c3 =', np.cos(-40*deg), ' s3 =', np.sin(-40*deg), ' c23 =', np.cos(-10*deg), ' s23 =', np.sin(-10*deg))
print('   hold still: g =', GAn, '; arm stretched horizontally (0,0,0): g =', np.array(sp.lambdify([ra3], GA.subs(subR).subs(pR), 'numpy')(np.zeros(3)), float).ravel())
MdA = sp.lambdify([ra3, rad3], sp.diff(MA, t).subs(subR).subs(pR), 'numpy')
dEA = qdA@MAn@qddA + 0.5*qdA@np.array(MdA(qA, qdA), float)@qdA + qdA@GAn
print('   RRR power: qd.Q =', qdA@tauA_n, ' dK/dt + dP/dt =', dEA)
check('RRR power balance', abs(qdA@tauA_n - dEA) < 1e-9)
check('RRR M positive-definite & symmetric', np.allclose(MAn, MAn.T) and np.all(np.linalg.eigvalsh(MAn) > 0))
# energy conservation, free swing of the 3D arm
MfL = sp.lambdify([ra3], MA.subs(subR).subs(pR), 'numpy'); CfL = sp.lambdify([ra3, rad3], CA.subs(subR).subs(pR), 'numpy'); GfL = sp.lambdify([ra3], GA.subs(subR).subs(pR), 'numpy')
def rhs3(_, s):
    qq, vv = s[:3], s[3:]
    return np.concatenate([vv, np.linalg.solve(np.array(MfL(qq), float), -np.array(CfL(qq, vv), float).ravel() - np.array(GfL(qq), float).ravel())])
def E3(s):
    qq, vv = s[:3], s[3:]
    return 0.5*vv@np.array(MfL(qq), float)@vv + 9.81*(2*(0.5 + 0.6*np.sin(qq[1])) + 1*(0.5 + 0.6*np.sin(qq[1]) + 0.4*np.sin(qq[1]+qq[2])))
s0 = np.array([0, 30*deg, -40*deg, 1.0, 0, 0]); sol = solve_ivp(rhs3, [0, 3], s0, rtol=1e-11, atol=1e-11)
check('RRR free swing energy constant over 3 s', abs(E3(sol.y[:, -1]) - E3(s0)) < 1e-7, 'drift %.1e' % abs(E3(sol.y[:, -1]) - E3(s0)))

# ============ 7. planar 3R (W8 practical Q3) =============
q3 = [sp.Function('p%d' % i)(t) for i in (1, 2, 3)]
Ls = (3.0, 2.0, 1.0); ms = (3.0, 2.0, 1.0)
a1, a12, a123 = q3[0], q3[0] + q3[1], q3[0] + q3[1] + q3[2]
px = [Ls[0]*sp.cos(a1), Ls[0]*sp.cos(a1) + Ls[1]*sp.cos(a12), Ls[0]*sp.cos(a1) + Ls[1]*sp.cos(a12) + Ls[2]*sp.cos(a123)]
py = [Ls[0]*sp.sin(a1), Ls[0]*sp.sin(a1) + Ls[1]*sp.sin(a12), Ls[0]*sp.sin(a1) + Ls[1]*sp.sin(a12) + Ls[2]*sp.sin(a123)]
K3 = sum(sp.Rational(1, 2)*ms[i]*(sp.diff(px[i], t)**2 + sp.diff(py[i], t)**2) for i in range(3))
P3 = sum(ms[i]*g*py[i] for i in range(3))
tau3, M3, C3, G3 = lagrange(q3, K3, P3)
f3, sub3, (a3_, ad3_, add3_) = numfun(q3, tau3, {g: 9.81})
M3f = sp.lambdify([a3_], M3.subs(sub3).subs({g: 9.81}), 'numpy'); G3f = sp.lambdify([a3_], G3.subs(sub3).subs({g: 9.81}), 'numpy')
q0 = np.array([0, 0, 90*deg]); T = 2.0
qf = np.array([90*deg, 90*deg, 0]); Dq = qf - q0
qt = lambda tt: q0 + Dq*(3*(tt/T)**2 - 2*(tt/T)**3)
qdt = lambda tt: Dq*(6*tt/T**2 - 6*tt**2/T**3)
qddt = lambda tt: Dq*(6/T**2 - 12*tt/T**3)
print('3R practical:')
print('   M(q0) =', np.round(np.array(M3f(q0), float), 4).tolist(), ' g(q0) =', np.round(np.array(G3f(q0), float).ravel(), 4).tolist())
for tt in (0.0, 1.0, 2.0):
    print('   t=%g: q(deg)=' % tt, np.round(qt(tt)/deg, 3).tolist(), ' qd=', np.round(qdt(tt), 4).tolist(), ' qdd=', np.round(qddt(tt), 4).tolist(), ' Q=', np.round(f3(qt(tt), qdt(tt), qddt(tt)), 3).tolist())
M0 = np.array(M3f(q0), float); G0 = np.array(G3f(q0), float).ravel()
print('   hand: M qdd(0) =', M0@qddt(0.0), ' Q(0) = ', M0@qddt(0.0) + G0)
check('3R: Q(0) = (548.814, 160.176, 2.356)', np.allclose(M0@qddt(0.0) + G0, [548.814, 160.176, 2.356], atol=2e-3))
# Jacobian-method hand M: tips (3,0),(5,0),(5,1)
tips = [np.array([3, 0.]), np.array([5, 0.]), np.array([5, 1.])]; origins = [np.array([0, 0.]), np.array([3, 0.]), np.array([5, 0.])]
Jc = []
for i in range(3):
    J = np.zeros((2, 3))
    for j in range(i+1):
        r = tips[i] - origins[j]; J[:, j] = [-r[1], r[0]]
    Jc.append(J)
Mj = sum(ms[i]*Jc[i].T@Jc[i] for i in range(3))
print('   M(q0) by m_i J_i^T J_i =', Mj.tolist())
check('3R: M(q0) Jacobian method == Lagrange', np.allclose(Mj, M0, atol=1e-9))
print('   g(q0) by hand: g1 =', 9.81*(3*3 + 2*5 + 1*5), ' g2 =', 9.81*(2*2 + 1*2), ' g3 =', 9.81*(1*0))
print('   Q(0) rows: ', 133*(90*deg*3/T**2*0+3*np.pi/2/ (T**2)), )
# at t=1: qdd=0 => Q = C + g
print('   t=1: qd = Dq*1.5/T =', Dq*1.5/T)

# ============ 8. cost (p47) =============
ne_m = lambda n: 104*n - 77; ne_a = lambda n: 92*n - 70
la_m = lambda n: 32*n**4 + 86*n**3 + 171*n**2 + 53*n - 128; la_a = lambda n: 25*n**4 + 66*n**3 + 129*n**2 + 42*n - 96
check('cost n=6: 547, 482, 66394, 51456', (ne_m(6), ne_a(6), la_m(6), la_a(6)) == (547, 482, 66394, 51456))
print('   n=2: NE', ne_m(2), ne_a(2), ' Lagrange', la_m(2), la_a(2), ' n=3: NE', ne_m(3), ' Lagrange', la_m(3), ' ratio n=6:', la_m(6)/ne_m(6))

# ============ 9. reused exam-style questions (Practice Q&A / mock), re-verified here =============
# Q7.3 2R arm point masses
def rr_tau(m1_, m2_, L1_, L2_, q, qd, qdd, gg=9.81):
    c2_, s2_ = np.cos(q[1]), np.sin(q[1]); h_ = m2_*L1_*L2_*s2_
    M_ = np.array([[m1_*L1_**2 + m2_*(L1_**2 + 2*L1_*L2_*c2_ + L2_**2), m2_*(L1_*L2_*c2_ + L2_**2)], [m2_*(L1_*L2_*c2_ + L2_**2), m2_*L2_**2]])
    C_ = np.array([-h_*(2*qd[0]*qd[1] + qd[1]**2), h_*qd[0]**2])
    G_ = np.array([(m1_ + m2_)*gg*L1_*np.cos(q[0]) + m2_*gg*L2_*np.cos(q[0] + q[1]), m2_*gg*L2_*np.cos(q[0] + q[1])])
    return M_, C_, G_, M_@qdd + C_ + G_
M_, C_, G_, T_ = rr_tau(1.5, 0.8, 0.9, 0.6, np.array([60*deg, -30*deg]), np.array([0.8, 1.2]), np.array([-1.0, 2.5]))
print('Q7.3: M =', M_.tolist(), ' C =', C_.tolist(), ' g =', G_.tolist(), ' tau =', T_.tolist())
check('Q7.3 tau = (13.713, 3.998)', np.allclose(T_, [13.713, 3.998], atol=1e-3))
# uniform rods 2R (mock): m=(3,2), l=(0.6,0.4), COM at mid, Ic=ml^2/12, q=(45,-30), qdd=(1,2) at rest
mu1, mu2, lu1, lu2 = 3.0, 2.0, 0.6, 0.4
qq = np.array([45*deg, -30*deg]); c2u = np.cos(qq[1])
I1u, I2u = mu1*lu1**2/12, mu2*lu2**2/12
Mu = np.array([[I1u + mu1*lu1**2/4 + I2u + mu2*(lu1**2 + lu2**2/4 + lu1*lu2*c2u), I2u + mu2*(lu2**2/4 + lu1*lu2*c2u/2)], [I2u + mu2*(lu2**2/4 + lu1*lu2*c2u/2), I2u + mu2*lu2**2/4]])
Gu = np.array([(mu1*lu1/2 + mu2*lu1)*9.81*np.cos(qq[0]) + mu2*lu2/2*9.81*np.cos(qq[0] + qq[1]), mu2*lu2/2*9.81*np.cos(qq[0] + qq[1])])
Tu = Mu@np.array([1.0, 2.0]) + Gu
print('uniform rods: M =', Mu.tolist(), ' g =', Gu.tolist(), ' tau =', Tu.tolist())
check('uniform rods tau = (20.5888, 4.3181)', np.allclose(Tu, [20.5888, 4.3181], atol=1e-3))
# verify uniform-rod M by Lagrange with rod energies
mu_ = sp.symbols('mu_')
Kr1 = sp.Rational(1, 2)*(3*0.6**2/12 + 3*(0.3)**2)*sp.diff(q1, t)**2
xc2 = 0.6*sp.cos(q1) + 0.2*sp.cos(q1 + q2); yc2 = 0.6*sp.sin(q1) + 0.2*sp.sin(q1 + q2)
Kr2 = sp.Rational(1, 2)*2*(sp.diff(xc2, t)**2 + sp.diff(yc2, t)**2) + sp.Rational(1, 2)*(2*0.4**2/12)*(sp.diff(q1, t) + sp.diff(q2, t))**2
Pr_ = 3*9.81*0.3*sp.sin(q1) + 2*9.81*yc2
tau_u, Mu_s, Cu_s, Gu_s = lagrange([q1, q2], Kr1 + Kr2, Pr_)
fu, subu, (ua, uad, uadd) = numfun([q1, q2], tau_u, {})
check('uniform rods: Lagrange (rod energies) agrees with the formula', np.allclose(fu(qq, np.zeros(2), np.array([1.0, 2.0])), Tu, atol=1e-9))
# RP mock (massless arm, point mass m=2): theta=30, d=0.6, thd=0.5, dd=0.2, thdd=1, ddd=-0.3
mrp, thr, dr, thd_, dd_, thdd_, ddd_ = 2.0, 30*deg, 0.6, 0.5, 0.2, 1.0, -0.3
t1 = mrp*dr**2*thdd_ + 2*mrp*dr*dd_*thd_ + mrp*9.81*dr*np.cos(thr); f2_ = mrp*ddd_ - mrp*dr*thd_**2 + mrp*9.81*np.sin(thr)
print('RP point mass mock: tau1 =', t1, ' f2 =', f2_)
check('RP mock (11.1549, 8.9100)', abs(t1 - 11.1549) < 1e-3 and abs(f2_ - 8.91) < 1e-3)
# spot-the-error RP
m_, d_, dd1_, ddd1_, th_, thd1_, thdd1_ = 1.5, 0.4, 0.3, 0.5, 0.9, 1.1, 0.6
f2e = m_*(ddd1_ - d_*thd1_**2) + m_*9.81*np.sin(th_); t1e = m_*d_**2*thdd1_ + 2*m_*d_*dd1_*thd1_ + m_*9.81*d_*np.cos(th_)
print('RP error question: f2 =', f2e, ' tau1 =', t1e, ' student f2 =', m_*ddd1_ + m_*9.81*np.cos(th_), ' student tau1 =', m_*9.81*d_*np.cos(th_))
check('RP error question (11.551, 4.1988)', abs(f2e - 11.551) < 1e-3 and abs(t1e - 4.1988) < 1e-3)
# inertia shifted
Mb = 4.0; Wb, Lb, Hb = 0.3, 0.2, 0.5
Icb = Mb/12*np.diag([Lb**2 + Hb**2, Wb**2 + Hb**2, Lb**2 + Wb**2]); pb = np.array([0, 0.1, 0.25])
IAb = Icb + Mb*(pb@pb*np.eye(3) - np.outer(pb, pb))
print('shifted box tensor:', np.round(IAb, 5).tolist(), ' centroid', np.round(np.diag(Icb), 5).tolist())
check('shifted box: Ixx = 0.38667, Iyz = -0.1', abs(IAb[0, 0] - 0.38667) < 1e-4 and abs(IAb[1, 2] + 0.1) < 1e-12)
# holding with wrench w on slide 2R
q_ = np.array([30*deg, 45*deg]); L1_, L2_ = 1.0, 0.8
Jv = np.array([[-L1_*np.sin(q_[0]) - L2_*np.sin(q_[0]+q_[1]), -L2_*np.sin(q_[0]+q_[1])], [L1_*np.cos(q_[0]) + L2_*np.cos(q_[0]+q_[1]), L2_*np.cos(q_[0]+q_[1])]])
wv = np.array([-5.0, 12.0]); Qh = Gn + Jv.T@wv
print('hold with wrench w=(-5,12): J =', np.round(Jv, 4).tolist(), ' J^T w =', (Jv.T@wv).tolist(), ' Q =', Qh.tolist())
check('hold with wrench (46.759, 8.380)', abs(Qh[0] - 46.759) < 1e-3 and abs(Qh[1] - 8.380) < 1e-3)
# cart pendulum (phi from downward vertical) Q7.1
Mc, mc, lc_, ph_, phd_, phdd_, xdd_ = 2.0, 1.0, 0.5, 30*deg, 2.0, 3.0, 1.5
Fc = (Mc + mc)*xdd_ + mc*lc_*phdd_*np.cos(ph_) - mc*lc_*phd_**2*np.sin(ph_); tc = mc*lc_**2*phdd_ + mc*lc_*xdd_*np.cos(ph_) + mc*9.81*lc_*np.sin(ph_)
print('cart pendulum: F =', Fc, ' tau =', tc)
# 2R with q1 and prismatic - hold: total check 3R t=1 C+g done above
print('ALL CHECKS PASSED' if okall else 'SOME CHECKS FAILED')
