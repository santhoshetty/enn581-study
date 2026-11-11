"""v2 booklet W9 (computed torque, Cartesian, force, hybrid): numbers in booklet/w09.tex. FAIL lines must not appear."""
import numpy as np, sympy as sp
from scipy.integrate import solve_ivp
np.set_printoptions(precision=4, suppress=True)
G = 9.81; deg = np.pi/180
fails = []
def chk(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: fails.append(name)
def close(name, got, want, tol=1e-3):
    ok = np.allclose(np.asarray(got, float), np.asarray(want, float), atol=tol, rtol=tol)
    print(("PASS " if ok else "FAIL ") + f"{name}: got {np.round(np.asarray(got,float),5).tolist()} want {np.asarray(want).tolist()}")
    if not ok: fails.append(name)
def show(l, x): print(f"  {l}: {np.round(np.asarray(x,float),5).tolist()}")

print("=== 1 SISO cubic spring: symbolic closed loop")
x, xd, xdd, m, b, k, kp, kv, xD, xdD, xddD = sp.symbols('x xd xdd m b k kp kv xD xdD xddD')
fp = xddD + kv*(xdD-xd) + kp*(xD-x)
f = m*fp + b*xd + k*x**3
xdd_sol = sp.solve(sp.Eq(m*xdd + b*xd + k*x**3, f), xdd)[0]
chk("plant xdd = f' (alpha=m, beta=b xd + k x^3)", sp.simplify(xdd_sol - fp) == 0)
# EQ1 numbers
m1, b1, k1, wn = 2., .5, 4., 8.
kp1, kv1 = wn**2, 2*wn
E1, Ed1 = .15-.10, 0-(-.2)
fp1 = 0 + kv1*Ed1 + kp1*E1; be1 = b1*(-.2) + k1*.10**3; f1 = m1*fp1 + be1
close("EQ1 kp,kv", [kp1, kv1], [64, 16]); close("EQ1 E,Edot", [E1, Ed1], [.05, .2])
close("EQ1 f'", fp1, 6.4); close("EQ1 beta", be1, -0.096); close("EQ1 f", f1, 12.704)
print("  e_ss with constant extra force fw=5 N:", -5/(m1*kp1))
close("EQ1 ess", -5/(m1*kp1), -0.0390625)
# simulate to confirm ess
fw = 5.
def rhs(t, s):
    X, V = s; fc = m1*(0 + kv1*(0-V) + kp1*(0.15-X)) + b1*V + k1*X**3
    return [V, (fc + fw - b1*V - k1*X**3)/m1]
sol = solve_ivp(rhs, [0, 5], [0.1, 0], rtol=1e-10, atol=1e-12)
close("EQ1 sim ess", 0.15 - sol.y[0, -1], -0.0390625, 1e-5)
# v1 worked
fpw = .5 + 20*.02*5 - 0  # placeholder replaced below
m_, b_, k_ = 2., 4., 50.; fpv = .5 + 20*(.4-.3) + 100*(.12-.10); bev = b_*.3 + k_*.1**3
close("v1 cubic: f',beta,f", [fpv, bev, m_*fpv+bev], [4.5, 1.25, 10.25])

print("=== 2 single link + friction")
q, t = sp.symbols('q t'); mm, L, g = sp.symbols('m L g', positive=True)
qt = sp.Function('q')(t)
Lag = sp.Rational(1,2)*mm*L**2*qt.diff(t)**2 - mm*g*L*sp.sin(qt)
tau = sp.diff(sp.diff(Lag, qt.diff(t)), t) - sp.diff(Lag, qt)
chk("Lagrange single link", sp.simplify(tau - (mm*L**2*qt.diff(t,2) + mm*g*L*sp.cos(qt))) == 0)
al = .5*.25*0+2*.5**2; tp = 2 + 16*(-.1) + 64*.05; bt = .3*1 + .2 + 2*G*.5*np.cos(30*deg)
close("v1 single link tau'", tp, 3.6); close("beta", bt, 8.9957); close("tau", al*tp+bt, 10.7957)
# EQ2 (Q9.1)
mt, mh, Lt, kpt, qt0 = 3., 2.5, .6, 81., 20*deg
ess = (mt-mh)*G*np.cos(qt0)/(mt*Lt*kpt)
close("EQ2 ess", ess, 0.03161, 2e-4); print("  deg:", ess/deg)

print("=== 3 imprecise model: 10 % mass error, single link m=2 L=0.5")
m0, L0, kp0, kv0 = 2., .5, 100., 20.
for mh_, name in [(1.8, "m_hat=0.9m"), (2.2, "m_hat=1.1m")]:
    for q0 in (0, 30*deg, 90*deg):
        def rhs(t, s):
            Q, V = s; tp = 0 + kv0*(0-V) + kp0*(q0-Q)
            tau = mh_*L0**2*tp + mh_*G*L0*np.cos(Q)
            return [V, (tau - m0*G*L0*np.cos(Q))/(m0*L0**2)]
        s = solve_ivp(rhs, [0, 8], [q0, 0], rtol=1e-11, atol=1e-13)
        # steady state error vs prediction ((m-mh)/mh)(g/L)cos q/kp  (q is the actual angle at equilibrium)
        qf = s.y[0, -1]
        pred = (m0-mh_)/mh_*G/L0*np.cos(qf)/kp0
        print(f"  {name} hold q_d={q0/deg:.0f} deg: sim e_ss={q0-qf:.6f} rad ({(q0-qf)/deg:.4f} deg), formula {pred:.6f}, cos(q_d) version {(m0-mh_)/mh_*G/L0*np.cos(q0)/kp0:.6f}")
        chk(f"{name} q_d={q0/deg:.0f}: sim = formula (actual q)", abs((q0-qf)-pred) < 1e-8)
close("ratio (m-mh)/mh", [(2-1.8)/1.8, (2-2.2)/2.2], [0.1111, -0.0909])
close("g/L", G/L0, 19.62)
close("ess 0.9m, q=0", (2-1.8)/1.8*19.62/100, 0.0218, 1e-4)
close("ess 0.9m, q=30", (2-1.8)/1.8*19.62*np.cos(30*deg)/100, 0.01888, 1e-4)
close("ess 1.1m, q=0", (2-2.2)/2.2*19.62/100, -0.017836, 1e-4)
print("  0.9m q=0 deg:", 0.0218/deg, " 30:", 0.018878/deg, " 1.1m:", -0.017836/deg)

print("=== 4 MIMO general (2R point masses l=1,1 m=1,1)")
q1, q2, d1, d2, a1, a2 = sp.symbols('q1 q2 d1 d2 a1 a2')
m1_, m2_, l1, l2 = 1, 1, 1, 1
qs = sp.Matrix([q1, q2]); ds = sp.Matrix([d1, d2])
p1 = sp.Matrix([l1*sp.cos(q1), l1*sp.sin(q1)]); p2 = p1 + sp.Matrix([l2*sp.cos(q1+q2), l2*sp.sin(q1+q2)])
Jv = lambda p: p.jacobian(qs)
K = sp.Rational(1,2)*(m1_*(Jv(p1)*ds).dot(Jv(p1)*ds) + m2_*(Jv(p2)*ds).dot(Jv(p2)*ds))
M = sp.simplify(sp.hessian(K, list(ds))); gs = sp.Symbol('g')
P = gs*(m1_*p1[1] + m2_*p2[1]); grav = sp.Matrix([sp.diff(P, q1), sp.diff(P, q2)])
Cv = sp.Matrix([sum(sp.diff(M[i,j], qs[k_])*ds[j]*ds[k_] for j in range(2) for k_ in range(2)) - sp.Rational(1,2)*sum(sp.diff(M[j,k_], qs[i])*ds[j]*ds[k_] for j in range(2) for k_ in range(2)) for i in range(2)])
print("  M =", sp.simplify(M)); print("  C =", sp.simplify(Cv)); print("  g =", sp.simplify(grav))
chk("M11 = 3+2c2, M12 = 1+c2, M22 = 1", sp.simplify(M - sp.Matrix([[3+2*sp.cos(q2), 1+sp.cos(q2)],[1+sp.cos(q2), 1]])) == sp.zeros(2))
chk("C1 = -s2(2 d1 d2 + d2^2), C2 = s2 d1^2", sp.simplify(Cv - sp.Matrix([-sp.sin(q2)*(2*d1*d2+d2**2), sp.sin(q2)*d1**2])) == sp.zeros(2,1))
chk("g1 = g(2c1+c12), g2 = g c12", sp.simplify(grav - gs*sp.Matrix([2*sp.cos(q1)+sp.cos(q1+q2), sp.cos(q1+q2)])) == sp.zeros(2,1))
fM = sp.lambdify((q1,q2), M, 'numpy'); fC = sp.lambdify((q1,q2,d1,d2), Cv, 'numpy'); fg = sp.lambdify((q1,q2,gs), grav, 'numpy')
fJ = sp.lambdify((q1,q2), p2.jacobian(qs), 'numpy')
rng = np.random.default_rng(1)
Kp, Kv = np.diag([100., 100.]), np.diag([20., 20.])
ok = True
for _ in range(20):   # exact model: E'' + Kv E' + Kp E = 0
    qq = rng.uniform(-2, 2, 2); dd = rng.uniform(-1, 1, 2); qdD = rng.uniform(-1, 1, 2); qddD = rng.uniform(-1, 1, 2); qD = rng.uniform(-2, 2, 2)
    Mq = np.array(fM(*qq), float); Cq = np.array(fC(*qq, *dd), float).ravel(); gq = np.array(fg(*qq, G), float).ravel()
    Qp = qddD + Kv@(qdD-dd) + Kp@(qD-qq); Q = Mq@Qp + Cq + gq
    qdd = np.linalg.solve(Mq, Q - Cq - gq)
    ok &= np.allclose((qddD-qdd) + Kv@(qdD-dd) + Kp@(qD-qq), 0, atol=1e-9)
chk("exact model: Edd+Kv Ed+Kp E=0 (20 states)", ok)
ok = True
for _ in range(20):   # imprecise: slide p.28
    qq = rng.uniform(-2, 2, 2); dd = rng.uniform(-1, 1, 2); qdD = rng.uniform(-1, 1, 2); qddD = rng.uniform(-1, 1, 2); qD = rng.uniform(-2, 2, 2)
    Mq = np.array(fM(*qq), float); Cq = np.array(fC(*qq, *dd), float).ravel(); gq = np.array(fg(*qq, G), float).ravel()
    Mh = 1.1*Mq + np.array([[.1, .05],[.05, .2]]); Ch = .9*Cq; gh = 1.2*gq; fh = np.array([.1, .1]); ft = np.array([.3, -.2]) 
    E = qD-qq; Ed = qdD-dd; Qp = qddD + Kv@Ed + Kp@E; Q = Mh@Qp + Ch + gh + fh
    qdd = np.linalg.solve(Mq, Q - Cq - gq - ft)
    lhs = (qddD-qdd) + Kv@Ed + Kp@E
    rhs = np.linalg.solve(Mh, (Mq-Mh)@qdd + (Cq-Ch) + (gq-gh) + (ft-fh))
    ok &= np.allclose(lhs, rhs, atol=1e-9)
chk("imprecise model: slide p.28 equation (20 states)", ok)

print("=== 5 gains for critical damping")
for w in (10, 15, 20): print(f"  wn={w}: kp={w**2}, kv={2*w}, t2%={5.8335/w:.3f} s, t5%={4.7439/w:.3f} s")
from scipy.optimize import brentq
c2 = brentq(lambda z: (1+z)*np.exp(-z)-0.02, 1, 20); c5 = brentq(lambda z: (1+z)*np.exp(-z)-0.05, 1, 20)
close("2% and 5% settling constants", [c2, c5], [5.8335, 4.7439], 1e-3)
close("wn for 0.5 s (2%)", c2/0.5, 11.667, 1e-3); close("kp,kv for wn=12", [144, 24], [144, 24])
close("kv^2=4kp roots equal", np.roots([1, 24, 144]).real, [-12, -12], 1e-6)
close("e(0.1) = 0.1(1+10t)e^-10t at kp=100,kv=20", 0.1*(1+1)*np.exp(-1), 0.073576, 1e-4)

print("=== 6 practical 3-link (L=3,2,1; m=3,2,1)")
th = sp.symbols('t1 t2 t3'); dth = sp.symbols('d1 d2 d3'); Ls = [3, 2, 1]; ms = [3, 2, 1]
pts = []; acc = sp.Matrix([0, 0]); ang = 0
for i in range(3):
    ang = ang + th[i]; acc = acc + Ls[i]*sp.Matrix([sp.cos(ang), sp.sin(ang)]); pts.append(acc)
qv = sp.Matrix(th); dv = sp.Matrix(dth)
K3 = sum(sp.Rational(1,2)*ms[i]*(pts[i].jacobian(qv)*dv).dot(pts[i].jacobian(qv)*dv) for i in range(3))
M3 = sp.simplify(sp.hessian(K3, list(dv))); g3 = sp.Matrix([sp.diff(sum(ms[i]*gs*pts[i][1] for i in range(3)), th[j]) for j in range(3)])
f3 = sp.lambdify((*th, gs), (M3, g3), 'numpy')
Mq0, gq0 = f3(0, 0, np.pi/2, G); Mq0 = np.array(Mq0, float); gq0 = np.array(gq0, float).ravel()
show("M(q0)", Mq0); show("g(q0)", gq0)
close("M(q0)", Mq0, [[103,31,1],[31,13,1],[1,1,1]]); close("g(q0)", gq0, [235.44, 58.86, 0])
c1_, c12_, c123_ = sp.cos(th[0]), sp.cos(th[0]+th[1]), sp.cos(th[0]+th[1]+th[2]); c2_, c3_, c23_ = sp.cos(th[1]), sp.cos(th[2]), sp.cos(th[1]+th[2])
Mform = sp.Matrix([[67+36*c2_+6*c23_+4*c3_, 13+18*c2_+3*c23_+4*c3_, 1+3*c23_+2*c3_],[0, 13+4*c3_, 1+2*c3_],[0,0,1]])
Mform = sp.Matrix(3, 3, lambda i, j: Mform[min(i,j), max(i,j)])
chk("M closed form (symbolic)", sp.simplify(M3 - Mform) == sp.zeros(3))
chk("g closed form", sp.simplify(g3 - gs*sp.Matrix([18*c1_+6*c12_+c123_, 6*c12_+c123_, c123_])) == sp.zeros(3, 1))
Mg, gg = f3(np.pi/2, np.pi/2, 0, G); show("M(q_goal)", Mg); show("g(q_goal)", np.array(gg, float).ravel())
close("M goal", np.array(Mg, float), [[71,17,3],[17,17,3],[3,3,1]]); close("g goal", np.array(gg, float).ravel(), [-68.67, -68.67, -9.81])
T = 2.; Dq = np.array([np.pi/2, np.pi/2, -np.pi/2]); qdd0 = 6*Dq/T**2
Q0 = Mq0@qdd0 + gq0; show("qdd_d(0)", qdd0); show("Q(0)", Q0)
close("Q(0)", Q0, [548.81, 160.18, 2.356], 1e-3)
close("peak speed 1.5 D/T", 1.5*Dq[0]/T, 1.1781, 1e-4)
close("link1 alone: alpha, beta0, tau0", [27, 88.29, 27*qdd0[0]+88.29], [27, 88.29, 151.907], 1e-3)
close("full arm joint 1 alpha, beta", [Mq0[0,0], gq0[0]], [103, 235.44])
close("m1=2.8: M11hat, g1hat, forcing", [103-.2*9, 235.44-.2*G*3, (.2*G*3)/101.2], [101.2, 229.554, .0582], 1e-3)
close("single link m=3,L=3,mh=2.8: ratio, g/L, e_ss(kp=100) mrad", [1/14, G/3, (1/14)*(G/3)/100*1000], [.0714, 3.27, 2.336], 1e-3)
# gain choice for the practical: wn per joint
close("practical joint 1: kp,kv = 100,20; alpha*kp, alpha*kv (N m/rad, N m s/rad)", [27*100, 27*20], [2700, 540])

print("=== 7 EQ3: 2R computed torque numbers")
qq = np.array([30, 60])*deg; dd = np.array([1.0, -0.5]); qddD = np.array([.5, 0]); Ev = np.array([.02, -.01]); Evd = np.zeros(2)
Mq = np.array(fM(*qq), float); Cq = np.array(fC(*qq, *dd), float).ravel(); gq = np.array(fg(*qq, G), float).ravel()
Qp = qddD + Kv@Evd + Kp@Ev; Q = Mq@Qp + Cq + gq
show("M", Mq); show("C", Cq); show("g", gq); show("Q'", Qp); show("Q", Q)
show("M Q'", Mq@Qp)
Qp0 = np.array([.5, 0.]); show("M@[0.5,0] (coupling)", Mq@Qp0)

print("=== 8 Cartesian")
Jq = np.array(fJ(*qq), float); show("J", Jq)
close("J at (30,60)", Jq, [[-1.5, -1], [.866, 0]], 1e-3)
Fv = np.array([3., -4.]); show("J^T F", Jq.T@Fv); close("Q=J^T F", Jq.T@Fv, [-7.9641, -3.0], 1e-4)
J2 = np.array(fJ(60*deg, -30*deg), float)
# Q9.2 arm l=0.5,0.4
def J2R(a, b_, l1, l2): return np.array([[-l1*np.sin(a)-l2*np.sin(a+b_), -l2*np.sin(a+b_)],[l1*np.cos(a)+l2*np.cos(a+b_), l2*np.cos(a+b_)]])
Ja = J2R(60*deg, -30*deg, .5, .4); show("J (0.5,0.4)", Ja); Qa = Ja.T@np.array([-2., 6.]); show("Q", Qa)
close("Q9.2", Qa, [4.8445, 2.4785], 1e-3)
# Cartesian law vs joint law, point masses 2R at q=(30,60): xdd_d, K_p e_x, rest
Jinv = np.linalg.inv(Jq); Mx = Jinv.T@Mq@Jinv; gx = Jinv.T@gq
show("det J", [np.linalg.det(Jq)]); show("Mx", Mx); show("gx", gx)
ex = np.array([.02, -.01]); Kpx = np.diag([400., 400.]); a_x = Kpx@ex   # xdd_d=0, velocities 0
Fc = Mx@a_x + gx; Qc = Jq.T@Fc; show("F", Fc); show("Q via J^T F", Qc)
Qj = Mq@(Jinv@a_x) + gq; show("Q via joint space qdd=J^-1 a", Qj)
close("Cartesian = joint route (zero velocity)", Qc, Qj, 1e-6)
# slide p.31 error
qdv = np.array([.7, -.4]); eps = 1e-6
def Jf(qv): return np.array(fJ(*qv), float)
dJinv = (np.linalg.inv(Jf(qq+eps*qdv)) - np.linalg.inv(Jf(qq-eps*qdv)))/(2*eps)
Jdot = (Jf(qq+eps*qdv) - Jf(qq-eps*qdv))/(2*eps)
chk("d/dt(J^-1) = -J^-1 Jdot J^-1", np.allclose(dJinv, -Jinv@Jdot@Jinv, atol=1e-5))
# full c_x check with velocity
qd_ = np.array([.7, -.4]); xdd_ = np.array([.3, .2])
qdd_ = Jinv@(xdd_ - Jdot@qd_ ); Qjs = Mq@qdd_ + np.array(fC(*qq, *qd_), float).ravel() + gq
cx = Jinv.T@(np.array(fC(*qq, *qd_), float).ravel() - Mq@Jinv@Jdot@qd_)
chk("Q = J^T(Mx xdd + cx + gx)", np.allclose(Jq.T@(Mx@xdd_ + cx + gx), Qjs))
# singular start of practical arm: det of planar J
close("det J 2R = l1 l2 s2", np.linalg.det(Jq), np.sin(60*deg))

print("=== 9 force control")
def opt(mm, ke, kpp, fd): X = mm/ke*kpp; return X, fd/X, fd/(1+X)
X, E1_, E2_ = opt(2, 1000, 100, 5); close("m=2 ke=1000 kp=100 fdist=5: X,E1,E2", [X, E1_, E2_], [.2, 25, 4.1667], 1e-4)
X, E1_, E2_ = opt(2, 20, 100, 5); close("ke=20: X,E1,E2", [X, E1_, E2_], [10, .5, .4545], 1e-4)
close("kp for E2=1% fdist", 99/(2/1000), 49500)
X, E1_, E2_ = opt(1.5, 300, 64, 8); close("Q9.4: x,E1,E2", [X, E1_, E2_], [.32, 25, 6.0606], 1e-4)
X, E1_, E2_ = opt(2, 800, 100, 6); close("Q9.6: x, E1, E2", [X, E1_, E2_], [.25, 24, 4.8], 1e-4)
close("E2/E1 = X/(1+X)", 0.2/1.2, (5/1.2)/25, 1e-6)
# char. polynomials
close("Opt 2: wn^2 = kp + ke/m = 600; critical kv", [100+1000/2, 2*np.sqrt(600)], [600, 48.99], 1e-3)
close("Opt 1: critical kv for kp=100", 2*np.sqrt(100), 20)
# simulate both options
def sim(opt2, mm=2., ke=1000., kpp=100., kvv=20., fd=10., fdist=5.):
    a = mm/ke
    def rhs(t, s):
        x_, v_ = s; fe = ke*x_; E = fd - fe; Ed = -ke*v_
        fcmd = a*(0 + kvv*Ed + kpp*E) + (fd if opt2 else fe)
        return [v_, (fcmd - fdist - fe)/mm]
    s = solve_ivp(rhs, [0, 5], [0, 0], rtol=1e-10, atol=1e-12); return fd - ke*s.y[0, -1]
close("sim Opt1 ss E = 25", sim(False), 25, 1e-4); close("sim Opt2 ss E = 4.1667", sim(True), 4.1667, 1e-4)
# practical law == Option 2
fdv, fe, xdv = 10., 7., .003; Ex = fdv - fe; Exd = -1000*xdv
o2 = 2/1000*(20*Exd + 100*Ex) + fdv; pr = 2*(1/1000*100*Ex - 20*xdv) + fdv
close("Opt2 law = practical law = 10.48", [o2, pr], [10.48, 10.48], 1e-9)
close("practical: terms 0.6 and 0.12", [2*100*Ex/1000, 2*20*xdv], [.6, .12])

print("=== 10 hybrid")
S = np.diag([1., 1., 0]); Sp = np.eye(3) - S
close("wipe: F", S@[12, -3, 9] + Sp@[1, 2, 10], [12, -3, 10])
close("Q9.5 eraser F", S@[2, -1, 6] + Sp@[.5, .3, 12], [2, -1, 12])
S6 = np.diag([0, 0, 1, 0, 0, 1.]); S6p = np.eye(6) - S6
Fh = S6@[1, 2, -15, .3, -.2, .8] + S6p@[-3, 4, 2, .5, -.6, .1]; show("peg F", Fh)
close("peg wrench", Fh, [-3, 4, -15, .5, -.6, .8])
chk("S S' = 0, S+S'=I", np.allclose(S6@S6p, 0) and np.allclose(S6+S6p, np.eye(6)))
# wall normal x
Sx = np.diag([0, 1, 1.]); close("wall normal x: F", Sx@[12, -3, 9] + (np.eye(3)-Sx)@[1, 2, 10], [1, -3, 9])
# eraser Q: J^T F with 2R? (just statement)

print("=== 11 extras: scaled model, 3-link C closed form")
ok = True
for _ in range(20):
    qq = rng.uniform(-2, 2, 2); dd = rng.uniform(-1, 1, 2); qdD = rng.uniform(-1, 1, 2); qddD = rng.uniform(-1, 1, 2); qD = rng.uniform(-2, 2, 2)
    Mq = np.array(fM(*qq), float); Cq = np.array(fC(*qq, *dd), float).ravel(); gq = np.array(fg(*qq, G), float).ravel(); lam = 0.9
    E = qD-qq; Ed = qdD-dd; Qp = qddD + Kv@Ed + Kp@E; Q = lam*(Mq@Qp + Cq + gq)
    qdd = np.linalg.solve(Mq, Q - Cq - gq); lhs = (qddD-qdd) + Kv@Ed + Kp@E
    rhs = (1-lam)/lam*(qdd + np.linalg.solve(Mq, Cq+gq))
    ok &= np.allclose(lhs, rhs, atol=1e-9)
chk("all parameters x0.9: RHS = ((1-l)/l)(qdd + M^-1(C+g))", ok)
Cs = sp.Matrix([sum(sp.diff(M3[i,j], th[kk])*dth[j]*dth[kk] for j in range(3) for kk in range(3)) - sp.Rational(1,2)*sum(sp.diff(M3[j,kk], th[i])*dth[j]*dth[kk] for j in range(3) for kk in range(3)) for i in range(3)])
s2_, s3_, s23_ = sp.sin(th[1]), sp.sin(th[2]), sp.sin(th[1]+th[2]); A_, B_, D_ = 18, 3, 2
d1_, d2_, d3_ = dth
C1f = -A_*s2_*(2*d1_*d2_+d2_**2) - B_*s23_*(2*d1_*(d2_+d3_)+(d2_+d3_)**2) - D_*s3_*(2*(d1_+d2_)*d3_+d3_**2)
C2f = (A_*s2_+B_*s23_)*d1_**2 - D_*s3_*(2*(d1_+d2_)*d3_+d3_**2)
C3f = B_*s23_*d1_**2 + D_*s3_*(d1_+d2_)**2
chk("3-link C closed form (Christoffel)", sp.simplify(Cs - sp.Matrix([C1f, C2f, C3f])) == sp.zeros(3,1))
chk("A,B,D = (m2+m3)L1L2, m3L1L3, m3L2L3 = 18,3,2", (3*3*2, 1*3*1, 1*2*1) == (18, 3, 2))

print("=== 12 EQ4 critical damping of Option 2 loop")
close("Q9.4: wn^2 = kp + ke/m = 264, critical kv", [64+300/1.5, 2*np.sqrt(264)], [264, 32.496], 1e-3)
close("Q9.4 ratio E2/E1", 6.0606/25, 0.2424, 1e-3)
close("mass-error EQ2 cos20", np.cos(20*deg), 0.9397, 1e-4)
print("FAILS:", fails)
raise SystemExit(1 if fails else 0)
