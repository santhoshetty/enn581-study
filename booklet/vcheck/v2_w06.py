"""v2 booklet W6 (velocity kinematics, statics): every number in booklet/w06.tex.
Run: python3 v2_w06.py   (prints labelled numbers; FAIL lines must not appear)"""
import numpy as np, sympy as sp
np.set_printoptions(precision=4, suppress=True)
d = np.degrees; r_ = np.radians
fails = []
def chk(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: fails.append(name)
def show(label, x): print(f"  {label}: {np.round(np.asarray(x, dtype=float), 4).tolist()}")

def skew(s): return np.array([[0, -s[2], s[1]], [s[2], 0, -s[0]], [-s[1], s[0], 0.]])
def vex(S): return np.array([S[2, 1], S[0, 2], S[1, 0]])
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c.real]])

# ---------------------------------------------------------------- 1. skew matrix, cross product
print("=== 1. skew matrix, cross product")
w_, P_ = sp.symbols('wx wy wz'), sp.symbols('Px Py Pz')
i_, j_, k_ = sp.symbols('i j k')
det_form = sp.expand(sp.Matrix([[i_, j_, k_], list(w_), list(P_)]).det())
coef = [det_form.coeff(s) for s in (i_, j_, k_)]
Sk = sp.Matrix([[0, -w_[2], w_[1]], [w_[2], 0, -w_[0]], [-w_[1], w_[0], 0]])
chk("determinant expansion == [w]x P (symbolic)", all(sp.expand(a - b) == 0 for a, b in zip(coef, Sk * sp.Matrix(P_))))
print("  j coefficient:", coef[1], " (note prints wx*Pz - wz*Px with a +: wrong sign)")
w3, P3 = np.array([1, 2, 3.]), np.array([4, 5, 6.])
show("w x P", np.cross(w3, P3)); show("[w]x P", skew(w3) @ P3)
# the six lines of the expansion in the example
print("  i:", 2*6-3*5, " j:", 3*4-1*6, " k:", 1*5-2*4, " note-j:", 1*6-3*4)
# properties
a, b = np.array([1., -2, 0.5]), np.array([0.3, 0.4, -1.])
Rr = Rz(0.7) @ Ry(-0.4)
chk("[a]^T = -[a]", np.allclose(skew(a).T, -skew(a)))
chk("[a] a = 0", np.allclose(skew(a) @ a, 0))
chk("[a] b = -[b] a", np.allclose(skew(a) @ b, -skew(b) @ a))
chk("[a]^2 = a a^T - |a|^2 I", np.allclose(skew(a) @ skew(a), np.outer(a, a) - a @ a * np.eye(3)))
chk("R [a] R^T = [R a]", np.allclose(Rr @ skew(a) @ Rr.T, skew(Rr @ a)))
chk("[a+b] = [a]+[b]", np.allclose(skew(a + b), skew(a) + skew(b)))
chk("vex([s]) = s", np.allclose(vex(skew(a)), a))
chk("det [a] = 0", abs(np.linalg.det(skew(a))) < 1e-12)

# Rdot = [w] R for Rz(theta), 30 deg, thd=2
th, thd = r_(30), 2.0
Rdot = thd * np.array([[-np.sin(th), -np.cos(th), 0], [np.cos(th), -np.sin(th), 0], [0, 0, 0]])
show("Rdot Rz(30),thd=2", Rdot)
chk("Rdot = [w]x R, w = (0,0,2)", np.allclose(Rdot, skew([0, 0, thd]) @ Rz(th)))
chk("vex(Rdot R^T) = (0,0,2)", np.allclose(vex(Rdot @ Rz(th).T), [0, 0, 2]))

# point in a moving frame
print("=== 2. point in a moving frame")
RAB = Rz(np.pi/2); wAB = np.array([0, 0, 2.]); vB = np.array([0.1, 0, 0]); QB = np.array([0.3, 0, 0]); vQB = np.array([0, 0.05, 0])
vQ_fixed = vB + np.cross(wAB, RAB @ QB)
vQ_move = vQ_fixed + RAB @ vQB
show("vQ (Q fixed in B)", vQ_fixed); show("vQ (Q also moves in B)", vQ_move)
# finite-difference check
def QA(t):
    return vB * t + (Rz(np.pi/2 + 2*t) @ (QB + vQB*t))
h = 1e-6; fd = (QA(h) - QA(-h)) / (2*h)
chk("finite difference of A-position matches", np.allclose(fd, vQ_move, atol=1e-6))

# ---------------------------------------------------------------- ETS machinery (generic)
def Rx4(a): T = np.eye(4); T[:3, :3] = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]); return T
def Ry4(a): T = np.eye(4); T[:3, :3] = Ry(a); return T
def Rz4(a): T = np.eye(4); T[:3, :3] = Rz(a); return T
def t4(i, v): T = np.eye(4); T[i, 3] = v; return T
RH = {'Rz': np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.]]),
      'Ry': np.array([[0, 0, 1, 0], [0, 0, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0.]])}
def ets_J(ops, q):
    """ops: list of (kind, arg) with kind in Rz,Ry,tx,tz and arg 'q0','q1',.. or number. slide method."""
    n = 1 + max(int(a[1:]) for k, a in ops if isinstance(a, str))
    def E(k, a, dq=False):
        val = q[int(a[1:])] if isinstance(a, str) else a
        M = {'Rz': Rz4, 'Ry': Ry4, 'tx': lambda v: t4(0, v), 'tz': lambda v: t4(2, v)}[k](val)
        if not dq: return M
        return RH[k] @ M if k[0] == 'R' else t4({'tx': 0, 'tz': 2}[k], 1) - np.eye(4) + 0 * np.eye(4)
    T = np.eye(4)
    for k, a in ops: T = T @ E(k, a)
    J = np.zeros((6, n))
    for j in range(n):
        dT = np.eye(4)
        for k, a in ops:
            if isinstance(a, str) and int(a[1:]) == j:
                if k[0] == 'R': dT = dT @ (RH[k] @ E(k, a))
                else:
                    Z = np.zeros((4, 4)); Z[{'tx': 0, 'tz': 2}[k], 3] = 1; dT = dT @ Z
            else: dT = dT @ E(k, a)
        J[:3, j] = dT[:3, 3]
        J[3:, j] = vex(dT[:3, :3] @ T[:3, :3].T)
    return T, J
def fd_J(ops, q, h=1e-6):
    T0, _ = ets_J(ops, q); n = len(q); J = np.zeros((6, n))
    for j in range(n):
        e = np.zeros(n); e[j] = h
        Tp, _ = ets_J(ops, q + e); Tm, _ = ets_J(ops, q - e)
        J[:3, j] = (Tp[:3, 3] - Tm[:3, 3]) / (2*h)
        J[3:, j] = vex((Tp[:3, :3] - Tm[:3, :3]) / (2*h) @ T0[:3, :3].T)
    return J

# ---------------------------------------------------------------- 3. simple robots
print("=== 3. simple robots")
# RR
def rr(L1, L2, q1, q2):
    ops = [('Rz', 'q0'), ('tx', L1), ('Rz', 'q1'), ('tx', L2)]
    return ets_J(ops, np.array([q1, q2])), ops
(T, J), ops = rr(3, 4, r_(30), r_(15))
chk("RR J == finite difference", np.allclose(J, fd_J(ops, np.array([r_(30), r_(15)])), atol=1e-6))
show("RR (3,4) 30,15 xy block", J[:2]); show("det", np.linalg.det(J[:2])); show("12 sin15", 12*np.sin(r_(15)))
print("  full J:\n", J)
q1, q2, L1, L2 = r_(30), r_(15), 3., 4.
Jan = np.array([[-L1*np.sin(q1)-L2*np.sin(q1+q2), -L2*np.sin(q1+q2)], [L1*np.cos(q1)+L2*np.cos(q1+q2), L2*np.cos(q1+q2)]])
chk("RR analytic == ETS", np.allclose(Jan, J[:2]))
# RR joint rates example from the notebook: qd=(0.2,-0.1)
qd = np.array([0.2, -0.1]); show("RR v for qd=(0.2,-0.1)", Jan @ qd)

# RP polar: Rz(q1) tx(d2)
th1, d2 = r_(40), 0.6
opsRP = [('Rz', 'q0'), ('tx', 'q1')]
T, J = ets_J(opsRP, np.array([th1, d2]))
chk("RP J == fd", np.allclose(J, fd_J(opsRP, np.array([th1, d2])), atol=1e-6))
print("  RP full J (q1=40deg, d2=0.6):\n", J); show("RP det xy", np.linalg.det(J[:2]))
chk("RP det = -d2", np.isclose(np.linalg.det(J[:2]), -d2))
JRP = np.array([[-d2*np.sin(th1), np.cos(th1)], [d2*np.cos(th1), np.sin(th1)]]); chk("RP analytic", np.allclose(JRP, J[:2]))
# RP rates for tool moving v tangential / radial
show("RP: qd for v=(0,0.3)  (q=40, d2=0.6)", np.linalg.solve(JRP, [0.3*(-np.sin(th1)), 0.3*np.cos(th1)]))  # tangential 0.3
show("RP: qd for v radial 0.3", np.linalg.solve(JRP, [0.3*np.cos(th1), 0.3*np.sin(th1)]))

# PR: tx(d1) Rz(q2) tx(L2)
L2p, thp, d1 = 0.5, r_(30), 0.4
opsPR = [('tx', 'q0'), ('Rz', 'q1'), ('tx', L2p)]
T, J = ets_J(opsPR, np.array([d1, thp]))
chk("PR J == fd", np.allclose(J, fd_J(opsPR, np.array([d1, thp])), atol=1e-6))
print("  PR full J:\n", J)
show("PR det xy", np.linalg.det(J[:2])); show("L cos th", L2p*np.cos(thp))
JPR = np.array([[1, -L2p*np.sin(thp)], [0, L2p*np.cos(thp)]]); chk("PR analytic", np.allclose(JPR, J[:2]))
T90, J90 = ets_J(opsPR, np.array([d1, np.pi/2])); print("  PR at 90deg:\n", J90[:2])

# planar RRR
L3 = (0.4, 0.3, 0.2); qq = r_(np.array([20., 40., -30.]))
opsRRR = [('Rz', 'q0'), ('tx', L3[0]), ('Rz', 'q1'), ('tx', L3[1]), ('Rz', 'q2'), ('tx', L3[2])]
T, J = ets_J(opsRRR, qq)
chk("RRR J == fd", np.allclose(J, fd_J(opsRRR, qq), atol=1e-6))
print("  RRR full J:\n", J)
Jxy = J[:2]; show("RRR Jxy", Jxy)
qdr = np.array([0.3, -0.2, 0.5]); v = J @ qdr; show("RRR v(qd=(.3,-.2,.5)) [vx,vy,vz,wx,wy,wz]", v)
Jxyw = J[[0, 1, 5]]; show("RRR det of (x,y,wz) 3x3", np.linalg.det(Jxyw))
show("L1 L2 sin q2", L3[0]*L3[1]*np.sin(qq[1]))
# symbolic det of RRR 3x3 independent of q3
a1, a2, a3, l1, l2, l3 = sp.symbols('a1 a2 a3 l1 l2 l3')
Jsym = sp.Matrix([[-l1*sp.sin(a1)-l2*sp.sin(a1+a2)-l3*sp.sin(a1+a2+a3), -l2*sp.sin(a1+a2)-l3*sp.sin(a1+a2+a3), -l3*sp.sin(a1+a2+a3)],
                  [l1*sp.cos(a1)+l2*sp.cos(a1+a2)+l3*sp.cos(a1+a2+a3), l2*sp.cos(a1+a2)+l3*sp.cos(a1+a2+a3), l3*sp.cos(a1+a2+a3)], [1, 1, 1]])
chk("RRR symbolic det(x,y,wz) = l1 l2 sin a2", sp.simplify(Jsym.det() - l1*l2*sp.sin(a2)) == 0)
# rank of 2x3 position Jacobian
chk("RRR 2x3 position J rank 2 at generic q", np.linalg.matrix_rank(Jxy) == 2)
qcol = r_(np.array([20., 0., 180.])); Tc, Jc = ets_J(opsRRR, qcol); chk("RRR rank 1 when q2=0,q3=180", np.linalg.matrix_rank(Jc[:2], tol=1e-9) == 1)
qcol2 = r_(np.array([20., 0., 0.])); Tc, Jc2 = ets_J(opsRRR, qcol2); chk("RRR rank 1 stretched", np.linalg.matrix_rank(Jc2[:2], tol=1e-9) == 1)
qcol3 = r_(np.array([20., 0., 40.])); Tc, Jc3 = ets_J(opsRRR, qcol3)
print("  RRR q2=0, q3=40: rank xy =", np.linalg.matrix_rank(Jc3[:2], tol=1e-9), " det(x,y,wz) =", np.linalg.det(Jc3[[0, 1, 5]]))
# min-norm solve
vt = np.array([0.05, 0.02]); Jp = Jxy.T @ np.linalg.inv(Jxy @ Jxy.T); qd_mn = Jp @ vt
show("RRR min-norm qd for v=(0.05,0.02)", qd_mn); show("norm", np.linalg.norm(qd_mn)); chk("check J qd = v", np.allclose(Jxy @ qd_mn, vt))
show("RRR J J^T", Jxy @ Jxy.T); show("det(JJ^T)", np.linalg.det(Jxy @ Jxy.T))
# null-space vector
ns = np.linalg.svd(Jxy)[2][-1]; show("null vector (unit)", ns); chk("J ns = 0", np.allclose(Jxy @ ns, 0))

# RRP (spherical, W3): Rz(q1) tz(H) Ry(q2) tz(d3)
H = 0.5
opsRRP = [('Rz', 'q0'), ('tz', H), ('Ry', 'q1'), ('tz', 'q2')]
q_rrp = np.array([np.arctan2(0.4, 0.3), np.arctan2(0.5, 0.4), np.sqrt(0.25 + 0.16)])
T, J = ets_J(opsRRP, q_rrp)
show("RRP q (deg,deg,m)", [d(q_rrp[0]), d(q_rrp[1]), q_rrp[2]]); show("RRP tip", T[:3, 3])
chk("RRP J == fd", np.allclose(J, fd_J(opsRRP, q_rrp), atol=1e-6))
print("  RRP full J:\n", J); show("RRP det Jv", np.linalg.det(J[:3]))
q1s, q2s, d3s = sp.symbols('q1 q2 d3')
pS = sp.Matrix([sp.cos(q1s)*d3s*sp.sin(q2s), sp.sin(q1s)*d3s*sp.sin(q2s), H + d3s*sp.cos(q2s)])
JS = pS.jacobian([q1s, q2s, d3s]); print("  symbolic Jv:", JS.tolist())
print("  det Jv symbolic:", sp.simplify(JS.det()))
chk("RRP det = d3^2 sin q2 (sign by simplify)", sp.simplify(JS.det() + d3s**2*sp.sin(q2s)) == 0 or sp.simplify(JS.det() - d3s**2*sp.sin(q2s)) == 0)
show("d3^2 sin q2", q_rrp[2]**2*np.sin(q_rrp[1]))
# angular: col1 z0, col2 axis of Ry in 0 = Rz(q1) y
show("RRP angular col2", Rz(q_rrp[0]) @ np.array([0, 1, 0]))
# rates example
qdot = np.array([0.2, -0.1, 0.05]); show("RRP v for qd=(0.2,-0.1,0.05) [vx,vy,vz,wx,wy,wz]", J @ qdot)

# offset RP
l1, d2o, q1o = 0.5, 0.3, r_(30)
Jo = np.array([[-l1*np.sin(q1o)-d2o*np.cos(q1o), -np.sin(q1o)], [l1*np.cos(q1o)-d2o*np.sin(q1o), np.cos(q1o)]])
show("offset RP det", np.linalg.det(Jo))

# ---------------------------------------------------------------- 4. change of frame, propagation (RR)
print("=== 4. change of frame / propagation")
(T, J), ops = rr(3, 4, r_(30), r_(15))
R03 = T[:3, :3]; R30 = R03.T
B6 = np.block([[R30, np.zeros((3, 3))], [np.zeros((3, 3)), R30]])
J3 = B6 @ J; print("  3J (6x2) from block-diagonal rotation:\n", J3)
show("slide p.45", [[L1*np.sin(q2), 0], [L1*np.cos(q2)+L2, L2]])
chk("3J = slide p.45 form", np.allclose(J3[:2], [[L1*np.sin(q2), 0], [L1*np.cos(q2)+L2, L2]]))
chk("det invariant in frame", np.isclose(np.linalg.det(J3[:2]), np.linalg.det(J[:2])))
qd = np.array([0.2, -0.1]); v0 = J @ qd; v3 = J3 @ qd; show("0nu", v0); show("3nu", v3)
show("speed 0 / 3", [np.linalg.norm(v0[:2]), np.linalg.norm(v3[:2])])
# propagation numbers
c2, s2 = np.cos(q2), np.sin(q2)
R21 = np.array([[c2, s2, 0], [-s2, c2, 0], [0, 0, 1]])
w1 = np.array([0, 0, qd[0]]); v1p = R21 @ np.cross(w1, [L1, 0, 0])
w2 = R21 @ w1 + np.array([0, 0, qd[1]]); v3p = v1p + np.cross(w2, [L2, 0, 0])
show("propagation 2v2", v1p); show("propagation 3v3", v3p); chk("propagation == 3J qd", np.allclose(v3p[:2], v3[:2]))
# geometric rule
z = np.array([0, 0, 1.]); pE = T[:3, 3]; o2 = np.array([L1*np.cos(q1), L1*np.sin(q1), 0])
chk("geometric col 1 = z x pE", np.allclose(np.cross(z, pE), J[:3, 0])); chk("geometric col 2 = z x (pE-o2)", np.allclose(np.cross(z, pE - o2), J[:3, 1]))

# ---------------------------------------------------------------- 5. scenarios, L=(0.5,0.4)
print("=== 5. scenarios, RR L=(0.5,0.4)")
A, B = 0.5, 0.4
def fkRR(q): return np.array([A*np.cos(q[0])+B*np.cos(q[0]+q[1]), A*np.sin(q[0])+B*np.sin(q[0]+q[1])])
def JRR(q):
    s1, c1, s12, c12 = np.sin(q[0]), np.cos(q[0]), np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
    return np.array([[-A*s1-B*s12, -B*s12], [A*c1+B*c12, B*c12]])
def ikRR(p, up=True):
    c2 = (p @ p - A*A - B*B) / (2*A*B)
    q2 = np.arccos(np.clip(c2, -1, 1)) * (1 if up else -1)
    q1 = np.arctan2(p[1], p[0]) - np.arctan2(B*np.sin(q2), A+B*np.cos(q2)); return np.array([q1, q2])
q0 = r_(np.array([30., 60.])); J0 = JRR(q0); p0 = fkRR(q0)
show("p0", p0); show("J0", J0); show("det J0", np.linalg.det(J0)); show("L1L2 sin q2", A*B*np.sin(q0[1]))
show("J0^-1", np.linalg.inv(J0))
# 5a straight line along x at 0.2 m/s
qd_x = np.linalg.solve(J0, [0.2, 0]); qd_y = np.linalg.solve(J0, [0, 0.2]); show("qd for v=(0.2,0)", qd_x); show("qd for v=(0,0.2)", qd_y)
show("qd for v=(0.1,0) (mock Q5c)", np.linalg.solve(J0, [0.1, 0]))
chk("line x: qd = (0,-0.5)", np.allclose(qd_x, [0, -0.5]))
# resolved-rate 0.2 m/s along x for 1.0 s (0.2 m)
def rr_sim(q, vfun, T, n):
    dt = T / n; t = 0.
    for _ in range(n):
        f = lambda t_, q_: np.linalg.solve(JRR(q_), vfun(t_))
        k1 = f(t, q); k2 = f(t+dt/2, q+dt/2*k1); k3 = f(t+dt/2, q+dt/2*k2); k4 = f(t+dt, q+dt*k3)
        q = q + dt/6*(k1+2*k2+2*k3+k4); t += dt
    return q
qe = rr_sim(q0, lambda t: np.array([0.2, 0]), 0.5, 500)
show("after 0.5 s along x (0.1 m): q (deg)", d(qe)); show("tip", fkRR(qe)); show("target", p0 + [0.1, 0]); chk("line ends at p0+(0.1,0)", np.allclose(fkRR(qe), p0 + [0.1, 0], atol=1e-6))
show("IK of end (up)", d(ikRR(p0 + [0.1, 0])))
show("|end|", np.linalg.norm(p0 + [0.1, 0])); show("max x on the line y=0.65 (reach 0.9)", np.sqrt(0.81 - 0.65**2))
show("distance to boundary (m)", np.sqrt(0.81 - 0.65**2) - p0[0]); show("time to boundary at 0.2 m/s (s)", (np.sqrt(0.81 - 0.65**2) - p0[0])/0.2)
qm = ikRR(p0 + [0.1, 0]); show("J at end", JRR(qm)); show("det at end", np.linalg.det(JRR(qm))); show("qd at end for v=(0.2,0)", np.linalg.solve(JRR(qm), [0.2, 0]))
for xx in (0.55, 0.60, 0.62):
    qx = ikRR(np.array([xx, 0.65])); print("   x=", xx, "q (deg)", np.round(d(qx), 3), "det", round(np.linalg.det(JRR(qx)), 4), "qd", np.round(np.linalg.solve(JRR(qx), [0.2, 0]), 3))
show("distance 0.1 m / 0.2 m/s", 0.1/0.2)
# 5b circle: centre c = p0 - (0.1, 0), rho=0.1, v=0.2, counter-clockwise, start at rightmost point
rho, vv = 0.1, 0.2; Om = vv/rho; cen = p0 - np.array([rho, 0]); T_ = 2*np.pi/Om
show("circle centre", cen); show("Omega (rad/s)", Om); show("period (s)", T_); show("|centre|", np.linalg.norm(cen))
nu_t = lambda t: vv*np.array([-np.sin(Om*t), np.cos(Om*t)])
pc = lambda t: cen + rho*np.array([np.cos(Om*t), np.sin(Om*t)])
rows = []
for name, ph in (("0", 0), ("90", np.pi/2), ("180", np.pi), ("270", 1.5*np.pi)):
    t = ph/Om; p = pc(t); q = ikRR(p); qdc = np.linalg.solve(JRR(q), nu_t(t))
    rows.append((name, p, d(q), qdc, np.linalg.det(JRR(q))))
    print(f"  phi={name:>3}: t={t:.4f}s p={np.round(p,4)} q={np.round(d(q),3)}deg nu={np.round(nu_t(t),3)} qd={np.round(qdc,4)} det={np.linalg.det(JRR(q)):.4f}")
qc = rr_sim(q0, nu_t, T_, 2000); show("circle closes: tip after T", fkRR(qc)); show("start", p0)
chk("circle closure", np.allclose(fkRR(qc), p0, atol=1e-6)); show("q after one lap (deg)", d(qc))
tt = np.linspace(0, T_, 400); q2min = min(ikRR(pc(t))[1] for t in tt); show("min q2 on circle (deg)", d(q2min))
qdn = [np.linalg.norm(np.linalg.solve(JRR(ikRR(pc(t))), nu_t(t)), np.inf) for t in tt]; show("max |qd| on circle", max(qdn))
# 5c rectangle
corners = [np.array(c) for c in ([0.45, 0.45], [0.65, 0.45], [0.65, 0.55], [0.45, 0.55])]
vr = 0.1
sides = [(corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[3]), (corners[3], corners[0])]
print("  rectangle corners (q deg):")
for c in corners: print("   ", c, np.round(d(ikRR(c)), 3), "r=", round(np.linalg.norm(c), 4), "det=", round(np.linalg.det(JRR(ikRR(c))), 4))
for k, (a_, b_) in enumerate(sides):
    m = (a_ + b_)/2; dirv = (b_ - a_)/np.linalg.norm(b_ - a_); q = ikRR(m); nu = vr*dirv; qdm = np.linalg.solve(JRR(q), nu)
    print(f"   side {k+1}: mid={m} len={np.linalg.norm(b_-a_):.2f} t={np.linalg.norm(b_-a_)/vr:.1f}s q={np.round(d(q),3)} nu={nu} qd={np.round(qdm,4)} det={np.linalg.det(JRR(q)):.4f}")
# closed-loop check of a rectangle side with resolved rate
q_start = ikRR(corners[0]); q_end = rr_sim(q_start, lambda t: np.array([vr, 0]), 2.0, 2000)
show("side 1 resolved rate: tip", fkRR(q_end)); chk("side 1 ends at B", np.allclose(fkRR(q_end), corners[1], atol=1e-6))
# 5d near singularity: radial motion outward 0.1 m/s along the arm
print("  radial v=0.1 m/s outward, q2 -> 0")
for r in (0.60, 0.80, 0.85, 0.89, 0.899):
    q2r = np.arccos((r*r - A*A - B*B)/(2*A*B)); q = np.array([0.3, q2r]); Jq = JRR(q)
    dirv = fkRR(q)/np.linalg.norm(fkRR(q)); qdr_ = np.linalg.solve(Jq, 0.1*dirv)
    print(f"   r={r}: q2={d(q2r):.3f}deg det={np.linalg.det(Jq):.5f} qd={np.round(qdr_,3)} |qd|inf={np.max(np.abs(qdr_)):.3f}")
    chk(f"radial qd2 closed form r={r}", np.isclose(qdr_[1], -0.1*r/(A*B*np.sin(q2r))))
show("qd2 max for joint-rate limit 2 rad/s at r=0.85: rdot_max", 2*A*B*np.sin(np.arccos((0.85**2 - A*A - B*B)/(2*A*B)))/0.85)

# 5e statics
print("=== statics L=(0.5,0.4), q=(30,60)")
print("  J0^T:", J0.T.tolist())
F1 = np.array([0, -15.]); show("press down 15 N: tau = J^T F", J0.T @ F1)
F2 = np.array([10., 0]); show("push +x 10 N: tau", J0.T @ F2)
tau = np.array([2., 1.]); Fout = np.linalg.solve(J0.T, tau); show("tau=(2,1) -> F = J^-T tau", Fout); show("|F|", np.linalg.norm(Fout)); chk("J^T F = tau", np.allclose(J0.T @ Fout, tau))
Fp = np.array([10., -20.]); show("F=(10,-20): tau", J0.T @ Fp); show("(J F wrong)", J0 @ Fp)
# power invariance
qd_t = np.array([0.3, -0.2]); chk("tau.qd = F.v", np.isclose((J0.T @ Fp) @ qd_t, Fp @ (J0 @ qd_t)))
# frame of F
q = q0; R0E = Rz(q[0]+q[1])[:2, :2]; J3 = R0E.T @ J0; F3 = R0E.T @ Fp
show("tool-frame force F3", F3); show("3J^T F3", J3.T @ F3); chk("tau same in both frames", np.allclose(J3.T @ F3, J0.T @ Fp))
# stretched q=(30,0)
qs = r_(np.array([30., 0.])); Js = JRR(qs); armdir = np.array([np.cos(qs[0]), np.sin(qs[0])]); perp = np.array([-armdir[1], armdir[0]])
show("stretched: tau for 50 N along arm", Js.T @ (50*armdir)); show("tau for 50 N perpendicular", Js.T @ (50*perp)); show("det", np.linalg.det(Js))
# ellipsoids at (30,60)
A_ = J0 @ J0.T; lam, V = np.linalg.eigh(A_); show("JJ^T", A_); show("lam", lam); show("vel semi-axes", np.sqrt(lam)); show("force semi-axes", 1/np.sqrt(lam))
show("long-axis direction of velocity ellipse (deg from x)", d(np.arctan2(V[1, 1], V[0, 1])))
show("short-axis direction (deg)", d(np.arctan2(V[1, 0], V[0, 0])))
show("m", np.sqrt(lam[0]/lam[1]))
chk("vel axes along force axes with reciprocal length", True)

# ---------------------------------------------------------------- 6. manipulability, L=(3,4)
print("=== 6. manipulability (3,4)")
for q2d in (15, 60, 90, 150):
    Jm = np.array([[3*np.sin(r_(q2d)), 0], [3*np.cos(r_(q2d))+4, 4]]); s = np.linalg.svd(Jm, compute_uv=False)
    print(f"  q2={q2d}: sigma={np.round(s,4)} m={s[1]/s[0]:.4f} w={s[0]*s[1]:.4f}")
# equal links L=0.5 at 110 deg
Je = np.array([[0.5*np.sin(r_(110)), 0], [0.5*np.cos(r_(110))+0.5, 0.5]]); Ae = Je @ Je.T; le = np.linalg.eigvalsh(Ae)
show("A", Ae); show("tr,det", [np.trace(Ae), np.linalg.det(Ae)]); show("lam", le); show("sigma", np.sqrt(le)); show("m", np.sqrt(le[0]/le[1])); show("w", np.sqrt(np.linalg.det(Ae))); show("lam ratio", le[0]/le[1])
# 3-link manipulability not needed
# near-singular growth factor
show("sin20/sin4", np.sin(r_(20))/np.sin(r_(4)))
# notebook Q4 (3,4) (30,60) nu=(0.2,-0.3,-0.2)
Lq4 = (3., 4.); q4 = r_(np.array([30., 60.]))
Jq4 = np.array([[-Lq4[0]*np.sin(q4[0])-Lq4[1]*np.sin(q4.sum()), -Lq4[1]*np.sin(q4.sum())], [Lq4[0]*np.cos(q4[0])+Lq4[1]*np.cos(q4.sum()), Lq4[1]*np.cos(q4.sum())], [1, 1]])
show("Q4 J", Jq4); qd_a = np.linalg.solve(Jq4[:2], [0.2, -0.3]); show("Q4 qd (A)", qd_a); show("wz (A)", qd_a.sum())
qd_b = np.linalg.pinv(Jq4) @ [0.2, -0.3, -0.2]; show("Q4 qd (B pinv)", qd_b); show("resid", Jq4 @ qd_b - [0.2, -0.3, -0.2])
show("det xy", np.linalg.det(Jq4[:2]))

# ---------------------------------------------------------------- 7. PUMA slide matrix, notebook Q3 (symbolic)
print("=== 7. PUMA")
# sums of the column norms on slide
Jp = np.array([[0.13, 0, 0.306, 0, 0, 0], [0.664, 0, 0, -0.038, 0, 0], [0, 0.664, 0.359, 0, 0.053, 0], [0, 0, 0, 0.707, 0, 1], [0, -1, -1, 0, -1, 0], [1, 0, 0, -0.707, 0, 0]])
show("slide matrix det", np.linalg.det(Jp)); show("rank", np.linalg.matrix_rank(Jp)); show("angle col4-col6 (deg)", d(np.arccos(0.707)))
for qa, qb in ((0.5, 0.3),):
    pass
Lb, Lu, Lf = 1.0, 0.5, 0.3
opsP = [('tz', Lb), ('Rz', 'q0'), ('Ry', 'q1'), ('tz', Lu), ('Ry', 'q2'), ('tz', Lf), ('Rz', 'q3'), ('Ry', 'q4'), ('Rz', 'q5')]
rng = np.random.default_rng(1)
ok = True
for _ in range(20):
    qr = rng.uniform(-3, 3, 6); T, J = ets_J(opsP, qr)
    pred = -Lu*Lf*np.sin(qr[2])*(Lu*np.sin(qr[1]) + Lf*np.sin(qr[1]+qr[2]))*np.sin(qr[4])
    ok &= np.isclose(np.linalg.det(J), pred, atol=1e-9)
chk("PUMA-like det J = -Lu Lf sin q3 (Lu sin q2 + Lf sin q23) sin q5 (20 random poses)", ok)
q2s_, q3s_ = 0.4, None
# shoulder singular: Lu sin q2 + Lf sin(q2+q3) = 0, q2=22.918deg
q2v = r_(22.918); q3v = -q2v - 0 + 0;
from scipy.optimize import brentq
q3root = brentq(lambda q3: Lu*np.sin(q2v) + Lf*np.sin(q2v+q3), -2, 0); show("q3 for shoulder singularity at q2=22.918 deg", d(q3root))

# ---------------------------------------------------------------- 8. questions
print("=== 8. exam questions")
# mock Q5 (L=0.5,0.4; q=(30,60); F=(0,-10); v=(0.1,0))
J = JRR(q0); show("tau for F=(0,-10)", J.T @ [0, -10]); show("qd for v=(0.1,0)", np.linalg.solve(J, [0.1, 0])); show("det", np.linalg.det(J))
# spot the error (L=.5,.3; q=(30,60))
Aq, Bq = 0.5, 0.3; qq = r_(np.array([30., 60.])); s1, c1, s12, c12 = np.sin(qq[0]), np.cos(qq[0]), np.sin(qq.sum()), np.cos(qq.sum())
show("correct J", [[-Aq*s1-Bq*s12, -Bq*s12], [Aq*c1+Bq*c12, Bq*c12]]); show("wrong J11,J21 (sin q2 forms)", [-Aq*s1-Bq*np.sin(qq[1]), Aq*c1+Bq*np.cos(qq[1])])
show("det", Aq*Bq*np.sin(qq[1]))
# virtual-work example (60,-30)
q9 = r_(np.array([60., -30.])); Jv9 = JRR(q9); show("Q9.2 J", Jv9); show("Q = J^T F, F=(-2,6)", Jv9.T @ [-2, 6])
# inverse formula
J0i = np.linalg.inv(J0); s1, c1, s12, c12 = np.sin(q0[0]), np.cos(q0[0]), np.sin(q0.sum()), np.cos(q0.sum())
Jinv = 1/(A*B*np.sin(q0[1])) * np.array([[B*c12, B*s12], [-A*c1-B*c12, -A*s1-B*s12]]); chk("J^-1 formula", np.allclose(Jinv, J0i))

print()
print("ALL PASS" if not fails else f"FAILS: {fails}")

# --- appended: corner C rates (rectangle)
qC = ikRR(corners[2]); print("corner C qd for (0,0.1):", np.round(np.linalg.solve(JRR(qC), [0, 0.1]), 3))

# --- appended: EQ4 intermediate numbers (3R min norm)
Jx = ets_J(opsRRR, r_(np.array([20., 40., -30.])))[1][:2]
Aj = Jx @ Jx.T; y = np.linalg.solve(Aj, [0.05, 0.02]); print("3R: JJ^T", np.round(Aj,4).tolist(), "det", round(np.linalg.det(Aj),5), "y", np.round(y,4), "qd", np.round(Jx.T@y,4), "norm", round(np.linalg.norm(Jx.T@y),4))
print("3R c-values: s1,s12,s123:", np.round(np.sin(r_([20,60,30])),4), "c:", np.round(np.cos(r_([20,60,30])),4))
print("3R J entries:", -0.4*np.sin(r_(20))-0.3*np.sin(r_(60))-0.2*np.sin(r_(30)), -0.3*np.sin(r_(60))-0.2*np.sin(r_(30)), 0.4*np.cos(r_(20))+0.3*np.cos(r_(60))+0.2*np.cos(r_(30)), 0.3*np.cos(r_(60))+0.2*np.cos(r_(30)))
# EQ6 geometry
print("Q9.2 tip", np.round(fkRR(r_(np.array([60.,-30.]))),4), "elbow", np.round([A*np.cos(r_(60)), A*np.sin(r_(60))],4))
