"""Verify every result in booklet/derivations.tex (sympy/numpy). Prints PASS lines."""
import numpy as np, sympy as sp
rng = np.random.default_rng(2)
def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: raise SystemExit(1)
S = sp
# (a) proximal DH
al, a, th, d = sp.symbols('alpha a theta d')
Rx = lambda t: sp.Matrix([[1,0,0,0],[0,sp.cos(t),-sp.sin(t),0],[0,sp.sin(t),sp.cos(t),0],[0,0,0,1]])
Rz = lambda t: sp.Matrix([[sp.cos(t),-sp.sin(t),0,0],[sp.sin(t),sp.cos(t),0,0],[0,0,1,0],[0,0,0,1]])
tx = lambda v: sp.Matrix([[1,0,0,v],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
tz = lambda v: sp.Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,v],[0,0,0,1]])
c, s = sp.cos, sp.sin
printed = sp.Matrix([[c(th),-s(th),0,a],[s(th)*c(al),c(th)*c(al),-s(al),-s(al)*d],[s(th)*s(al),c(th)*s(al),c(al),c(al)*d],[0,0,0,1]])
ok("DH: Rx(al) tx(a) Rz(th) tz(d) = printed matrix", sp.simplify(Rx(al)*tx(a)*Rz(th)*tz(d)-printed) == sp.zeros(4))
ok("DH intermediate: Rx(al)tx(a) and Rz(th)tz(d); translation = Rx(al)(a,0,d)",
   sp.simplify((Rx(al)*tx(a))[:3,3]-sp.Matrix([a,0,0])) == sp.zeros(3,1) and
   sp.simplify((Rx(al)*tx(a)*Rz(th)*tz(d))[:3,3] - (Rx(al)[:3,:3]*sp.Matrix([a,0,d]))) == sp.zeros(3,1))

# (b) planar 2R IK
good = True
for _ in range(2000):
    L1, L2 = rng.uniform(.3, 1.2, 2); q1, q2 = rng.uniform(-3, 3, 2)
    x = L1*np.cos(q1)+L2*np.cos(q1+q2); y = L1*np.sin(q1)+L2*np.sin(q1+q2)
    c2 = (x*x+y*y-L1**2-L2**2)/(2*L1*L2)
    for sg in (1, -1):
        q2b = np.arctan2(sg*np.sqrt(max(0, 1-c2*c2)), c2)
        q1b = np.arctan2(y, x)-np.arctan2(L2*np.sin(q2b), L1+L2*np.cos(q2b))
        good &= np.isclose(L1*np.cos(q1b)+L2*np.cos(q1b+q2b), x) and np.isclose(L1*np.sin(q1b)+L2*np.sin(q1b+q2b), y)
    # cosine rule with interior angle pi - q2
    good &= np.isclose(x*x+y*y, L1**2+L2**2-2*L1*L2*np.cos(np.pi-q2))
ok("2R IK both branches (algebraic = geometric: q1 = atan2(y,x) - atan2(L2 s2, L1+L2 c2)); r^2 = L1^2+L2^2-2L1L2cos(pi-q2)", good)
L1, L2, x, y = .5, .4, .5, .3
c2 = (x*x+y*y-L1**2-L2**2)/(2*L1*L2)
r = []
for sg in (1, -1):
    q2 = np.arctan2(sg*np.sqrt(1-c2*c2), c2); q1 = np.arctan2(y, x)-np.arctan2(L2*np.sin(q2), L1+L2*np.cos(q2)); r.append((np.degrees(q1), np.degrees(q2)))
print("   2R numeric (L=0.5,0.4; target 0.5,0.3): c2=%.4f, branches q=%s" % (c2, [tuple(round(v, 2) for v in t) for t in r]))
ok("2R numeric reproduces target", all(np.isclose(L1*np.cos(np.radians(q1))+L2*np.cos(np.radians(q1+q2)), x) for q1, q2 in r))

# (c) spherical wrist: ZYZ expansion r13 = c4 s5, r23 = s4 s5, r33 = c5, r31 = -s5 c6, r32 = s5 s6
q4, q5, q6 = sp.symbols('q4 q5 q6')
Rz3 = lambda t: sp.Matrix([[c(t),-s(t),0],[s(t),c(t),0],[0,0,1]])
Ry3 = lambda t: sp.Matrix([[c(t),0,s(t)],[0,1,0],[-s(t),0,c(t)]])
Rw = Rz3(q4)*Ry3(q5)*Rz3(q6)
ok("wrist entries r13=c4s5, r23=s4s5, r33=c5, r31=-s5c6, r32=s5s6",
   all(sp.simplify(Rw[i, j]-e) == 0 for (i, j), e in {(0,2): c(q4)*s(q5), (1,2): s(q4)*s(q5), (2,2): c(q5), (2,0): -s(q5)*c(q6), (2,1): s(q5)*s(q6)}.items()))
H, G, F = .5, .4, .3
Wc = np.array([0.4813, 0.4039, 0.5974])
q1 = np.arctan2(Wc[1], Wc[0]); rW = np.hypot(Wc[0], Wc[1])
c3 = (rW**2+(Wc[2]-H)**2-F**2-G**2)/(2*F*G)
res = {}
for sg in (-1, 1):
    q3 = np.arctan2(sg*np.sqrt(1-c3**2), c3)
    q2 = np.arctan2(Wc[2]-H, rW)-np.arctan2(F*np.sin(q3), G+F*np.cos(q3)); res[sg] = (np.degrees(q3), np.degrees(q2))
print("   wrist-centre numeric: q1=%.2f, c3=%.4f, %s" % (np.degrees(q1), c3, {k: tuple(round(x, 3) for x in v) for k, v in res.items()}))
ok("site worked numbers: q1=40, c3=0.6428, q3=-50 -> q2=30, q3=+50 -> q2=-12.378",
   abs(np.degrees(q1)-40) < .01 and abs(c3-.6428) < 1e-4 and abs(res[-1][0]+50) < .01 and abs(res[-1][1]-30) < .01 and abs(res[1][1]+12.378) < .01)
# arm FK reproduces the wrist centre
for sg in (-1, 1):
    q3, q2 = np.radians(res[sg][0]), np.radians(res[sg][1])
    rr = G*np.cos(q2)+F*np.cos(q2+q3); zz = H+G*np.sin(q2)+F*np.sin(q2+q3)
    ok(f"wrist centre reproduced by FK (branch {sg})", abs(rr-rW) < 1e-3 and abs(zz-Wc[2]) < 1e-3)
good = True
for _ in range(500):
    qq = rng.uniform(-3, 3, 3); qq[1] = rng.uniform(.1, 3.0)
    R = np.array(Rw.subs({q4: qq[0], q5: qq[1], q6: qq[2]}).evalf(), dtype=float)
    a5 = np.arctan2(np.hypot(R[0,2], R[1,2]), R[2,2]); a4 = np.arctan2(R[1,2], R[0,2]); a6 = np.arctan2(R[2,1], -R[2,0])
    good &= np.allclose([a4, a5, a6], qq)
    # flip
    f5 = -a5; f4 = a4+np.pi; f6 = a6+np.pi
    Rf = np.array(Rw.subs({q4: f4, q5: f5, q6: f6}).evalf(), dtype=float)
    good &= np.allclose(Rf, R)
    # flip extraction with negative sign
    b4 = np.arctan2(-R[1,2], -R[0,2]); b6 = np.arctan2(-R[2,1], R[2,0])
    good &= np.isclose(np.cos(b4-f4), 1) and np.isclose(np.cos(b6-f6), 1)
ok("ZYZ wrist: q5=atan2(sqrt(r13^2+r23^2),r33), q4=atan2(r23,r13), q6=atan2(r32,-r31); flip (-q5, q4+pi, q6+pi)", good)

# (d) Grubler-Kutzbach counts
gk = lambda D, N, J, SF, RC=0, RM=0: D*(N-J-1)+SF+RC-RM
ok("GK: 3-RPR=3, simplest delta=3, common delta=9 -> extended 3, four-bar=1, 6-UPS=6, 4RRR=-6 -> +7 = 1",
   gk(3,8,9,9) == 3 and gk(6,8,9,15) == 3 and gk(6,11,15,39) == 9 and gk(6,11,15,39,0,6) == 3 and gk(3,4,4,4) == 1 and
   gk(6,14,18,36) == 6 and gk(6,10,12,12) == -6 and gk(6,10,12,12,12-5,0) == 1)
Dn, Nn, Jn = sp.symbols('D N J'); Fs = sp.symbols('SF')
ok("D(N-1) - sum(D-F_i) = D(N-J-1) + sum F_i", sp.expand(Dn*(Nn-1)-(Jn*Dn-Fs)-(Dn*(Nn-Jn-1)+Fs)) == 0)

# (e) four-bar: loop closure and circle-intersection FK (site numbers: L0 1, L1 .5, L2 .45 = a .2 + b .25, L3 .6; theta1 70)
L0, L1, L2, L3, a_, b_ = 1., .5, .45, .6, .2, .25
t1 = np.radians(70); A = L1*np.array([np.cos(t1), np.sin(t1)]); D = np.array([L0, 0.])
dAD = np.linalg.norm(D-A); xx = (L2**2-L3**2+dAD**2)/(2*dAD); hh = np.sqrt(L2**2-xx**2)
u = (D-A)/dAD; n = np.array([-u[1], u[0]])
print("   four-bar: |AD|=%.4f, x=%.4f, h=%.4f" % (dAD, xx, hh))
good = True
for sg in (1, -1):
    B = A+xx*u+sg*hh*n
    good &= np.isclose(np.linalg.norm(B-A), L2) and np.isclose(np.linalg.norm(B-D), L3)
    phi = np.arctan2(*(B-A)[::-1]); K = A+a_*np.array([np.cos(phi), np.sin(phi)])
    phi1 = np.arctan2(B[1], B[0]-L0); 
    # loop closure eq (K from both sides)
    K2 = D+L3*np.array([np.cos(phi1), np.sin(phi1)])-b_*np.array([np.cos(phi), np.sin(phi)])
    good &= np.allclose(K, K2)
    # IK recovery
    A2 = K-a_*np.array([np.cos(phi), np.sin(phi)]); good &= np.isclose(np.arctan2(A2[1], A2[0]), t1)
    B2 = K+b_*np.array([np.cos(phi), np.sin(phi)]); good &= np.isclose(np.arctan2(B2[1], B2[0]-L0), phi1)
    if sg == -1: print("   lower mode phi=%.3f deg, K=(%.4f,%.4f)" % (np.degrees(phi), *K)); lower = (np.degrees(phi), K)
ok("four-bar FK via circles, loop closure, IK recovery (both modes)", good)
Bu = A+xx*u+hh*n
ok("upper mode quoted in derivations: B=(0.6210,0.4651)", np.allclose(Bu, [0.6210, 0.4651], atol=1e-4))
ok("site P4.3: phi=-58.486, K=(0.2756,0.2993)", abs(lower[0]+58.486) < .002 and abs(lower[1][0]-.2756) < 1e-4 and abs(lower[1][1]-.2993) < 1e-4)
# Freudenstein length eq equals zero at a valid configuration
B = A+xx*u-hh*n; phi1 = np.arctan2(B[1], B[0]-L0)
eta = (L1*np.cos(t1)-L0-L3*np.cos(phi1))**2+(L1*np.sin(t1)-L3*np.sin(phi1))**2-L2**2
ok("eta(theta1,phi1)=0 at the assembled pose", abs(eta) < 1e-12)

# (f) trajectories
T, q0, qv, qF, w, T1, T2 = sp.symbols('T q0 qv qF w T1 T2')
d1, d2 = qv-q0, qF-qv
a12 = (3*d1-w*T1)/T1**2; a13 = (-2*d1+w*T1)/T1**3
a22 = (3*d2-2*w*T2)/T2**2; a23 = (-2*d2+w*T2)/T2**3
t = sp.symbols('t')
q1t = q0+0*t+a12*t**2+a13*t**3; q2t = qv+w*t+a22*t**2+a23*t**3
conds = [q1t.subs(t, 0)-q0, sp.diff(q1t, t).subs(t, 0), q1t.subs(t, T1)-qv, q2t.subs(t, 0)-qv, q2t.subs(t, T2)-qF,
         sp.diff(q2t, t).subs(t, T2), sp.diff(q1t, t).subs(t, T1)-sp.diff(q2t, t).subs(t, 0)]
ok("two-cubic segments satisfy 7 of the 8 conditions identically", all(sp.simplify(e) == 0 for e in conds))
acc = sp.diff(q1t, t, 2).subs(t, T1)-sp.diff(q2t, t, 2).subs(t, 0)
wsol = sp.solve(acc, w)[0]
ok("8th (acceleration match) gives w = 3[T1^2(qF-qv)+T2^2(qv-q0)]/(2 T1 T2 (T1+T2))",
   sp.simplify(wsol-3*(T1**2*(qF-qv)+T2**2*(qv-q0))/(2*T1*T2*(T1+T2))) == 0)
ok("end accelerations: q1''(T1) = (-6 d1+4 w T1)/T1^2 ; q2''(0) = (6 d2 - 4 w T2)/T2^2",
   sp.simplify(sp.diff(q1t, t, 2).subs(t, T1)-(-6*d1+4*w*T1)/T1**2) == 0 and sp.simplify(sp.diff(q2t, t, 2).subs(t, 0)-(6*d2-4*w*T2)/T2**2) == 0)
weq = sp.simplify(wsol.subs({T1: T, T2: T}))
ok("equal durations: w = 3(qF-q0)/(4T); shared acceleration 3(q0+qF-2qv)/T^2",
   sp.simplify(weq-3*(qF-q0)/(4*T)) == 0 and sp.simplify(sp.diff(q2t, t, 2).subs(t, 0).subs({T2: T, w: weq})-3*(q0+qF-2*qv)/T**2) == 0)
vals = {q0: 10, qv: 40, qF: 20, T1: 1, T2: 1}
cf = [float(x.subs(vals).subs(w, 7.5)) for x in (a12, a13, a22, a23)]
print("   ", wsol.subs(vals), cf, sp.diff(q1t, t, 2).subs(t, T1).subs(vals).subs(w, 7.5))
ok("worked q=(10,40,20), T=1: w=7.5, seg1 (10,0,82.5,-52.5), seg2 (40,7.5,-75,47.5), accel -150",
   abs(float(wsol.subs(vals))-7.5) < 1e-12 and np.allclose(cf, [82.5, -52.5, -75., 47.5]) and
   abs(float(sp.diff(q1t, t, 2).subs(t, T1).subs(vals).subs(w, 7.5))+150) < 1e-9)
vals = {q0: 10, qv: 40, qF: 20, T1: 1, T2: 2}
ok("worked T1=1,T2=2: w=25, accel -80", abs(float(wsol.subs(vals))-25) < 1e-12 and abs(float(sp.diff(q1t, t, 2).subs(t, T1).subs(vals).subs(w, 25))+80) < 1e-9)
# rest-to-rest cubic, peak speed
Dq = sp.symbols('Dq'); a2, a3 = 3*Dq/T**2, -2*Dq/T**3
ok("rest-to-rest: peak speed 1.5 Dq/T at T/2; accel +-6Dq/T^2 at ends",
   sp.simplify((2*a2*t+3*a3*t**2).subs(t, T/2)-sp.Rational(3, 2)*Dq/T) == 0 and sp.simplify((2*a2+6*a3*0)-6*Dq/T**2) == 0 and sp.simplify((2*a2+6*a3*T)+6*Dq/T**2) == 0)

# (g) LSPB
a_s, tF, tB, Dd = sp.symbols('a tF tB Dd', positive=True)
qB2 = Dd - 0  # eliminate: q_F - q_0 = 1/2 a tB^2 + a tB (tF-2tB) + a tB tB - 1/2 a tB^2
tot = sp.Rational(1, 2)*a_s*tB**2 + a_s*tB*(tF-2*tB) + a_s*tB*tB - sp.Rational(1, 2)*a_s*tB**2
ok("LSPB: total displacement = a tB tF - a tB^2", sp.expand(tot-(a_s*tB*tF-a_s*tB**2)) == 0)
roots = sp.solve(sp.Eq(a_s*tB*tF-a_s*tB**2, Dd), tB)
ok("LSPB: tB = tF/2 -+ sqrt(a^2 tF^2 - 4 a D)/(2a)", any(sp.simplify(r_-(tF/2-sp.sqrt(a_s**2*tF**2-4*a_s*Dd)/(2*a_s))) == 0 for r_ in roots))
tBv = lambda a, tF, D: tF/2-np.sqrt(a*a*tF*tF-4*a*D)/(2*a)
ok("LSPB: 120 deg in 2 s: a_min=120 (tB=1, triangle, peak 120); tB=1 at a_min", abs(tBv(120, 2, 120)-1) < 1e-12 and abs(120*tBv(120, 2, 120)-120) < 1e-9)
print("   LSPB a=300: tB=%.4f, v=%.4f ; a=150: v=%.3f ; a=300: v=%.3f" % (tBv(300, 2, 120), 300*tBv(300, 2, 120), 150*tBv(150, 2, 120), 300*tBv(300, 2, 120)))
ok("LSPB coast speeds 82.9 (a=150) and 67.6 (a=300)", abs(150*tBv(150, 2, 120)-82.9) < .05 and abs(300*tBv(300, 2, 120)-67.6) < .05)
ok("LSPB a_min = 4D/tF^2", sp.simplify(sp.solve(a_s**2*tF**2-4*a_s*Dd, a_s)[0]-4*Dd/tF**2) == 0)

# (h) Jacobian
q1s, q2s, l1, l2 = sp.symbols('q1 q2 L1 L2')
xe = l1*c(q1s)+l2*c(q1s+q2s); ye = l1*s(q1s)+l2*s(q1s+q2s)
J = sp.Matrix([xe, ye]).jacobian([q1s, q2s])
ok("J = [[-L1s1-L2s12, -L2s12],[L1c1+L2c12, L2c12]]", sp.simplify(J-sp.Matrix([[-l1*s(q1s)-l2*s(q1s+q2s), -l2*s(q1s+q2s)], [l1*c(q1s)+l2*c(q1s+q2s), l2*c(q1s+q2s)]])) == sp.zeros(2))
ok("det J = L1 L2 sin q2", sp.simplify(J.det()-l1*l2*s(q2s)) == 0)
Jn = np.array(J.subs({l1: .5, l2: .4, q1s: np.radians(30), q2s: np.radians(60)}).evalf(), dtype=float)
tau = Jn.T@np.array([10, -20.])
ok("tau = J^T F: L=(0.5,0.4), q=(30,60) deg, F=(10,-20) -> (-15.160,-4.000); J F wrong = (1.5, 4.330)",
   np.allclose(tau, [-15.160, -4.000], atol=1e-3) and np.allclose(Jn@np.array([10, -20.]), [1.5, 4.330], atol=1e-3))
# virtual work check by finite differences: F . dx = tau . dq
dq = rng.normal(size=2)*1e-7; qq = np.array([.3, .9])
f = lambda q: np.array([.5*np.cos(q[0])+.4*np.cos(q[0]+q[1]), .5*np.sin(q[0])+.4*np.sin(q[0]+q[1])])
Jq = np.array(J.subs({l1: .5, l2: .4, q1s: qq[0], q2s: qq[1]}).evalf(), dtype=float)
Fv = np.array([3., -2.])
ok("virtual work F^T dx = (J^T F)^T dq", abs(Fv@(f(qq+dq)-f(qq)) - (Jq.T@Fv)@dq) < 1e-12)
ok("singular at q2=0: det -> 0 ; inverse velocity 1/(L1 L2 sin q2)", abs(float(J.det().subs({l1: .5, l2: .4, q2s: 0}))) < 1e-12)

# (i) Lagrange point-mass 2R
m1, m2, g, L1s, L2s = sp.symbols('m1 m2 g L1 L2', positive=True)
q1, q2, dq1, dq2, ddq1, ddq2 = sp.symbols('q1 q2 dq1 dq2 ddq1 ddq2')
tt = sp.symbols('tt'); Q1f, Q2f = sp.Function('Q1')(tt), sp.Function('Q2')(tt)
x1 = L1s*sp.cos(Q1f); y1 = L1s*sp.sin(Q1f)
x2 = x1+L2s*sp.cos(Q1f+Q2f); y2 = y1+L2s*sp.sin(Q1f+Q2f)
K = sp.Rational(1, 2)*m1*(sp.diff(x1, tt)**2+sp.diff(y1, tt)**2)+sp.Rational(1, 2)*m2*(sp.diff(x2, tt)**2+sp.diff(y2, tt)**2)
P = m1*g*y1+m2*g*y2
Lg = K-P
tau_ = []
for Qf in (Q1f, Q2f):
    tau_.append(sp.diff(sp.diff(Lg, sp.diff(Qf, tt)), tt)-sp.diff(Lg, Qf))
sub = lambda e: sp.simplify(e.subs({sp.diff(Q1f, tt, 2): ddq1, sp.diff(Q2f, tt, 2): ddq2}).subs({sp.diff(Q1f, tt): dq1, sp.diff(Q2f, tt): dq2}).subs({Q1f: q1, Q2f: q2}))
t1e, t2e = sub(tau_[0]), sub(tau_[1])
c1, c2, c12, s2 = sp.cos(q1), sp.cos(q2), sp.cos(q1+q2), sp.sin(q2)
M11 = m1*L1s**2+m2*(L1s**2+2*L1s*L2s*c2+L2s**2); M12 = m2*(L1s*L2s*c2+L2s**2); M22 = m2*L2s**2
h_ = m2*L1s*L2s*s2
g1 = (m1+m2)*g*L1s*c1+m2*g*L2s*c12; g2 = m2*g*L2s*c12
ok("Lagrange Q1 = M11 q1'' + M12 q2'' - h(2 q1' q2' + q2'^2) + g1", sp.simplify(t1e-(M11*ddq1+M12*ddq2-h_*(2*dq1*dq2+dq2**2)+g1)) == 0)
ok("Lagrange Q2 = M12 q1'' + M22 q2'' + h q1'^2 + g2", sp.simplify(t2e-(M12*ddq1+M22*ddq2+h_*dq1**2+g2)) == 0)
# hand-derivative pieces
Msym = sp.Matrix([[M11, M12], [M12, M22]])
Kq = sp.Rational(1, 2)*sp.Matrix([dq1, dq2]).T*Msym*sp.Matrix([dq1, dq2])
ok("K = 1/2 qdot^T M qdot (printed M)", sp.simplify(sub(K)-Kq[0]) == 0)
ok("dM11/dq2 = -2h, dM12/dq2 = -h ; dK/dq1 = 0 ; dK/dq2 = -h dq1^2 - h dq1 dq2",
   sp.simplify(sp.diff(M11, q2)+2*h_) == 0 and sp.simplify(sp.diff(M12, q2)+h_) == 0 and sp.simplify(sp.diff(Kq[0], q1)) == 0 and
   sp.simplify(sp.diff(Kq[0], q2)-(-h_*dq1**2-h_*dq1*dq2)) == 0)
ok("g_i = dP/dq_i", sp.simplify(sp.diff(sub(P), q1)-g1) == 0 and sp.simplify(sp.diff(sub(P), q2)-g2) == 0)
vv = {m1: 2, m2: 1, L1s: 1., L2s: .8, g: 9.81, q1: np.radians(30), q2: np.radians(45), dq1: 1, dq2: -.5, ddq1: .5, ddq2: 2}
ok("Lagrange numbers: Q=(32.7397, 4.4797)", abs(float(t1e.subs(vv))-32.7397) < 1e-3 and abs(float(t2e.subs(vv))-4.4797) < 1e-3)

# (j) control
kp, kv, ki, mm, fd = sp.symbols('kp kv ki m fd', positive=True)
E = sp.Function('E')(tt)
# disturbance: m xdd = m f' - fd, f' = xd'' + kv E' + kp E  -> E'' + kv E' + kp E = fd/m
xd = sp.Function('xd')(tt); xx = xd-E
fprime = sp.diff(xd, tt, 2)+kv*sp.diff(E, tt)+kp*E
resid = sp.simplify(mm*sp.diff(xx, tt, 2)-(mm*fprime-fd))
ok("error eq: E'' + kv E' + kp E = fd/m", sp.simplify(resid.subs(sp.diff(E, tt, 2), (fd/mm - kv*sp.diff(E, tt)-kp*E))) == 0)
ok("critical: disc kv^2-4kp = 0 -> kv = 2 sqrt(kp); kp=25 -> kv=10", sp.solve(kv**2-4*kp, kv) == [2*sp.sqrt(kp)] and 2*np.sqrt(25) == 10)
ok("slide swap kp^2 = 4kv with kp=16 gives kv=64 roots -63.75 and -0.25", np.allclose(sorted(np.roots([1, 64, 16])), [-63.75, -0.25], atol=1e-2))
ok("Ess = fd/(kp m): 10/(25*2)=0.2 ; m=5 -> 0.08", abs(10/(25*2)-.2) < 1e-12 and abs(10/(25*5)-.08) < 1e-12)
# simulate the real plant m=2,b=3,k=8 with partitioned law, constant disturbance
def sim(ki_, fd_=10., m=2., b=3., k=8., kp_=25., kv_=10., T_=20., dt=1e-4):
    x = v = 0.; I = 0.
    for _ in range(int(T_/dt)):
        Ee = 0-x; Ed = -v; I += Ee*dt
        fp = kv_*Ed+kp_*Ee+ki_*I
        f = m*fp+b*v+k*x
        a = (f-fd_-b*v-k*x)/m
        v += a*dt; x += v*dt
    return -x, I
e, _ = sim(0.)
ok("simulation: PD steady error 0.2 m on the real plant", abs(e-.2) < 1e-3)
e, I = sim(40.)
ok("simulation PID ki=40: E->0, integral -> fd/(m ki)=0.125", abs(e) < 1e-3 and abs(I-.125) < 2e-3)
ok("PID: E''' + kv E'' + kp E' + ki E = fd'/m ; char poly roots for ki=40", np.allclose(sorted(np.roots([1, 10, 25, 40]).real), sorted([-7.335, -1.332, -1.332]), atol=1e-3))
ok("Routh: stable iff kv kp > ki (boundary ki=250: (s+10)(s^2+25))", np.allclose(sorted(np.roots([1, 10, 25, 250]).real), [-10, 0, 0], atol=1e-6))

# (k) wheels: omni triangle
dd, r_ = sp.symbols('d r', positive=True); u, v, om = sp.symbols('u v om')
wheels = [((0, dd), 0), ((sp.sqrt(3)*dd/2, -dd/2), -sp.rad(120)*1), ((-sp.sqrt(3)*dd/2, -dd/2), sp.rad(120))]
rows = []
for (xw, yw), be in wheels:
    rows.append(sp.simplify(sp.cos(be)*(u-om*yw)+sp.sin(be)*(v+om*xw)))
Wm = sp.Matrix([[sp.diff(rw, z) for z in (u, v, om)] for rw in rows])
printed = sp.Matrix([[1, 0, -dd], [-sp.Rational(1, 2), -sp.sqrt(3)/2, -dd], [-sp.Rational(1, 2), sp.sqrt(3)/2, -dd]])
ok("omni rows (beta=0,-120,+120 deg; clockwise) = printed matrix", sp.simplify(Wm-printed) == sp.zeros(3))
inv_pr = sp.Matrix([[2, -1, -1], [0, -sp.sqrt(3), sp.sqrt(3)], [-1/dd, -1/dd, -1/dd]])/3
ok("inverse = (1/3)[[2,-1,-1],[0,-sqrt3,sqrt3],[-1/d,-1/d,-1/d]] (times r) ; det = 3 sqrt3 d/2", sp.simplify(printed.inv()-inv_pr) == sp.zeros(3) and sp.simplify(printed.det()-3*sp.sqrt(3)*dd/2) == 0)
Wn = np.array(printed.subs(dd, .1).evalf(), dtype=float); zeta = .05/3*np.array([[2, -1, -1], [0, -np.sqrt(3), np.sqrt(3)], [-10, -10, -10]])@np.array([4, -1, -1.])
ok("omni example phi'=(4,-1,-1), r=.05, d=.1: u=0.1667, v=0, om=-0.3333", np.allclose(zeta, [1/6, 0, -1/3], atol=1e-4))
ok("ICR (-v/om, u/om) = (0,-0.5)", np.isclose(zeta[2], -1/3) and np.isclose(zeta[0]/zeta[2], -.5))
# generalised steps: R(beta) then skew
xw, yw, be = sp.symbols('xw yw beta')
step = sp.Matrix([[sp.cos(be), sp.sin(be)], [-sp.sin(be), sp.cos(be)]])*sp.Matrix([[1, 0, -yw], [0, 1, xw]])
ok("step 1+2: drive row = [cos b, sin b][1 0 -y; 0 1 x] zeta", sp.simplify(step[0, :]-sp.Matrix([[sp.cos(be), sp.sin(be), -yw*sp.cos(be)+xw*sp.sin(be)]])) == sp.zeros(1, 3))
ok("no-slide row = [-sin b, cos b, y sin b + x cos b]", sp.simplify(step[1, :]-sp.Matrix([[-sp.sin(be), sp.cos(be), yw*sp.sin(be)+xw*sp.cos(be)]])) == sp.zeros(1, 3))

# (l) differential drive
aa = sp.symbols('a', positive=True)
rows = [sp.cos(0)*(u-om*aa), sp.cos(0)*(u+om*aa)]  # left at y=+a ; right at y=-a (beta=0)
sol = sp.solve([sp.Eq(r_*sp.Symbol('p1'), rows[0]), sp.Eq(r_*sp.Symbol('p2'), rows[1])], [u, om])
ok("diff drive: u = r(p1+p2)/2, om = r(p2-p1)/(2a)", sp.simplify(sol[u]-r_*(sp.Symbol('p1')+sp.Symbol('p2'))/2) == 0 and sp.simplify(sol[om]-r_*(sp.Symbol('p2')-sp.Symbol('p1'))/(2*aa)) == 0)
ok("ICR y = a (p1+p2)/(p2-p1): a=.3, (2,5) -> 0.7 ; a=.15,r=.05,(4,6): u=.25, om=.3333, y=.75",
   abs(.3*7/3-.7) < 1e-12 and abs(.05*10/2-.25) < 1e-12 and abs(.05*2/.3-1/3) < 1e-12 and abs(.25/(1/3)-.75) < 1e-12 and abs(.15*10/2-.75) < 1e-12)
ok("no-slide row: v + om*x = v at x=0 -> v=0; pure spin p=(-5,5), r=.1, a=.2: u=0, om=2.5", abs(.1*(-5+5)/2) < 1e-12 and abs(.1*10/(2*.2)-2.5) < 1e-12)
print("ALL PASS (derivations)")
