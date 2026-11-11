"""v2 W11 booklet checks: wheeled-robot dynamics (Newton-Euler, chassis frame {C}, inertial {O}).
Run:  python3 v2_w11.py   -> prints every number used in w11.tex"""
import numpy as np, sympy as sp
np.set_printoptions(precision=6, suppress=True)
ok = True
def chk(label, got, want, tol=1e-6):
    global ok
    g = np.array(got, float); w = np.array(want, float)
    good = np.allclose(g, w, atol=tol)
    ok &= good
    print(("[PASS] " if good else "[FAIL] ") + label, "got", np.round(g, 6).tolist(), "want", np.round(w, 6).tolist())

def Rz(t): return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
def Jd(t, td): return td*np.array([[-np.sin(t), -np.cos(t), 0], [np.cos(t), -np.sin(t), 0], [0, 0, 0]])
def Mmat(m, I, xG, yG):
    return np.array([[m, 0, -m*yG], [0, m, m*xG], [-m*yG, m*xG, I+m*(xG**2+yG**2)]])
def cvec(m, xG, yG, u, v, w):
    return np.array([-m*w*(v+xG*w), m*w*(u-yG*w), m*w*(xG*u+yG*v)])

# ---------- symbolic derivation: acceleration terms, rows ----------
u, v, w, ud, vd, wd, x, y, m, I = sp.symbols('u v w ud vd wd x y m I')
Pc2 = sp.Matrix([ud, vd, 0]); om = sp.Matrix([0, 0, w]); omd = sp.Matrix([0, 0, wd])
Pg = sp.Matrix([x, y, 0]); Pc1 = sp.Matrix([u, v, 0])
t1 = omd.cross(Pg); t2 = om.cross(Pc1); t3 = om.cross(om.cross(Pg))
print("tangential   w_dot x P_G      =", list(t1)[:2])
print("Coriolis-like w x Pdot_C      =", list(t2)[:2])
print("centripetal  w x w x P_G      =", list(t3)[:2])
a = Pc2 + t1 + t2 + t3
print("a_G (C) =", list(a)[:2])
fx = m*a[0]; fy = m*a[1]
nz = I*wd - (y*fx - x*fy)
row3 = sp.expand(nz)
Msym = sp.Matrix([[m, 0, -m*y], [0, m, m*x], [-m*y, m*x, I+m*(x**2+y**2)]])
csym = sp.Matrix([-m*w*(v+x*w), m*w*(u-y*w), m*w*(x*u+y*v)])
zd = sp.Matrix([ud, vd, wd])
rhs = Msym*zd + csym
print("rows of M zdot + c - (fx,fy,nz) =", [sp.simplify(e) for e in (rhs - sp.Matrix([sp.expand(fx), sp.expand(fy), row3]))])

# ---------- numeric: velocity of G, scenario A ----------
xG, yG = 0.3, 0.2; uu, vv, ww = 1.0, 0.5, 2.0; th = np.radians(30)
vG_C = np.array([uu-ww*yG, vv+ww*xG])
vG_O = Rz(th)[:2, :2] @ vG_C
print("A: vG in {C} =", vG_C, " vG in {O} (theta=30deg) =", vG_O, " speed", np.linalg.norm(vG_C))
# finite-difference cross-check of {O} velocity
def posG(t):
    th_t = th + ww*t
    # constant zeta in {C}: simulate small step: C moves with J(theta) zeta
    return None
# analytic check via sympy: position of C(t), theta(t)=th+w t, zeta constant
tt = sp.symbols('tt')
thS = th + ww*tt
Cx = sp.integrate(uu*sp.cos(thS)-vv*sp.sin(thS), tt); Cy = sp.integrate(uu*sp.sin(thS)+vv*sp.cos(thS), tt)
Gx = Cx + xG*sp.cos(thS)-yG*sp.sin(thS); Gy = Cy + xG*sp.sin(thS)+yG*sp.cos(thS)
chk("A: dG/dt at t=0 equals R(theta)[u - w yG, v + w xG]", [float(sp.diff(Gx, tt).subs(tt, 0)), float(sp.diff(Gy, tt).subs(tt, 0))], vG_O)

# ---------- B: acceleration of G, all four terms ----------
uB, vB, wB, udB, vdB, wdB = 1.5, 0.2, 0.8, 0.3, -0.1, 0.5; xB, yB = 0.2, 0.1
t_pc = np.array([udB, vdB]); t_tan = wdB*np.array([-yB, xB]); t_cor = wB*np.array([-vB, uB]); t_cen = -wB**2*np.array([xB, yB])
aG = t_pc+t_tan+t_cor+t_cen
print("B: terms", t_pc, t_tan, t_cor, t_cen, " sum", aG)
# finite difference check: zeta(t)=zeta0+zetadot0*t, theta(t)=th0+w t+0.5 wd t^2
th0 = 0.4
tt = sp.symbols('tt')
thS = th0+wB*tt+sp.Rational(1, 2)*wdB*tt**2
ux = uB+udB*tt; vx = vB+vdB*tt
vCx = ux*sp.cos(thS)-vx*sp.sin(thS); vCy = ux*sp.sin(thS)+vx*sp.cos(thS)
# velocity of G in {O}: vC + (wdot_vec) x R p
wS = sp.diff(thS, tt)
vGx = vCx - wS*(xB*sp.sin(thS)+yB*sp.cos(thS)); vGy = vCy + wS*(xB*sp.cos(thS)-yB*sp.sin(thS))
aGO = np.array([float(sp.diff(vGx, tt).subs(tt, 0)), float(sp.diff(vGy, tt).subs(tt, 0))])
chk("B: a_G in {O} rotated back into {C} equals four-term sum", Rz(th0)[:2, :2].T @ aGO, aG)
print("B: printed (v-theta-dot) y form would give a_y =", float(t_pc[1]+t_tan[1]+wB*vB+t_cen[1]))

# ---------- C: Q11.4 (C=G) ----------
print("C: Q11.4 a =", 0.3-0.2*0.8, -0.1+1.5*0.8, " printed form a_y =", -0.1+0.2*0.8)

# ---------- D: Q11.1 offset COM at rest ----------
m1, I1, x1, y1 = 8, 0.3, 0.05, 0.02
M1 = Mmat(m1, I1, x1, y1); print("D: M =\n", M1)
zd1 = np.linalg.solve(M1, [6, 0, 0.5]); print("D: udot, vdot, thetadd =", zd1)
chk("D: M33 = 0.3232", M1[2, 2], 0.3232)
chk("D: solution", zd1, [0.79133, -0.10333, 2.06667], 1e-4)
print("D: wrong M33=I gives thetadd =", 0.5/0.3 if False else None)
M1w = M1.copy(); M1w[2, 2] = 0.3
print("D: wrong M33=I: thetadd =", np.linalg.solve(M1w, [6, 0, 0.5])[2])

# ---------- E: diff drive, Q11.3 ----------
m3, I3, r3, a3 = 12, 0.4, 0.08, 0.25; tau = np.array([0.5, 0.9])
F = tau/r3; fx3 = F.sum(); nz3 = a3*(F[1]-F[0])
print("E: F =", F, "fx =", fx3, "nz =", nz3, " udot =", fx3/m3, " thetadd =", nz3/I3, " R =", (fx3/m3)/(nz3/I3), "I fx/(m nz)=", I3*fx3/(m3*nz3))
chk("E: B F", np.array([[1, 1], [0, 0], [-a3, a3]]) @ F, [fx3, 0, nz3])
# power check at t=1s
u1 = fx3/m3*1; w1 = nz3/I3*1
phi = np.array([(u1-a3*w1)/r3, (u1+a3*w1)/r3])
print("E: at t=1 s: u =", u1, " thetadot =", w1, " phidot_L,R =", phi, " power tau.phi =", tau@phi, " Q.zeta =", fx3*u1+nz3*w1)
chk("E: power", tau@phi, fx3*u1+nz3*w1)
print("E: lateral force to hold v=0 at t=1: m u w =", m3*u1*w1)

# ---------- F: circle scenario ----------
mC, vC_, RC = 12.0, 0.6, 1.5
wC = vC_/RC; fyC = mC*vC_**2/RC
aC, rC = 0.2, 0.1
print("F: thetadot =", wC, " f_y =", fyC, " (= m u thetadot =", mC*vC_*wC, ")  wheel speeds (a=0.2, r=0.1):", (vC_-aC*wC)/rC, (vC_+aC*wC)/rC)
print("F: slide-style: m=10,u=2,thetadot=0.5 -> R =", 2/0.5, " f_y =", 10*2*0.5)
# if wheels also spin up: with fx,nz
# ---------- G: one Euler step ----------
m5, I5, r5, a5 = 10, 0.25, 0.1, 0.2
xi = np.array([1.0, 2.0, np.pi/6]); zeta = np.array([1.0, 0.0, 1.6]); dt = 0.1
tauG = np.array([0.4, 0.6]); Fg = tauG/r5
Qw = np.array([Fg.sum(), 0, a5*(Fg[1]-Fg[0])])
c5 = cvec(m5, 0, 0, *zeta)
print("G: wheel wrench =", Qw, " c =", c5)
Mg = Mmat(m5, I5, 0, 0)
zd_free = np.linalg.solve(Mg, Qw-c5)
Qgrip = Qw.copy(); Qgrip[1] = c5[1]   # lateral no-slip reaction
zd_grip = np.linalg.solve(Mg, Qgrip-c5)
print("G: free-body zdot =", zd_free, "   with no-slip reaction f_y = %.1f N: zdot =" % c5[1], zd_grip)
z_new = zeta+dt*zd_grip
xid_new = Rz(xi[2]) @ z_new          # slide order: new zeta, J(theta_k)
xi_new = xi+dt*xid_new
xi_new_b = xi+dt*(Rz(xi[2]) @ zeta)  # explicit alternative: old zeta
print("G: zeta_new =", z_new, " xidot =", xid_new, " xi_new =", xi_new, " (theta deg %.3f)" % np.degrees(xi_new[2]), " alt (old zeta) xi_new =", xi_new_b)

# ---------- H: loads between frames, Q11.2 ----------
th2 = np.radians(120); QO = np.array([4, -2, 0.7])
QC = Rz(th2).T @ QO; print("H: Q_C =", QC, " |f| =", np.linalg.norm(QC[:2]), np.linalg.norm(QO[:2]), " with J instead of J^T:", (Rz(th2) @ QO))
# world accel example (v1)
QCw = np.array([10, 0, 0.4]); th30 = np.radians(30)
print("H2: Q_O =", Rz(th30) @ QCw)

# ---------- I: J^-1 Jdot, Lagrange cross-check ----------
th_ = 0.7; td_ = 1.3
Om = np.linalg.inv(Rz(th_)) @ Jd(th_, td_)
print("I: J^-1 Jdot =\n", Om)
chk("I: J^-1 Jdot = thetadot*[[0,-1,0],[1,0,0],[0,0,0]]", Om, td_*np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]))
# random state, offset G
rng = np.random.default_rng(3)
mm, II, xg, yg = 7.0, 0.35, 0.12, -0.05
Mb = Mmat(mm, II, xg, yg)
xi_ = rng.normal(size=3); xid = rng.normal(size=3); xidd = rng.normal(size=3)
J = Rz(xi_[2]); Jdot = Jd(xi_[2], xid[2])
zeta_ = np.linalg.solve(J, xid)
zdot_ = np.linalg.solve(J, xidd-Jdot@zeta_)
QC_ = Mb@zdot_+cvec(mm, xg, yg, *zeta_)
QO_slides = J @ QC_
MO = np.linalg.inv(J.T) @ Mb @ np.linalg.inv(J)
cO = np.linalg.inv(J.T) @ (cvec(mm, xg, yg, *zeta_) - Mb @ np.linalg.inv(J) @ Jdot @ np.linalg.inv(J) @ xid)
chk("I: M_O xi_dd + c_O = J Q_C (corrected p.34)", MO@xidd+cO, QO_slides)
cO_printed = np.linalg.inv(J.T) @ (cvec(mm, xg, yg, *zeta_) - Mb @ Jdot @ np.linalg.inv(J) @ xid)
print("I: printed p.34 gives", MO@xidd+cO_printed, " vs correct", QO_slides, " differ:", not np.allclose(MO@xidd+cO_printed, QO_slides))
# Lagrange (sympy): T = 1/2 m |v_G|^2 + 1/2 I thetadot^2 in world coordinates (x,y,theta of C)
X, Y, TH = sp.symbols('X Y TH'); Xd, Yd, THd = sp.symbols('Xd Yd THd'); Xdd, Ydd, THdd = sp.symbols('Xdd Ydd THdd')
xgS, ygS = sp.Rational(12, 100), sp.Rational(-5, 100)
# G position
GX = X+xgS*sp.cos(TH)-ygS*sp.sin(TH); GY = Y+xgS*sp.sin(TH)+ygS*sp.cos(TH)
q = sp.Matrix([X, Y, TH]); qd = sp.Matrix([Xd, Yd, THd]); qdd = sp.Matrix([Xdd, Ydd, THdd])
vGx = sum(sp.diff(GX, qi)*qdi for qi, qdi in zip(q, qd)); vGy = sum(sp.diff(GY, qi)*qdi for qi, qdi in zip(q, qd))
T = sp.Rational(1, 2)*sp.Rational(7)*(vGx**2+vGy**2)+sp.Rational(1, 2)*sp.Rational(35, 100)*THd**2
def ddt(expr):
    return sum(sp.diff(expr, qi)*qdi for qi, qdi in zip(q, qd))+sum(sp.diff(expr, qdi)*qddi for qdi, qddi in zip(qd, qdd))
QL = sp.Matrix([ddt(sp.diff(T, qdi))-sp.diff(T, qi) for qi, qdi in zip(q, qd)])
subs = {X: xi_[0], Y: xi_[1], TH: xi_[2], Xd: xid[0], Yd: xid[1], THd: xid[2], Xdd: xidd[0], Ydd: xidd[1], THdd: xidd[2]}
QL_n = np.array([float(e.subs(subs)) for e in QL])
chk("I: Lagrange (T=1/2 xi_d^T M_O xi_d, world coords) equals Newton-Euler J Q_C", QL_n, QO_slides)
# kinetic energy identity T = 1/2 zeta^T M zeta
Tn = float(T.subs(subs)); chk("I: T = 1/2 zeta^T M zeta", 0.5*zeta_@Mb@zeta_, Tn)
# C=G: Lagrange gives c_O = 0
chk("I: C=G c_O = 0 (th=90deg,u=1,v=0,w=2)", np.linalg.inv(Rz(np.pi/2).T) @ (cvec(10, 0, 0, 1, 0, 2) - Mmat(10, .25, 0, 0) @ np.linalg.inv(Rz(np.pi/2)) @ Jd(np.pi/2, 2) @ np.linalg.inv(Rz(np.pi/2)) @ (Rz(np.pi/2) @ [1, 0, 2])), [0, 0, 0])

# ---------- J: moment transfer example ----------
print("J: Nz =", 0.4+(0.0*10-0.05*4), " (G 0.05 m ahead of C, f=(10,4), n=0.4)")
# ---------- K: Swedish ----------
aS, bS = 0.2, 0.25; s = 1/np.sqrt(2)
pos = [(bS, aS), (-bS, aS), (-bS, -aS), (bS, -aS)]; dirs = [(s, s), (s, -s), (s, s), (s, -s)]
B = np.array([[d[0] for d in dirs], [d[1] for d in dirs], [p[0]*d[1]-p[1]*d[0] for p, d in zip(pos, dirs)]])
print("K: B =\n", B, "\n   rank", np.linalg.matrix_rank(B))
for Fv in ([5, -5, 5, -5], [5, 5, -5, -5], [5, 5, 5, 5], [5, 0, 0, 5]):
    print("   F =", Fv, "->", B@np.array(Fv, float))
print("   null space vector:", np.array([1, 1, 1, 1]) if np.allclose(B@np.ones(4), 0) else None, " B@(1,-1,-1,1)=", B@np.array([1, -1, -1, 1]), " B@[1,-1,-1,1]")
import scipy.linalg as sl
print("   null:", sl.null_space(B).T)
print("   m=10,I=0.6: accel for F=(5,-5,5,-5):", np.linalg.solve(Mmat(10, .6, 0, 0), B@np.array([5, -5, 5, -5.])))
# printed p.34 on the C=G test case
th9=np.pi/2; J9=Rz(th9); z9=np.array([1,0,2.]); c9=cvec(10,0,0,*z9); M9=Mmat(10,.25,0,0)
printed=np.linalg.inv(J9.T)@(c9-M9@Jd(th9,2)@np.linalg.inv(J9)@(J9@z9))
print("I2: printed p.34 c_O (C=G, th=90, u=1,v=0,w=2) =",printed)
chk("I2: printed form = (-20,20,0)",printed,[-20,20,0])
print("ALL OK" if ok else "SOME FAILED")
