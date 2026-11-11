"""Verify every formula in booklet/tables.tex (sympy/numpy). Prints PASS lines."""
import numpy as np, sympy as sp
rng = np.random.default_rng(1)
ok = lambda name, cond: print(("PASS " if cond else "FAIL ") + name) or (cond or exit(1))
a, b, x, y = sp.symbols('a b x y', real=True)

# trig identities
ok("sin(a+b)", sp.simplify(sp.expand_trig(sp.sin(a+b)) - (sp.sin(a)*sp.cos(b)+sp.cos(a)*sp.sin(b))) == 0)
ok("cos(a+b)", sp.simplify(sp.expand_trig(sp.cos(a+b)) - (sp.cos(a)*sp.cos(b)-sp.sin(a)*sp.sin(b))) == 0)
ok("tan(a+b)", sp.simplify(sp.expand_trig(sp.tan(a+b)) - (sp.tan(a)+sp.tan(b))/(1-sp.tan(a)*sp.tan(b))) == 0)
ok("sin2a", sp.simplify(sp.expand_trig(sp.sin(2*a)) - 2*sp.sin(a)*sp.cos(a)) == 0)
ok("cos2a 3 forms", all(sp.simplify(sp.expand_trig(sp.cos(2*a)) - f) == 0 for f in
   [sp.cos(a)**2-sp.sin(a)**2, 2*sp.cos(a)**2-1, 1-2*sp.sin(a)**2]))
ok("tan2a", sp.simplify(sp.expand_trig(sp.tan(2*a)) - 2*sp.tan(a)/(1-sp.tan(a)**2)) == 0)
ok("c1 c12 + s1 s12 = c2 ; s12 c1 - c12 s1 = s2",
   sp.simplify(sp.cos(a)*sp.cos(a+b)+sp.sin(a)*sp.sin(a+b) - sp.cos(b)) == 0 and
   sp.simplify(sp.sin(a+b)*sp.cos(a)-sp.cos(a+b)*sp.sin(a) - sp.sin(b)) == 0)
# a cos x + b sin x = R cos(x - phi), phi = atan2(b,a); also R sin(x + psi), psi = atan2(a,b)
good = True
for _ in range(2000):
    A, B, X = rng.normal(size=3)*3
    R = np.hypot(A, B); ph = np.arctan2(B, A); ps = np.arctan2(A, B)
    good &= abs(A*np.cos(X)+B*np.sin(X) - R*np.cos(X-ph)) < 1e-9
    good &= abs(A*np.cos(X)+B*np.sin(X) - R*np.sin(X+ps)) < 1e-9
ok("a cos x + b sin x = R cos(x-phi) = R sin(x+psi)", good)
# cosine and sine rules on random triangles
good = True
for _ in range(500):
    P = rng.normal(size=(3, 2)); Aa, Bb, Cc = P
    la, lb, lc = np.linalg.norm(Bb-Cc), np.linalg.norm(Aa-Cc), np.linalg.norm(Aa-Bb)
    ang = lambda u, v, w: np.arccos(np.dot(v-u, w-u)/np.linalg.norm(v-u)/np.linalg.norm(w-u))
    A_, B_, C_ = ang(Aa, Bb, Cc), ang(Bb, Aa, Cc), ang(Cc, Aa, Bb)
    good &= np.isclose(lc**2, la**2+lb**2-2*la*lb*np.cos(C_), rtol=1e-8)
    good &= np.isclose(la/np.sin(A_), lb/np.sin(B_), rtol=1e-6) and np.isclose(lb/np.sin(B_), lc/np.sin(C_), rtol=1e-6)
ok("cosine rule and sine rule", good)
ok("cos(pi-q) = -cos q", abs(np.cos(np.pi-0.7)+np.cos(0.7)) < 1e-12)

# atan2 quadrant table: atan2(y,x) vs atan(y/x) +/- pi
good = True
for _ in range(2000):
    X, Y = rng.normal(size=2)
    t = np.arctan(Y/X)
    if X > 0: good &= abs(np.arctan2(Y, X)-t) < 1e-12
    elif Y >= 0: good &= abs(np.arctan2(Y, X)-(t+np.pi)) < 1e-12
    else: good &= abs(np.arctan2(Y, X)-(t-np.pi)) < 1e-12
ok("atan2 = atan(y/x) (+pi if x<0,y>=0; -pi if x<0,y<0)", good)
ok("atan2 on axes", np.arctan2(1, 0) == np.pi/2 and np.arctan2(-1, 0) == -np.pi/2 and np.arctan2(0, -1) == np.pi)

# rotation matrices
def Rx(t): c, s = np.cos(t), np.sin(t); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(t): c, s = np.cos(t), np.sin(t); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rz(t): c, s = np.cos(t), np.sin(t); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
ok("R^T R = I, det = +1", all(np.allclose(R(.9).T@R(.9), np.eye(3)) and abs(np.linalg.det(R(.9))-1) < 1e-12 for R in (Rx, Ry, Rz)))
ok("same-axis angles add; different axes do not commute", np.allclose(Rz(.3)@Rz(.5), Rz(.8)) and not np.allclose(Rx(.3)@Ry(.5), Ry(.5)@Rx(.3)))
# right-multiply = about current axes: R = Rz(a) Rx(b): rotate by a about fixed z first then b about the (new) x. Equivalent: fixed-frame order reversed.
v = np.array([1., 2., 3.]); a_, b_ = .4, .7
# rotating the frame: first about fixed z by a, then about FIXED x by b = Rx(b) Rz(a); about current axes = Rz(a) Rx(b)
Rcur = Rz(a_) @ Rx(b_)          # Rz then Rx about the moved x-axis
xnew = Rz(a_) @ np.array([1., 0, 0])   # moved x-axis
# rotation about moved axis by b equals Rz(a) Rx(b) Rz(a)^T applied after Rz(a):
ok("current-axis rule: R_axis(b about moved x) Rz(a) = Rz(a) Rx(b)", np.allclose(Rz(a_)@Rx(b_)@Rz(a_).T@Rz(a_), Rz(a_)@Rx(b_)))
# Rodrigues check: Rz(a) Rx(b) Rz(a)^T is rotation by b about Rz(a) x
k = xnew; K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
ok("Rz(a)Rx(b)Rz(a)^T = rotation b about moved x-axis", np.allclose(Rz(a_)@Rx(b_)@Rz(a_).T, np.eye(3)+np.sin(b_)*K+(1-np.cos(b_))*K@K))

# homogeneous inverse
R = Rz(.4)@Ry(-.3)@Rx(1.1); t = np.array([.3, -1, 2.])
T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
Ti = np.eye(4); Ti[:3,:3] = R.T; Ti[:3,3] = -R.T@t
ok("homogeneous inverse [R^T, -R^T t]", np.allclose(T@Ti, np.eye(4)))

# cross product, skew
u, w = rng.normal(size=3), rng.normal(size=3)
S = lambda a: np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
ok("a x b = [a]x b ; components", np.allclose(np.cross(u, w), S(u)@w) and np.allclose(np.cross(u, w),
   [u[1]*w[2]-u[2]*w[1], u[2]*w[0]-u[0]*w[2], u[0]*w[1]-u[1]*w[0]]))
ok("|a x b| = |a||b| sin(theta); [a]x^T = -[a]x", abs(np.linalg.norm(np.cross(u, w)) -
   np.linalg.norm(u)*np.linalg.norm(w)*np.sin(np.arccos(u@w/np.linalg.norm(u)/np.linalg.norm(w)))) < 1e-12 and np.allclose(S(u).T, -S(u)))
# Rdot = [w]x R  (finite difference)
om = np.array([.3, -.5, .8]); dt = 1e-6
Rt = Rz(.2)@Ry(.5)
Rn = (np.eye(3) + S(om)*dt) @ Rt
ok("R-dot = [omega]x R", np.allclose((Rn-Rt)/dt, S(om)@Rt, atol=1e-5))

# 2x2 and 3x3 inverse
A2 = sp.Matrix(2, 2, sp.symbols('a b c d'))
ok("2x2 inverse", sp.simplify(A2.inv() - sp.Matrix([[A2[1,1], -A2[0,1]], [-A2[1,0], A2[0,0]]])/A2.det()) == sp.zeros(2))
a_,b_,c_,d_,e_,f_,g_,h_,i_ = sp.symbols('a b c d e f g h i')
A3 = sp.Matrix([[a_,b_,c_],[d_,e_,f_],[g_,h_,i_]])
adj = sp.Matrix([[e_*i_-f_*h_, c_*h_-b_*i_, b_*f_-c_*e_],
                 [f_*g_-d_*i_, a_*i_-c_*g_, c_*d_-a_*f_],
                 [d_*h_-e_*g_, b_*g_-a_*h_, a_*e_-b_*d_]])
det3 = a_*(e_*i_-f_*h_) - b_*(d_*i_-f_*g_) + c_*(d_*h_-e_*g_)
ok("3x3 det by first row", sp.expand(A3.det()-det3) == 0)
ok("3x3 inverse = adj/det (printed matrix)", sp.simplify(A3.inv()*A3.det() - adj) == sp.zeros(3))

# inertia of standard bodies about the centroid: integrate
m, L, ra, rb, rc, r, h = sp.symbols('m L a b c r h', positive=True)
X, Y, Z, rho, th, rr = sp.symbols('X Y Z rho theta rr', positive=True)
# rod (thin): I = int (x^2) dm
rod = sp.integrate((m/L)*X**2, (X, -L/2, L/2))
ok("slender rod mL^2/12", sp.simplify(rod - m*L**2/12) == 0)
# box a x b x c along x,y,z
V = ra*rb*rc; Xs, Ys, Zs = sp.symbols('Xs Ys Zs', real=True)
Ixx = sp.integrate((m/V)*(Ys**2+Zs**2), (Xs, -ra/2, ra/2), (Ys, -rb/2, rb/2), (Zs, -rc/2, rc/2))
ok("box Ixx = m(b^2+c^2)/12", sp.simplify(Ixx - m*(rb**2+rc**2)/12) == 0)
# cylinder radius r length h about axis z and about transverse x
Vc = sp.pi*r**2*h
Izz = sp.integrate((m/Vc)*rr**2*rr, (rr, 0, r), (th, 0, 2*sp.pi), (Zs, -h/2, h/2))
Ixx_c = sp.integrate((m/Vc)*((rr*sp.sin(th))**2+Zs**2)*rr, (rr, 0, r), (th, 0, 2*sp.pi), (Zs, -h/2, h/2))
ok("cylinder Izz = mr^2/2", sp.simplify(Izz - m*r**2/2) == 0)
ok("cylinder Ixx = m(3r^2+h^2)/12", sp.simplify(Ixx_c - m*(3*r**2+h**2)/12) == 0)
# sphere
ph = sp.symbols('phi', positive=True)
Vs = sp.Rational(4, 3)*sp.pi*r**3
Is = sp.integrate((m/Vs)*(rr*sp.sin(ph))**2*rr**2*sp.sin(ph), (rr, 0, r), (ph, 0, sp.pi), (th, 0, 2*sp.pi))
ok("solid sphere 2mr^2/5", sp.simplify(Is - sp.Rational(2, 5)*m*r**2) == 0)
# thin disc (axis perpendicular, and diameter)
Id = sp.integrate((m/(sp.pi*r**2))*rr**2*rr, (rr, 0, r), (th, 0, 2*sp.pi))
Idx = sp.integrate((m/(sp.pi*r**2))*(rr*sp.sin(th))**2*rr, (rr, 0, r), (th, 0, 2*sp.pi))
ok("thin disc axis: mr^2/2 ; diameter: mr^2/4", sp.simplify(Id - m*r**2/2) == 0 and sp.simplify(Idx - m*r**2/4) == 0)

# parallel axis, scalar and tensor (Monte-Carlo-free: exact via random point masses)
pts = rng.normal(size=(40, 3)); ms = rng.uniform(.5, 2, 40); M = ms.sum()
cm = (ms[:, None]*pts).sum(0)/M
def tensor(P, mass, o): 
    Q = P-o; return sum(mi*((qi@qi)*np.eye(3)-np.outer(qi, qi)) for mi, qi in zip(mass, Q))
IC = tensor(pts, ms, cm); off = np.array([.4, -.7, 1.3])
IO = tensor(pts, ms, cm+off)
ok("tensor parallel axis I_O = I_C + m(|r|^2 1 - r r^T)", np.allclose(IO, IC + M*((off@off)*np.eye(3)-np.outer(off, off))))
ok("scalar parallel axis I = I_C + m d^2 (axis z, d = perpendicular distance)",
   abs(IO[2,2]-IC[2,2]-M*(off[0]**2+off[1]**2)) < 1e-9)
ok("off-diagonal: I_xy' = I_xy - m x y", abs(IO[0,1]-(IC[0,1]-M*off[0]*off[1])) < 1e-9)
Rr = Rz(.3)@Ry(.2)
ok("rotated frame I_B = R I_A R^T", np.allclose(tensor((Rr@pts.T).T, ms, Rr@cm), Rr@IC@Rr.T))

# Euler angles
def zyz(al, be, ga): return Rz(al)@Ry(be)@Rz(ga)
al, be, ga = .6, 1.1, -.9
Rm = zyz(al, be, ga)
cA, sA, cB, sB, cG, sG = sp.symbols('cA sA cB sB cG sG')
Rsym = sp.Matrix([[cA, -sA, 0], [sA, cA, 0], [0, 0, 1]])*sp.Matrix([[cB, 0, sB], [0, 1, 0], [-sB, 0, cB]])*sp.Matrix([[cG, -sG, 0], [sG, cG, 0], [0, 0, 1]])
printed = sp.Matrix([[cA*cB*cG-sA*sG, -cA*cB*sG-sA*cG, cA*sB], [sA*cB*cG+cA*sG, -sA*cB*sG+cA*cG, sA*sB], [-sB*cG, sB*sG, cB]])
ok("ZYZ matrix as printed", sp.simplify(Rsym-printed) == sp.zeros(3))
def zyz_ex(R, sign=+1):
    be = np.arctan2(sign*np.hypot(R[0,2], R[1,2]), R[2,2])
    al = np.arctan2(sign*R[1,2], sign*R[0,2]); ga = np.arctan2(sign*R[2,1], -sign*R[2,0])
    return al, be, ga
good = True
for _ in range(500):
    q = rng.uniform(-3, 3, 3); q[1] = abs(q[1])*0.99+.01
    Rm = zyz(*q)
    for sg in (+1, -1): good &= np.allclose(zyz(*zyz_ex(Rm, sg)), Rm)
    good &= np.allclose(zyz_ex(Rm)[1], q[1]) 
ok("ZYZ extraction (both branches reproduce R)", good)
al2, be2, ga2 = zyz_ex(zyz(.6, 1.1, -.9), -1)
ok("ZYZ other branch: beta -> -beta, alpha+pi, gamma+pi", abs(be2+1.1) < 1e-12 and abs(np.angle(np.exp(1j*(al2-(.6+np.pi))))) < 1e-9 and abs(np.angle(np.exp(1j*(ga2-(-.9+np.pi))))) < 1e-9)
# ZYX
ps_, th_, ph_ = sp.symbols('psi theta phi')
Rz_s = lambda t: sp.Matrix([[sp.cos(t), -sp.sin(t), 0], [sp.sin(t), sp.cos(t), 0], [0, 0, 1]])
Ry_s = lambda t: sp.Matrix([[sp.cos(t), 0, sp.sin(t)], [0, 1, 0], [-sp.sin(t), 0, sp.cos(t)]])
Rx_s = lambda t: sp.Matrix([[1, 0, 0], [0, sp.cos(t), -sp.sin(t)], [0, sp.sin(t), sp.cos(t)]])
Rzyx = Rz_s(ps_)*Ry_s(th_)*Rx_s(ph_)
c, s = sp.cos, sp.sin
printed = sp.Matrix([[c(ps_)*c(th_), c(ps_)*s(th_)*s(ph_)-s(ps_)*c(ph_), c(ps_)*s(th_)*c(ph_)+s(ps_)*s(ph_)],
                     [s(ps_)*c(th_), s(ps_)*s(th_)*s(ph_)+c(ps_)*c(ph_), s(ps_)*s(th_)*c(ph_)-c(ps_)*s(ph_)],
                     [-s(th_), c(th_)*s(ph_), c(th_)*c(ph_)]])
ok("ZYX matrix as printed", sp.simplify(Rzyx-printed) == sp.zeros(3))
def zyx(p, t, f): return Rz(p)@Ry(t)@Rx(f)
def zyx_ex(R, alt=False):
    th = np.arctan2(-R[2,0], np.hypot(R[0,0], R[1,0]))
    ps = np.arctan2(R[1,0], R[0,0]); ph = np.arctan2(R[2,1], R[2,2])
    if alt: th, ps, ph = np.pi-th, ps+np.pi, ph+np.pi
    return ps, th, ph
good = True
for _ in range(500):
    q = rng.uniform(-3, 3, 3); q[1] = rng.uniform(-1.4, 1.4)
    Rm = zyx(*q)
    good &= np.allclose(zyx(*zyx_ex(Rm)), Rm) and np.allclose(zyx_ex(Rm), q)
    good &= np.allclose(zyx(*zyx_ex(Rm, True)), Rm)
ok("ZYX extraction: theta=atan2(-r31,sqrt(r11^2+r21^2)); alt branch (pi-theta, psi+pi, phi+pi)", good)

# small angle
d = .1
ok("sin x ~ x within 0.2% at 0.1 rad; cos x ~ 1-x^2/2 within 1e-5", abs(d-np.sin(d))/np.sin(d) < 2e-3 and abs((1-d*d/2)-np.cos(d)) < 1e-5)
ok("tan x ~ x within 0.4% at 0.1 rad", abs(d-np.tan(d))/np.tan(d) < 4e-3)
ok("Rz(d) ~ I + [d z]x", np.allclose(Rz(1e-3), np.eye(3)+S([0, 0, 1e-3]), atol=1e-6))
ok("(1+x)^n ~ 1+nx for small x", abs((1.01)**3-(1+.03)) < 5e-4)

# units
ok("1 deg = 0.017453 rad ; 1 rad = 57.2958 deg ; 1 rpm = 0.10472 rad/s ; 1 rad/s = 9.5493 rpm",
   abs(np.pi/180-0.017453) < 5e-7 and abs(180/np.pi-57.2958) < 5e-5 and abs(2*np.pi/60-0.10472) < 5e-6 and abs(60/(2*np.pi)-9.5493) < 5e-5)
ok("3000 rpm = 314.16 rad/s ; 90 deg = 1.5708 rad", abs(3000*2*np.pi/60-314.16) < 5e-3 and abs(np.deg2rad(90)-1.5708) < 5e-5)
# gearbox: energy balance (ideal), reflected inertia
Nn, Im, Qm, wl = 50., 1e-5, .02, 3.
ok("ideal gear: w_m = N w_l ; tau_l = N tau_m (power balance) ; I_reflected = N^2 I_m",
   abs(Qm*Nn*wl - (Nn*Qm)*wl) < 1e-15 and abs(.5*Im*(Nn*wl)**2 - .5*(Nn**2*Im)*wl**2) < 1e-15)

# quadratic formula and Vieta
good = True
for _ in range(500):
    A_, B_, C_ = rng.normal(size=3)
    if B_*B_-4*A_*C_ < 0: continue
    rts = np.sort([(-B_+s_*np.sqrt(B_*B_-4*A_*C_))/(2*A_) for s_ in (1, -1)])
    good &= np.allclose(rts, np.sort(np.roots([A_, B_, C_]))) and abs(rts.sum()+B_/A_) < 1e-9 and abs(rts.prod()-C_/A_) < 1e-9
ok("quadratic formula + Vieta (sum=-b/a, product=c/a)", good)

# cubic BC matrix and closed form
T_, q0, qF, v0, vF = sp.symbols('T q0 qF v0 vF')
Mx = sp.Matrix([[1, 0, 0, 0], [1, T_, T_**2, T_**3], [0, 1, 0, 0], [0, 1, 2*T_, 3*T_**2]])
sol = Mx.LUsolve(sp.Matrix([q0, qF, v0, vF]))
D_ = qF-q0
ok("cubic a0..a3 closed form", sp.simplify(sol[0]-q0) == 0 and sp.simplify(sol[1]-v0) == 0 and
   sp.simplify(sol[2]-(3*D_-(2*v0+vF)*T_)/T_**2) == 0 and sp.simplify(sol[3]-(-2*D_+(v0+vF)*T_)/T_**3) == 0)
ok("cubic worked: 10->50 in 2 s -> (10,0,30,-10)", [x.subs({q0: 10, qF: 50, v0: 0, vF: 0, T_: 2}) for x in sol] == [10, 0, 30, -10])
ok("cubic worked: 20->80, 3 s, vF=10 -> a2=16.667, a3=-3.333",
   abs(float(sol[2].subs({q0: 20, qF: 80, v0: 0, vF: 10, T_: 3}))-16.6667) < 1e-3 and abs(float(sol[3].subs({q0: 20, qF: 80, v0: 0, vF: 10, T_: 3}))+3.3333) < 1e-3)
# quintic rest-to-rest
a3_, a4_, a5_ = sp.symbols('a3 a4 a5')
qs = a3_*T_**3+a4_*T_**4+a5_*T_**5
s5 = sp.solve([qs-D_, sp.diff(a3_*sp.Symbol('t')**3+a4_*sp.Symbol('t')**4+a5_*sp.Symbol('t')**5, sp.Symbol('t')).subs(sp.Symbol('t'), T_),
               sp.diff(a3_*sp.Symbol('t')**3+a4_*sp.Symbol('t')**4+a5_*sp.Symbol('t')**5, sp.Symbol('t'), 2).subs(sp.Symbol('t'), T_)], [a3_, a4_, a5_])
ok("quintic rest-to-rest a3=10D/T^3, a4=-15D/T^4, a5=6D/T^5",
   sp.simplify(s5[a3_]-10*D_/T_**3) == 0 and sp.simplify(s5[a4_]+15*D_/T_**4) == 0 and sp.simplify(s5[a5_]-6*D_/T_**5) == 0)

# Routh-Hurwitz cubic s^3 + a2 s^2 + a1 s + a0
good = True
for _ in range(5000):
    c2, c1, c0 = rng.uniform(-2, 6, 3)
    stable = np.all(np.roots([1, c2, c1, c0]).real < 0)
    rule = c2 > 0 and c1 > 0 and c0 > 0 and c2*c1 > c0
    good &= (stable == rule)
ok("Routh-Hurwitz cubic: all coeffs > 0 and a2 a1 > a0", good)
ok("PID example kv=10,kp=25,ki=40: roots -7.335, -1.332+-1.918i ; ki=250 -> -10, +-5i",
   np.allclose(sorted(np.roots([1, 10, 25, 40]).real), sorted([-7.335, -1.332, -1.332]), atol=1e-3) and
   np.allclose(sorted(np.roots([1, 10, 25, 250]).real), [-10, 0, 0], atol=1e-6))

# second-order step response
def step_info(zeta, wn):
    t = np.linspace(0, 10/(zeta*wn), 400001)
    wd = wn*np.sqrt(1-zeta**2)
    ys = 1-np.exp(-zeta*wn*t)*(np.cos(wd*t)+zeta*wn/wd*np.sin(wd*t))
    Mp = ys.max()-1
    tp = t[ys.argmax()]
    out = np.where(abs(ys-1) > .02)[0][-1]
    return Mp, tp, t[out+1], wd
for zeta, wn in [(.6, 5.), (.3, 2.), (.8, 10.)]:
    Mp, tp, ts, wd = step_info(zeta, wn)
    good = abs(Mp-np.exp(-np.pi*zeta/np.sqrt(1-zeta**2))) < 1e-4 and abs(tp-np.pi/wd) < 1e-3 and abs(ts-4/(zeta*wn))/ts < .35
    ok(f"2nd order zeta={zeta}, wn={wn}: overshoot exp(-pi z/sqrt(1-z^2))={Mp:.4f}, peak pi/wd={tp:.3f}, 2% settle {ts:.3f} vs 4/(z wn)={4/(zeta*wn):.3f}", good)
ok("zeta=0.6 overshoot 9.48 %", abs(np.exp(-np.pi*.6/.8)-0.0948) < 5e-5)
ok("zeta from overshoot: zeta = -ln(Mp)/sqrt(pi^2+ln^2 Mp)", abs(-np.log(.0948)/np.sqrt(np.pi**2+np.log(.0948)**2)-0.6) < 1e-3)
print("ALL PASS (tables)")
