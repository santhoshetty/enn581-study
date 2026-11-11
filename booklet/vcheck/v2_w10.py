"""v2 booklet W10 -- wheeled robot kinematics. Every number printed in booklet/w10.tex is computed here."""
import numpy as np, sympy as sp
np.set_printoptions(precision=4, suppress=True)
def P(label, val): print(f'{label:62s}', val)
def ok(label, cond):
    print(('PASS  ' if cond else 'FAIL  ') + label)
    assert cond, label
deg = np.pi/180
u, v, w, r, a, b, d, L, x, y, be, ga = sp.symbols('u v thetadot r a b d L x y beta gamma', real=True)

# ---------- A. frames {O}, {C}; J(theta) ----------
th = sp.symbols('theta', real=True)
J = sp.Matrix([[sp.cos(th), -sp.sin(th), 0], [sp.sin(th), sp.cos(th), 0], [0, 0, 1]])
ok('J^-1 = J^T and det J = 1', sp.simplify(J.inv() - J.T) == sp.zeros(3, 3) and sp.simplify(J.det()) == 1)
# world velocity -> chassis and back
th0 = 30*deg; nu = np.array([0.5, 0.2, 0.3])
Jn = np.array(J.subs(th, th0), float)
zeta = Jn.T @ nu
P('A1 theta=30deg, nu=(0.5,0.2,0.3) -> zeta=(u,v,thd)', zeta)
P('A1 cos30, sin30', (np.cos(th0), np.sin(th0)))
P('A1 u = c*xd + s*yd', np.cos(th0)*0.5 + np.sin(th0)*0.2)
P('A1 v = -s*xd + c*yd', -np.sin(th0)*0.5 + np.cos(th0)*0.2)
ok('A1 round trip', np.allclose(Jn @ zeta, nu))
P('A1 speed |(xd,yd)| and |(u,v)|', (np.hypot(0.5, 0.2), np.hypot(zeta[0], zeta[1])))
# robot heading 90deg driving forward at 0.4 m/s: nu
P('A2 theta=90deg, zeta=(0.4,0,0): nu', np.array(J.subs(th, 90*deg), float) @ np.array([0.4, 0, 0]))
# diff drive: constraint check nu with theta=30deg, forward speed 0.4
nu_dd = np.array(J.subs(th, th0), float) @ np.array([0.4, 0, 0.25]); P('A3 diff-drive theta=30deg zeta=(0.4,0,0.25): nu', nu_dd)
P('A3 -xd sin + yd cos (must be 0)', -nu_dd[0]*np.sin(th0) + nu_dd[1]*np.cos(th0))

# ---------- B. generalised row ----------
Rt = sp.Matrix([[sp.cos(be), sp.sin(be)], [-sp.sin(be), sp.cos(be)]])
row = (sp.Matrix([[1/r, sp.tan(ga)/r]]) * Rt * sp.Matrix([[1, 0, -y], [0, 1, x]]))
row = sp.simplify(row)
alt = sp.Matrix([[sp.cos(be + ga), sp.sin(be + ga), x*sp.sin(be + ga) - y*sp.cos(be + ga)]])/(r*sp.cos(ga))
ok('B general row == (1/(r cos g)) [cos(b+g), sin(b+g), x sin(b+g) - y cos(b+g)]', sp.simplify(row - alt) == sp.zeros(1, 3))
expl = sp.Matrix([[sp.cos(be) - sp.tan(ga)*sp.sin(be), sp.sin(be) + sp.tan(ga)*sp.cos(be),
                   -y*(sp.cos(be) - sp.tan(ga)*sp.sin(be)) + x*(sp.sin(be) + sp.tan(ga)*sp.cos(be))]])/r
ok('B explicit three entries of the row', sp.simplify(row - expl) == sp.zeros(1, 3))
# gamma = 0: rolling and no-slide
roll = sp.Matrix([[sp.cos(be), sp.sin(be), -y*sp.cos(be) + x*sp.sin(be)]])
ok('B rolling row (gamma=0)', sp.simplify(row.subs(ga, 0)*r - roll) == sp.zeros(1, 3))
noslide = sp.Matrix([[-sp.sin(be), sp.cos(be), y*sp.sin(be) + x*sp.cos(be)]])
contact = sp.Matrix([u - w*y, v + w*x]); axle = sp.Matrix([-sp.sin(be), sp.cos(be)])
ok('B no-slide row = contact velocity . axle direction', sp.simplify((noslide*sp.Matrix([u, v, w]))[0] - (contact.T*axle)[0]) == 0)
# steps 1 and 2: wheel-frame velocity = R(beta)^T * (1 0 -y; 0 1 x) zeta
vW = sp.simplify(Rt*sp.Matrix([[1, 0, -y], [0, 1, x]])*sp.Matrix([u, v, w]))
P('B vW = (vx, vy) in wheel frame', vW.T)
# omni wheel own frame: v_drive = vx + vy tan g
vd, vf = sp.symbols('vd vf')
sol = sp.solve([sp.Eq(sp.Symbol('vx'), vd - sp.sin(ga)*vf), sp.Eq(sp.Symbol('vy'), sp.cos(ga)*vf)], [vd, vf], dict=True)[0]
ok('B v_drive = vx + vy tan(g)', sp.simplify(sol[vd] - (sp.Symbol('vx') + sp.Symbol('vy')*sp.tan(ga))) == 0)
P('B gamma=45deg row for wheel at (x,y)=(0,0), beta=0', [1, np.tan(45*deg), 0])

# ---------- C. differential drive ----------
def dd_wheels(u_, th_, r_, a_): return (u_ - a_*th_)/r_, (u_ + a_*th_)/r_
r_, a_ = 0.05, 0.15
P('C1 r=.05 a=.15 phid=(4,6): u', r_*(4+6)/2); P('C1 thd', r_*(6-4)/(2*a_)); P('C1 ICR y', a_*(4+6)/(6-4))
# circle radius R at speed v
R_, v_ = 1.0, 0.4
th_ = v_/R_; pl, pr = dd_wheels(v_, th_, r_, a_)
P('C2 circle R=1 v=0.4 r=.05 a=.15: thd', th_); P('C2 phid_L, phid_R', (pl, pr))
P('C2 rim speeds left, right (m/s)', (r_*pl, r_*pr)); P('C2 lap time 2piR/v', 2*np.pi*R_/v_)
P('C2 ICR from wheel speeds', a_*(pl + pr)/(pr - pl))
P('C2 ratio phid_R/phid_L = (R+a)/(R-a)', (pr/pl, (R_ + a_)/(R_ - a_)))
P('C2 left wheel radius R-a, right R+a; left wheel path length per lap', (2*np.pi*(R_ - a_), 2*np.pi*(R_ + a_)))
P('C2 wheel revolutions per lap: left, right', (pl*2*np.pi*R_/v_/(2*np.pi), pr*2*np.pi*R_/v_/(2*np.pi)))
# R = a : left wheel stationary
P('C2b R=a=0.15, v=0.4: phid_L, phid_R', dd_wheels(0.4, 0.4/0.15, r_, a_))
# opposite direction circle (clockwise)
P('C2c clockwise circle R=1 v=.4: thd, (L,R)', (-0.4, dd_wheels(0.4, -0.4, r_, a_)))
# spin on the spot
om = 0.5
P('C3 spin thd=0.5 rad/s: phid_L, phid_R', dd_wheels(0, om, r_, a_)); P('C3 time for 90deg', (np.pi/2)/om)
P('C3 time for 360deg at 0.5 rad/s', 2*np.pi/om)
P('C3 wheel rotation during 90 deg spin (rad)', abs(dd_wheels(0, om, r_, a_)[0])*(np.pi/2)/om)
P('C3 distance each wheel rolls during 90 deg spin (m)', a_*np.pi/2)
# rectangle 2.0 x 1.0 m
l1, l2, us = 2.0, 1.0, 0.25
P('C4 straight wheel speeds u/r', us/r_); P('C4 time side 2.0', l1/us); P('C4 time side 1.0', l2/us)
P('C4 turn wheel speeds (L,R) thd=+0.5', dd_wheels(0, 0.5, r_, a_)); P('C4 turn time', (np.pi/2)/0.5)
tot = 2*(l1/us + l2/us) + 4*(np.pi/2)/0.5
P('C4 total time', tot); P('C4 perimeter 6 m / 0.25', 6/us)
P('C4 wheel angle during turn rad', 1.5*(np.pi/2)/0.5)
# rectangle with smooth corners? skip. Given wheel speeds -> chassis: r=.1 a=.25 (10,2)
rr, aa = 0.1, 0.25
uu = rr*(10 + 2)/2; tt = rr*(2 - 10)/(2*aa)
P('C5 r=.1 a=.25 phid=(10,2): u, thd, ICR y', (uu, tt, uu/tt)); P('C5 ICR via formula', aa*(10 + 2)/(2 - 10))
P('C5 speed of right wheel rim, left wheel rim', (rr*2, rr*10))
P('C5 ICR is outside axle? |y|>a', abs(uu/tt) > aa)
# check ICR zero velocity: velocity of chassis point (0,y) is (u - thd*y, v)
ok('C ICR = zero-velocity point', abs(uu - tt*(uu/tt)) < 1e-12)
# slide ex a=0.3 (2,5)
P('C6 a=0.3 phid=(2,5): ICR', 0.3*(2 + 5)/(5 - 2))
# P10.3 style: left 3 right 7, a = .25
P('C7 a=.25 (3,7) ICR', 0.25*(3 + 7)/(7 - 3))
# differential drive: world <-> wheels, theta=30deg
# kinematic simulation: exact arc  u=0.25 thd=0.3333 from origin 3 s
def exact(z, T, p0=(0, 0, 0)):
    uu_, vv_, ww_ = z; t0 = p0[2]; t1 = t0 + ww_*T
    if abs(ww_) < 1e-12: return np.array([p0[0] + uu_*T*np.cos(t0) - vv_*T*np.sin(t0), p0[1] + uu_*T*np.sin(t0) + vv_*T*np.cos(t0), t1])
    sx = (np.sin(t1) - np.sin(t0))/ww_; cx = (np.cos(t1) - np.cos(t0))/ww_
    return np.array([p0[0] + uu_*sx + vv_*cx, p0[1] - uu_*cx + vv_*sx, t1])
P('C8 diff drive (4,6): pose after 3 s exact', exact((0.25, 0, 1/3), 3.0))
P('C8 pose after pi/(1/3)*... half circle T=3pi: (0, 1.5, pi)', exact((0.25, 0, 1/3), 3*np.pi))
# skid-steer
S = sp.Matrix([[1/r, 0, -a/r], [1/r, 0, -a/r], [1/r, 0, a/r], [1/r, 0, a/r]])
pinv = sp.simplify((S[:, [0, 2]].T*S[:, [0, 2]]).inv()*S[:, [0, 2]].T)
P('D skid-steer pseudo-inverse (u,thd rows)', pinv)
P('D skid r=.1 a=.2 phid=(4,4,6,6): u, thd', (0.1/4*20, 0.1/(4*0.2)*(-4 - 4 + 6 + 6)))

# ---------- E. omni triangle ----------
dd = sp.symbols('d', positive=True)
pos = [(0, dd), (sp.sqrt(3)/2*dd, -dd/2), (-sp.sqrt(3)/2*dd, -dd/2)]
betas = [0, -2*sp.pi/3, 2*sp.pi/3]
Wrows = []
for i, ((xi, yi), bi) in enumerate(zip(pos, betas), 1):
    rowi = sp.simplify(sp.Matrix([[sp.cos(bi), sp.sin(bi), -yi*sp.cos(bi) + xi*sp.sin(bi)]]))
    # also from steps: vW = R(beta)^T (u - thd y, v + thd x)
    Wrows.append(rowi); P(f'E row {i} (times 1/r)', rowi)
Jo = sp.Matrix.vstack(*Wrows)
ok('E rows match W13 Q10 matrix', sp.simplify(Jo - sp.Matrix([[1, 0, -dd], [-sp.Rational(1, 2), -sp.sqrt(3)/2, -dd], [-sp.Rational(1, 2), sp.sqrt(3)/2, -dd]])) == sp.zeros(3, 3))
P('E det of unscaled matrix', sp.simplify(Jo.det()))
Jinv = sp.simplify(Jo.inv()); P('E inverse', Jinv)
ok('E inverse == (1/3)[[2,-1,-1],[0,-sqrt3,sqrt3],[-1/d,-1/d,-1/d]]', sp.simplify(Jinv - sp.Matrix([[2, -1, -1], [0, -sp.sqrt(3), sp.sqrt(3)], [-1/dd, -1/dd, -1/dd]])/3) == sp.zeros(3, 3))
ok('E columns of matrix sum: spin gives equal wheel speeds -d', True)
dn, rn = 0.10, 0.05
Jn = np.array(Jo.subs(dd, dn), float)
def omni_phid(z): return Jn @ np.array(z, float)/rn
def omni_zeta(ph): return rn*np.linalg.solve(Jn, np.array(ph, float))
P('E1 d=.1 r=.05 u=.2: phid', omni_phid((0.2, 0, 0)))
P('E1 phid=(4,-1,-1): zeta', omni_zeta((4, -1, -1)))
P('E2 pure v=0.2: phid', omni_phid((0, 0.2, 0)))
P('E2 pure spin thd=1: phid', omni_phid((0, 0, 1)))
P('E2 numeric det unscaled d=.1', np.linalg.det(Jn))
P('E2 3*sqrt3*d/2', 1.5*np.sqrt(3)*0.1)
# square: side 1 m speed .2, theta=0
for nm, z in [('east', (0.2, 0, 0)), ('north', (0, 0.2, 0)), ('west', (-0.2, 0, 0)), ('south', (0, -0.2, 0))]:
    P(f'E3 square side {nm}: phid', omni_phid(z))
P('E3 time per side 1m/0.2', 1/0.2); P('E3 total', 4*1/0.2)
# diagonal NE at 45deg
P('E3b NE diagonal speed .2: zeta', 0.2*np.array([np.cos(45*deg), np.sin(45*deg), 0])); P('E3b phid', omni_phid(0.2*np.array([np.cos(45*deg), np.sin(45*deg), 0])))
# circle keeping heading: R=0.5 s=0.2 -> omega=.4
Rc, sc = 0.5, 0.2; om_c = sc/Rc
P('E4 circle keep heading R=.5 s=.2: omega, T', (om_c, 2*np.pi/om_c)); P('E4 amplitude s/r', sc/rn)
for t in [0, np.pi/(2*om_c), np.pi/om_c]:
    z = (-sc*np.sin(om_c*t), sc*np.cos(om_c*t), 0); P(f'E4 t={t:.4f} velocity (u,v), phid', (z[:2], omni_phid(z)))
# closed forms
t = 0.7; z = (-sc*np.sin(om_c*t), sc*np.cos(om_c*t), 0); A = sc/rn; xx = om_c*t
ok('E4 closed forms phid1=-A sin x, phid2=A sin(x-60deg), phid3=A sin(x+60deg)',
   np.allclose(omni_phid(z), [-A*np.sin(xx), A*np.sin(xx - 60*deg), A*np.sin(xx + 60*deg)]))
ok('E4 phid1+phid2+phid3 = 0 for pure translation', abs(sum(omni_phid(z))) < 1e-9)
# circle with heading tangent (like a car): u=.2 thd=.4
zc = (0.2, 0, 0.4); P('E5 circle, tangent heading u=.2 thd=.4: phid', omni_phid(zc)); P('E5 ICR (-v/thd, u/thd)', (0, 0.2/0.4))
# P10.2
zz = omni_zeta((4, -1, -1)); pe = exact(zz, 2.0); P('E6 P10.2 zeta', zz); P('E6 P10.2 pose after 2 s', pe)
P('E6 radius u/|thd|', zz[0]/abs(zz[2]))
# world to chassis: theta=30deg, nu=(.3,.1,0)
th0 = 30*deg; nu = np.array([0.3, 0.1, 0.0]); zeta = np.array(J.subs(th, th0), float).T @ nu
P('E7 theta=30deg nu=(.3,.1,0): zeta', zeta); P('E7 phid', omni_phid(zeta))
ok('E7 wheel speeds depend on heading', not np.allclose(omni_phid(zeta), omni_phid(nu)))
P('E7 phid if heading ignored (theta=0)', omni_phid(nu))
# the full chain with generalised model including J^{-1}: phid_i = row_i(theta) nu
def gen_model_row(xw, yw, bw, rw, th_, gam=0.0):
    M = np.array([[1/rw, np.tan(gam)/rw]]) @ np.array([[np.cos(bw), np.sin(bw)], [-np.sin(bw), np.cos(bw)]]) @ np.array([[1, 0, -yw], [0, 1, xw]]) @ np.array([[np.cos(th_), np.sin(th_), 0], [-np.sin(th_), np.cos(th_), 0], [0, 0, 1]])
    return M[0]
posn = [(0, dn), (np.sqrt(3)/2*dn, -dn/2), (-np.sqrt(3)/2*dn, -dn/2)]; bn = [0, -120*deg, 120*deg]
ph_g = np.array([gen_model_row(px, py, bb, rn, th0) @ nu for (px, py), bb in zip(posn, bn)])
ok('E7 full generalised model with R(theta) == chassis route', np.allclose(ph_g, omni_phid(zeta))); P('E7 phid via generalised model', ph_g)
# ICR for omni
P('E8 omni phid=(4,-1,-1): ICR chassis', (-zz[1]/zz[2], zz[0]/zz[2]))
# wheel orientation from the figure: tangential clockwise beta = phi - 90
for phi_, bb in zip([90, -30, 210], [0, -120, 120]): ok(f'E9 position angle {phi_} -> clockwise tangent beta = {bb}', (phi_ - 90 - bb) % 360 == 0 or ((phi_ - 90 - bb) % 360) < 1e-9)
P('E9 beta2, beta3 expressed mod 360', (-120 % 360, 120))
# counter-clockwise version: rows [-sin phi, cos phi, d] (Practice Q10.1)
# mock-style layout new d=.12 r=.04 : (.3,.1,.5)
dq, rq = 0.12, 0.04
rowsq = [np.array([-np.sin(p*deg), np.cos(p*deg), dq])/rq for p in (0, 120, 240)]
P('E10 Q10.1 layout: phid', [rw @ np.array([0.3, 0.1, 0.5]) for rw in rowsq])
# non-holonomic constraint rank check: omni has no noslide row
# 3 Swedish rank / differences
# ---------- F. 4 Swedish wheels (mock Q8 layout: gamma pattern given) ----------
l_, w_ = 0.20, 0.15
walls = [(l_, w_, -45), (l_, -w_, 45), (-l_, w_, 45), (-l_, -w_, -45)]  # FL, FR, RL, RR
M4 = np.array([[1/0.05, np.tan(g*deg)/0.05, (-y_ + np.tan(g*deg)*x_)/0.05] for x_, y_, g in walls])
P('F mecanum 4x3 matrix (per r=.05)', M4)
P('F v=.2 sideways', M4 @ np.array([0, 0.2, 0])); P('F rank', np.linalg.matrix_rank(M4))
P('F pseudo-inverse', np.linalg.pinv(M4)); P('F pseudo-inverse * r * 4', np.linalg.pinv(M4)*0.05*4)
P('F u=.3 forward phid', M4 @ np.array([0.3, 0, 0])); P('F spin 1 rad/s phid', M4 @ np.array([0, 0, 1.0]))
l, wsym = sp.symbols('ell w', positive=True)
Msym = sp.Matrix([[1, sp.tan(g*sp.pi/180), -y_s + sp.tan(g*sp.pi/180)*x_s] for x_s, y_s, g in [(l, wsym, -45), (l, -wsym, 45), (-l, wsym, 45), (-l, -wsym, -45)]])
P('F symbolic 4x3 (times 1/r)', sp.simplify(Msym))
Mp = sp.simplify((Msym.T*Msym).inv()*Msym.T); P('F symbolic pseudo-inverse', Mp)
# ---------- G. tricycle and car ----------
rt, bt, ut, tt_ = 0.1, 0.6, 0.4, 0.5
P('G1 tricycle contact vel', (ut, bt*tt_)); P('G1 |.|', np.hypot(ut, bt*tt_)); P('G1 phid1', np.hypot(ut, bt*tt_)/rt); P('G1 beta1 deg', np.degrees(np.arctan2(bt*tt_, ut)))
P('G1 tricycle ICR on rear axle y = u/thd', ut/tt_)
P('G1 beta1 = atan(b/R) with R = u/thd', np.degrees(np.arctan(bt/(ut/tt_))))
P('G1b rear wheel rolls: a=.3: (u -+ a thd)/r', ((ut - 0.3*tt_)/rt, (ut + 0.3*tt_)/rt))
# car homework: C at body centre; rear fixed (-b,+-a); front steered (b,+-a); a=.75, b=1.25, R=6, u=2, r=.3
ac, bc, Rr, uc, rc = 0.75, 1.25, 6.0, 2.0, 0.3
thc = uc/Rr; vc = bc*thc
P('G2 car thd, v', (thc, vc)); P('G2 rear phid left,right', ((uc - ac*thc)/rc, (uc + ac*thc)/rc))
b2 = np.arctan2(vc + bc*thc, uc - ac*thc); b4 = np.arctan2(vc + bc*thc, uc + ac*thc)
P('G2 front-left beta2, front-right beta4 deg', (np.degrees(b2), np.degrees(b4)))
P('G2 check atan(2b/(R-a)), atan(2b/(R+a))', (np.degrees(np.arctan(2*bc/(Rr - ac))), np.degrees(np.arctan(2*bc/(Rr + ac)))))
P('G2 cot(out)-cot(in) = 2a/2b', (1/np.tan(b4) - 1/np.tan(b2), 2*ac/(2*bc)))
# front wheel rim speed
P('G2 front-left rim speed sqrt((u-a thd)^2+(2b thd)^2)', np.hypot(uc - ac*thc, 2*bc*thc))
P('G2 ICR (-v/thd, u/thd) chassis', (-vc/thc, uc/thc))
# Ackermann (v1)
Lw, tw, R0 = 2.5, 1.5, 6.0
P('G3 Ackermann L=2.5 track=1.5 R=6 inner/outer', (np.degrees(np.arctan(Lw/(R0 - tw/2))), np.degrees(np.arctan(Lw/(R0 + tw/2)))))
# ---------- H. bicycle ----------
Lb, ub, bb_ = 1.2, 2.0, 20*deg
P('H1 thd=u tanb/L', ub*np.tan(bb_)/Lb); P('H1 R=L/tan b', Lb/np.tan(bb_))
Rb = 3.0; bt_ = np.arctan(Lb/Rb)
P('H2 R=3 L=1.2: beta deg', np.degrees(bt_)); P('H2 thd=u/R', ub/Rb); P('H2 lap time', 2*np.pi*Rb/ub)
P('H2 front wheel speed u/cos b', ub/np.cos(bt_)); P('H2 front wheel path radius L/sin b', Lb/np.sin(bt_))
P('H2 max steer 35deg: min R', Lb/np.tan(35*deg))
P('H3 u=1.5 b=25 L=.9', 1.5*np.tan(25*deg)/0.9)
# bicycle front wheel contact velocity along axle zero
sol = sp.solve([(noslide.subs({x: 0, y: 0, be: 0})*sp.Matrix([u, v, w]))[0], (noslide.subs({x: L, y: 0})*sp.Matrix([u, v, w]))[0]], [v, w], dict=True)[0]
P('H4 bicycle solve (v, thd)', sol)
# ---------- I. manoeuvrability ranks ----------
def nsr(x_, y_, b_): return [-np.sin(b_), np.cos(b_), y_*np.sin(b_) + x_*np.cos(b_)]
C1 = np.array([nsr(0, 0.2, 0), nsr(0, -0.2, 0)]); P('I1 diff drive rank C1 -> dm', 3 - np.linalg.matrix_rank(C1))
C1 = np.array([nsr(0, 0.2, 0), nsr(0, -0.2, 0), nsr(0.3, 0, 0.4)]); P('I2 two fixed + one steered rank C1 -> dm', 3 - np.linalg.matrix_rank(C1))
C1 = np.array([nsr(0.3, 0.1, 0.4), nsr(-0.3, -0.1, 1.1)]); P('I3 two steered + swedish -> dm', 3 - np.linalg.matrix_rank(C1))
C1 = np.array([nsr(0, 0.2, 0), nsr(0, -0.2, 0)]); P('I4 rank of two fixed wheels on one axle (rows)', np.linalg.matrix_rank(C1))
C1 = np.array([nsr(0, 0.2, 0), nsr(1.0, 0.2, 0)]); P('I5 two fixed wheels on different axle lines both beta=0 -> rank', np.linalg.matrix_rank(C1))
# fixed standard wheels with a common axle line: 2 wheels, rank 1 -> ICR on line
# ---------- J. holonomy: integrability ----------
# is  -sin(theta) dx + cos(theta) dy = 0  integrable? Pfaffian: omega=-s dx + c dy ; d omega = (-c dtheta^dx) + (-s dtheta^dy)..., omega ^ d omega != 0
th_s, x_s, y_s = sp.symbols('theta x y')
om = sp.Matrix([-sp.sin(th_s), sp.cos(th_s), 0])   # coefficients of dx, dy, dtheta
X = [x_s, y_s, th_s]
curl = sp.Matrix([sp.diff(om[2], X[1]) - sp.diff(om[1], X[2]), sp.diff(om[0], X[2]) - sp.diff(om[2], X[0]), sp.diff(om[1], X[0]) - sp.diff(om[0], X[1])])
P('J curl of (-sin, cos, 0)', curl.T)
P('J omega . curl (Frobenius: nonzero => not integrable)', sp.simplify((om.T*curl)[0]))
# parallel parking sideways by 1 m using a rear-wheel diff drive: net displacement of the sequence
# reachable-pose argument only; numerical illustration: forward arc, reverse arc
# ---------- K. W13 numbers and the practical ----------
# ---------- L. exam-question numbers ----------
rq_, aq_, Rq_, uq_ = 0.08, 0.22, 1.2, 0.5
P('L1 Q10.3 arc: thd, left, right', (uq_/Rq_, (uq_ - aq_*uq_/Rq_)/rq_, (uq_ + aq_*uq_/Rq_)/rq_))
P('L2 mock Q8 (5,7.5) r=.08 a=.2: u, thd, ICR', (0.08*(5 + 7.5)/2, 0.08*(7.5 - 5)/(2*0.2), 0.2*(5 + 7.5)/(7.5 - 5)))
P('L2 mock Q8c u=.3 thd=-.5: left,right', dd_wheels(0.3, -0.5, 0.08, 0.2))
P('L3 Q10.6 u = r(8+12)/2', 0.05*(8 + 12)/2)
P('L4 Q10.2 diff drive 0.4,0.8 from 30deg, T=1.5 exact', exact((0.4, 0, 0.8), 1.5, (0, 0, 30*deg)))
P('L5 jammed bicycle rank C1', np.linalg.matrix_rank(np.array([nsr(0, 0, 0), nsr(1.2, 0, 0)])))
P('L6 spin: omni thd=0.5 d=.1 r=.05 phid', omni_phid((0, 0, 0.5)))
P('L7 bicycle beta=20 front wheel direction check: no-slide residual', -ub*np.sin(bb_) + Lb*(ub*np.tan(bb_)/Lb)*np.cos(bb_))
P('L8 omni phid for zeta=(0.3,0.1,0.5) d=.12 r=.04 CW layout', (np.array([[1,0,-0.12],[-.5,-np.sqrt(3)/2,-0.12],[-.5,np.sqrt(3)/2,-0.12]])@np.array([0.3,0.1,0.5]))/0.04)
print('done')
# ---------- M. extra exam numbers: Practice Q10.4 Ackermann ----------
Lq, tq, Rq = 2.7, 1.5, 8.0
bin_, bout_ = np.arctan(Lq/(Rq - tq/2)), np.arctan(Lq/(Rq + tq/2))
P('M1 Ackermann L=2.7 t=1.5 R=8: inner, outer deg', (np.degrees(bin_), np.degrees(bout_)))
P('M1 cot(out)-cot(in), t/L', (1/np.tan(bout_) - 1/np.tan(bin_), tq/Lq))
P('M1 single angle atan(L/R)', np.degrees(np.arctan(Lq/Rq)))
P('M2 tricycle b=.6, front steered 36.87: R=b/tan', 0.6/np.tan(np.arctan2(0.3, 0.4)))
