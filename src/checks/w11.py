"""W11 Wheeled robot dynamics — verifies every number and identity used on the W11 tab and the ddsim widget vectors.
Run: python3 site/src/checks/w11.py  (writes w11_vectors.json for the widget tests)"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import sympy as sp
from scipy.integrate import solve_ivp

out = {}
t = sp.symbols('t', real=True)

# ================= 1. Acceleration of G expressed in {C} (slides pp. 19-24) =================
th = sp.Function('theta')(t); u = sp.Function('u')(t); v = sp.Function('v')(t)
xG, yG = sp.symbols('x_G y_G', real=True)
Rm = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])
vC = Rm * sp.Matrix([u, v])                       # inertial velocity of C
aC = vC.diff(t)
aG = aC + (Rm * sp.Matrix([xG, yG])).diff(t, 2)   # inertial acceleration of G (G fixed in chassis)
aG_C = sp.simplify(Rm.T * aG)                     # expressed in {C}
w, wd = th.diff(t), th.diff(t, 2)
x_corr = u.diff(t) - yG*wd - v*w - xG*w**2
y_corr = v.diff(t) + xG*wd + u*w - yG*w**2        # u*thetadot (correct)
y_print = v.diff(t) + xG*wd + v*w - yG*w**2       # as printed on pp. 24-25 (v*thetadot)
check('accel of G in C: x component = slide p.24', sp.simplify(aG_C[0] - x_corr) == 0)
check('accel of G in C: y component = u*thetadot form (p.26 matrix)', sp.simplify(aG_C[1] - y_corr) == 0)
diff_print = sp.simplify(aG_C[1] - y_print)
check('printed y form (v*thetadot) is wrong: residual = (u - v)*thetadot', sp.simplify(diff_print - (u - v)*w) == 0)
# point-mass sanity: constant speed u on a circle (v=0, thetadot=w0): centripetal u*w0 along +y
u0, w0 = 2.0, 0.5
check('circle at constant u: a_y = u*thetadot = 1.0 m/s^2 (printed form would give 0)', abs(u0*w0 - 1.0) < 1e-12 and abs(0*w0) < 1e-12)
out['centripetal'] = dict(u=u0, w=w0, ay=u0*w0, fy_m10=10*u0*w0)
# derivation of the rotating-frame terms by unit vectors: xhat' = w yhat, yhat' = -w xhat
a_vec = sp.Matrix([sp.diff(u, t), sp.diff(v, t)]) + w*sp.Matrix([-v, u])
check('unit-vector derivation: a_C = (udot - v w, vdot + u w)', sp.simplify(a_vec - sp.simplify(Rm.T*aC)) == sp.zeros(2, 1))

# ================= 2. Matrix form M zeta_dot + c(zeta) = Q (p.26) =================
m, I, ud, vd, wdd, wc, uu, vv = sp.symbols('m I ud vd wdd w u v', real=True)
Fx = m*(ud - yG*wdd - vv*wc - xG*wc**2)
Fy = m*(vd + xG*wdd + uu*wc - yG*wc**2)       # corrected y (u*thetadot)
fx, fy = Fx, Fy
nz = I*wdd - (yG*fx - xG*fy)
Mmat = sp.Matrix([[m, 0, -m*yG], [0, m, m*xG], [-m*yG, m*xG, I + m*(xG**2 + yG**2)]])
cvec = sp.Matrix([-m*wc*(vv + xG*wc), m*wc*(uu - yG*wc), m*wc*(xG*uu + yG*vv)])
lhs = Mmat*sp.Matrix([ud, vd, wdd]) + cvec
check('p.26 matrix form reproduces f_x, f_y, n_z of p.25 (with u*thetadot)', sp.simplify(lhs - sp.Matrix([fx, fy, nz])) == sp.zeros(3, 1))
check('M is symmetric', Mmat == Mmat.T)
check('M[2,2] = inertia about C (parallel axis)', sp.simplify(Mmat[2, 2] - (I + m*(xG**2 + yG**2))) == 0)
# with the printed v*thetadot the matrix identity breaks
Fy_p = m*(vd + xG*wdd + vv*wc - yG*wc**2)
nz_p = I*wdd - (yG*Fx - xG*Fy_p)
check('with the printed v*thetadot the p.26 matrix is NOT reproduced', sp.simplify(lhs - sp.Matrix([Fx, Fy_p, nz_p])) != sp.zeros(3, 1))
# moment transfer sign (p.15): N_z = n_z + (y f_x - x f_y)  <=>  moment of force at C about G
fxv, fyv, xs, ys = sp.symbols('fxv fyv xs ys')
mom = (sp.Matrix([-xs, -ys, 0]).cross(sp.Matrix([fxv, fyv, 0])))[2]   # (-P_CG) x f_C
check('p.15 moment transfer: (-P_CG) x f = y f_x - x f_y (z)', sp.simplify(mom - (ys*fxv - xs*fyv)) == 0)

# ================= 3. Differential drive (C = G at axle midpoint), p.37 =================
m_, I_, r_, a_ = 10.0, 0.25, 0.1, 0.2
tL, tR = 0.4, 0.6
FL, FR = tL/r_, tR/r_                         # F1 = left (+y), F2 = right (-y)
B = np.array([[1, 1], [0, 0], [-a_, a_]])
Q = B @ np.array([FL, FR])
# moment from first principles: n_z = x f_y - y f_x for forces along +x at (0,+a) and (0,-a)
nz_fp = 0*FL - (a_)*FL + 0*FR - (-a_)*FR
close('p.37 matrix: wrench from F1=4 N (left), F2=6 N (right)', Q, [10.0, 0.0, 0.4])
close('p.37 third row = x f_y - y f_x for forces at (0,+/-a)', Q[2], nz_fp)
Mdd = np.diag([m_, m_, I_])
zd0 = np.linalg.solve(Mdd, Q)                  # at rest c = 0
close('worked example at rest: udot, vdot, thetaddot', zd0, [1.0, 0.0, 1.6])
Rpath = (Q[0]/m_) / (Q[2]/I_)
close('worked example: path radius R = udot/thetaddot', Rpath, 0.625)
close('R = I f_x/(m n_z)', I_*Q[0]/(m_*Q[2]), 0.625)
# state at t = 1 s with grip (v = 0)
t1 = 1.0
u1, w1 = zd0[0]*t1, zd0[2]*t1
c1 = np.array([-m_*0*w1, m_*u1*w1, 0.0])       # c = (-m v w, m u w, 0) for C = G
close('c(zeta) at t=1 s', c1, [0.0, 16.0, 0.0])
close('lateral wheel force needed to hold v=0 (f_y = m u w)', m_*u1*w1, 16.0)
close('free-body (no lateral grip): vdot = -u w', -u1*w1, -1.6)
phiR, phiL = (u1 + a_*w1)/r_, (u1 - a_*w1)/r_
close('wheel speeds at t=1 s (rad/s): phiR, phiL', [phiR, phiL], [13.2, 6.8])
P_mech = tL*phiL + tR*phiR
P_chassis = Q[0]*u1 + Q[2]*w1
close('power: tau.phidot = Q.zeta at t=1 s (W)', [P_mech, P_chassis], [10.64, 10.64])
# one forward-Euler step of the p.35 recipe from rest, dt = 0.1 s
dt_e = 0.1
zeta1 = np.zeros(3) + dt_e*zd0
close('Euler step from rest, dt=0.1: zeta = (0.1, 0, 0.16)', zeta1, [0.1, 0.0, 0.16])
out['dd_example'] = dict(m=m_, I=I_, r=r_, a=a_, tauL=tL, tauR=tR, FL=FL, FR=FR, fx=Q[0], nz=Q[2], udot=zd0[0], thetadd=zd0[2], R=Rpath,
                         u1=u1, w1=w1, fy_grip=m_*u1*w1, phiR=phiR, phiL=phiL, P=P_mech)

# W10 link: wheel kinematic matrix from p.6 (gamma = 0, beta = 0, wheels at (0, +/-a)); Q = Jw^T tau = B F
def wheel_row(r, gamma, beta, xw, yw):
    a1 = np.array([1/r, np.tan(gamma)/r])
    a2 = np.array([[np.cos(beta), np.sin(beta)], [-np.sin(beta), np.cos(beta)]])
    a3 = np.array([[1, 0, -yw], [0, 1, xw]])
    return a1 @ a2 @ a3                         # acts on zeta = (u, v, w) (theta = 0 so J = I)
Jw = np.vstack([wheel_row(r_, 0, 0, 0, +a_), wheel_row(r_, 0, 0, 0, -a_)])   # rows: left, right
close('p.6 wheel model, left/right rows', Jw, [[10, 0, -2], [10, 0, 2]])
close('Q = Jw^T tau equals B (tau/r)', Jw.T @ np.array([tL, tR]), B @ np.array([FL, FR]))

# ================= 4. Offset centre of mass example =================
xg, yg = 0.05, 0.0
Mo = np.array([[m_, 0, -m_*yg], [0, m_, m_*xg], [-m_*yg, m_*xg, I_ + m_*(xg**2 + yg**2)]])
Qo = np.array([10.0, 0.0, 0.4])
zo = np.linalg.solve(Mo, Qo)
close('offset COM x_G=0.05 m at rest: udot, vdot, thetaddot', zo, [1.0, -0.08, 1.6])
close('inertia about C = I + m x_G^2', Mo[2, 2], 0.275)
close('naive thetadd = n_z/(I + m x_G^2) = 1.4545 (wrong)', Qo[2]/Mo[2, 2], 0.4/0.275)
close('a_C,y = -x_G*thetadd = -0.08 (C swings sideways)', -xg*zo[2], zo[1])
out['offset'] = dict(xG=xg, Ic=Mo[2, 2], zeta_dot=zo.tolist(), naive=Qo[2]/Mo[2, 2])

# ================= 5. Inertial frame: M_O, c_O (p.34) and Q_C = J^T Q_O (p.32) =================
rng = np.random.default_rng(11)
def Jmat(th_):
    c, s = np.cos(th_), np.sin(th_)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
def Jdot(th_, w_):
    c, s = np.cos(th_), np.sin(th_)
    return w_*np.array([[-s, -c, 0], [c, -s, 0], [0, 0, 0.0]])
def cvec_num(M_, I_g, xg_, yg_, z):
    u_, v_, w_ = z
    return np.array([-M_*w_*(v_ + xg_*w_), M_*w_*(u_ - yg_*w_), M_*w_*(xg_*u_ + yg_*v_)])
def Mnum(M_, I_g, xg_, yg_):
    return np.array([[M_, 0, -M_*yg_], [0, M_, M_*xg_], [-M_*yg_, M_*xg_, I_g + M_*(xg_**2 + yg_**2)]])
ok_all, bad_slide = True, False
for case in range(5):
    M_, Ig = rng.uniform(5, 15), rng.uniform(0.1, 1)
    xg_, yg_ = rng.uniform(-.2, .2, 2)
    th_ = rng.uniform(-3, 3)
    z = rng.uniform(-2, 2, 3)               # (u, v, w)
    xi_dd = rng.uniform(-3, 3, 3)
    J, Jd = Jmat(th_), Jdot(th_, z[2])
    xi_d = J @ z
    M3 = Mnum(M_, Ig, xg_, yg_)
    zd = np.linalg.inv(J) @ (xi_dd - Jd @ z)                    # p.30
    QC = M3 @ zd + cvec_num(M_, Ig, xg_, yg_, z)                # p.26
    QO = np.linalg.inv(J.T) @ QC                                # p.32 inverted
    MO = np.linalg.inv(J.T) @ M3 @ np.linalg.inv(J)             # p.34
    cC = cvec_num(M_, Ig, xg_, yg_, z)
    cO_right = np.linalg.inv(J.T) @ (cC - M3 @ np.linalg.inv(J) @ Jd @ np.linalg.inv(J) @ xi_d)      # corrected
    cO_slide = np.linalg.inv(J.T) @ (cC - M3 @ Jd @ np.linalg.inv(J) @ xi_d)                       # as printed on p.34
    ok_all &= np.allclose(MO @ xi_dd + cO_right, QO, atol=1e-9)
    bad_slide |= not np.allclose(MO @ xi_dd + cO_slide, QO, atol=1e-6)
    ok_all &= abs(QO @ xi_d - QC @ z) < 1e-9                    # power invariance
check('inertial form with M_O=(J^T)^-1 M J^-1 and c_O=(J^T)^-1[c - M J^-1 Jdot J^-1 xi_dot] reproduces Q_O; power invariant (5 random cases)', ok_all)
check('p.34 line 3 as printed (M Jdot J^-1 xi_dot, no leading J^-1) does NOT reproduce Q_O', bad_slide)
# p.30 line 3 is right: zeta_dot = J^-1 [xi_dd - Jdot J^-1 xi_d]
th_, z, xi_dd = 0.7, np.array([1.0, 0.3, 0.8]), np.array([0.2, -0.5, 1.1])
J, Jd = Jmat(th_), Jdot(th_, z[2])
close('p.30: zeta_dot two forms agree', np.linalg.inv(J) @ (xi_dd - Jd @ z), np.linalg.inv(J) @ (xi_dd - Jd @ np.linalg.inv(J) @ (J @ z)))
# C = G: M_O = diag(m, m, I) and c_O = 0
M_, Ig = 10.0, 0.25
for th_ in (0.0, 0.5236, 2.0):
    J, Jd = Jmat(th_), Jdot(th_, 0.9)
    z = np.array([1.2, -0.4, 0.9]); xi_d = J @ z
    M3 = Mnum(M_, Ig, 0, 0)
    MO = np.linalg.inv(J.T) @ M3 @ np.linalg.inv(J)
    cO = np.linalg.inv(J.T) @ (cvec_num(M_, Ig, 0, 0, z) - M3 @ np.linalg.inv(J) @ Jd @ np.linalg.inv(J) @ xi_d)
    close(f'C=G: M_O = diag(m,m,I) at theta={th_:.4f}', MO, np.diag([M_, M_, Ig]))
    close(f'C=G: c_O = 0 at theta={th_:.4f}', cO, np.zeros(3))
# worked inertial-frame example at theta = 30 deg, at rest
J30 = Jmat(30*deg)
QO30 = np.linalg.inv(J30.T) @ Q
close('Q_O = J30 Q_C = (8.660, 5.000, 0.4)', QO30, [8.660254, 5.0, 0.4], tol=1e-5)
close('xi_dd = Q_O/m (xy), Q_O,z/I', np.linalg.solve(np.diag([m_, m_, I_]), QO30), [0.8660254, 0.5, 1.6], tol=1e-6)
out['inertial30'] = dict(QO=QO30.tolist(), xi_dd=np.linalg.solve(np.diag([m_, m_, I_]), QO30).tolist())

# ================= 6. Swedish wheel illustration (my assumption: wheels at (+/-b,+/-a), 45 deg) =================
s = 1/np.sqrt(2)
aa, bb = 0.3, 0.2
pos = [(bb, aa), (-bb, aa), (-bb, -aa), (bb, -aa)]
dirs = [(s, s), (s, -s), (s, s), (s, -s)]
cols = [[d[0], d[1], p[0]*d[1] - p[1]*d[0]] for p, d in zip(pos, dirs)]
Bs = np.array(cols).T
close('Swedish illustration third row = s(b-a), s(b-a), s(a-b), s(a-b)', Bs[2], [s*(bb-aa), s*(bb-aa), s*(aa-bb), s*(aa-bb)])
check('Swedish illustration: 3x4 with rank 3', np.linalg.matrix_rank(Bs) == 3)
check('Swedish illustration: one-dimensional null space (internal forces)', 4 - np.linalg.matrix_rank(Bs) == 1)
nsp = np.linalg.svd(Bs)[2][-1]
close('null space direction gives zero net wrench', Bs @ nsp, np.zeros(3))
close('equal forces F on all four: f_x = 4 s F, f_y = 0, n_z = 0', Bs @ np.ones(4), [4*s, 0, 0])

# ================= 7. Widget simulation (RK4, dt = 5 ms): test vectors =================
DT = 0.005
def deriv(S, tl, tr, grip, p=(10.0, 0.25, 0.1, 0.2)):
    mm, II, rr, aa_ = p
    x, y, th_, u_, v_, w_ = S
    fx_, nz_ = (tl + tr)/rr, aa_*(tr - tl)/rr
    return np.array([u_*np.cos(th_) - v_*np.sin(th_), u_*np.sin(th_) + v_*np.cos(th_), w_,
                     fx_/mm + (0 if grip else v_*w_), 0 if grip else -u_*w_, nz_/II])
def rk4(S, tl, tr, grip, dt=DT):
    k1 = deriv(S, tl, tr, grip); k2 = deriv(S + dt/2*k1, tl, tr, grip)
    k3 = deriv(S + dt/2*k2, tl, tr, grip); k4 = deriv(S + dt*k3, tl, tr, grip)
    return S + dt/6*(k1 + 2*k2 + 2*k3 + k4)
def simulate(tl, tr, grip, T):
    S = np.zeros(6)
    for _ in range(int(round(T/DT))):
        S = rk4(S, tl, tr, grip)
    return S
def pack(S, tl, tr, grip, T):
    x, y, th_, u_, v_, w_ = S
    return dict(tauL=tl, tauR=tr, grip=grip, T=T, x=x, y=y, theta=th_, u=u_, v=v_, thetadot=w_,
                phiL=(u_ - 0.2*w_)/0.1, phiR=(u_ + 0.2*w_)/0.1)
cases = [(0.4, 0.6, True), (0.4, 0.6, False), (0.3, 0.3, True), (-0.2, 0.2, True)]
vecs = []
for tl, tr, gr in cases:
    S = simulate(tl, tr, gr, 2.0)
    vecs.append(pack(S, tl, tr, gr, 2.0))
    print('   sim', tl, tr, gr, np.round(S, 6).tolist())
# (a) grip on, constant torques: exact circle of radius R, x = R sin(theta), y = R (1 - cos(theta))
S = simulate(0.4, 0.6, True, 2.0)
Tt = 2.0
close('grip on: u(2 s) = udot*t = 2.0', S[3], 2.0)
close('grip on: thetadot(2 s) = 3.2', S[5], 3.2)
close('grip on: theta(2 s) = 0.5*1.6*4 = 3.2 rad', S[2], 3.2, tol=1e-9)
close('grip on: x = R sin(theta) (analytic circle)', S[0], 0.625*np.sin(S[2]), tol=1e-7)
close('grip on: y = R (1 - cos(theta)) (analytic circle)', S[1], 0.625*(1 - np.cos(S[2])), tol=1e-7)
out_circle = dict(R=0.625, x=S[0], y=S[1])
# (b) RK4(5 ms) vs high-accuracy solve_ivp for all four cases
for tl, tr, gr in cases:
    sol = solve_ivp(lambda tt, S_: deriv(S_, tl, tr, gr), (0, 2), np.zeros(6), rtol=1e-11, atol=1e-12, method='DOP853')
    close(f'RK4 5 ms vs DOP853, tauL={tl}, tauR={tr}, grip={gr}', simulate(tl, tr, gr, 2.0), sol.y[:, -1], tol=1e-7, show=False)
# (c) equal torques: straight line, u = (F_L+F_R)/m t; opposite: spin in place
close('equal torques 0.3/0.3: u(2 s) = 1.2, theta stays 0', [vecs[2]['u'], vecs[2]['theta']], [1.2, 0.0])
close('opposite -0.2/+0.2: u = 0, thetadot(2 s) = 6.4 rad/s', [vecs[3]['u'], vecs[3]['thetadot']], [0.0, 6.4])
close('opposite torques: robot stays at the origin', [vecs[3]['x'], vecs[3]['y']], [0, 0], tol=1e-9)
# (d) grip off: v drifts, vdot = -u w; check initial rate and sign
Sg = simulate(0.4, 0.6, False, 2.0)
check('grip off: lateral velocity v(2 s) is negative (robot slides to its right while turning left)', Sg[4] < -0.5, f'v={Sg[4]:.4f}')
out['sim'] = vecs
out['circle'] = out_circle

# ================= 8. Guided-problem numbers =================
# P11.1: m=8, I=0.2, r=0.08, a=0.15, tauL=0.3, tauR=0.5
m2, I2, r2, a2, tl2, tr2 = 8.0, 0.2, 0.08, 0.15, 0.3, 0.5
Fl2, Fr2 = tl2/r2, tr2/r2
fx2, nz2 = Fl2 + Fr2, a2*(Fr2 - Fl2)
close('GP1: F_L, F_R = 3.75, 6.25 N', [Fl2, Fr2], [3.75, 6.25])
close('GP1: f_x = 10 N, n_z = 0.375 N m', [fx2, nz2], [10.0, 0.375])
close('GP1: udot = 1.25, thetadd = 1.875', [fx2/m2, nz2/I2], [1.25, 1.875])
close('GP1: R = 0.6667 m', (fx2/m2)/(nz2/I2), 2/3)
# P11.3: lateral force: m=10, u=2, w=0.5
close('GP3: f_y = m u w = 10 N', 10*2*0.5, 10.0)
# P11.4 offset
close('GP4: vdot = -0.08, thetadd = 1.6', [zo[1], zo[2]], [-0.08, 1.6])
# mini-quiz 2: equal torques 0.5 N m each, r=0.1, m=10
Fq = 0.5/0.1
close('Q2: F = 5 N each, udot = 1.0, thetadd = 0', [(2*Fq)/10, 0.2*(Fq - Fq)/0.25], [1.0, 0.0])
# GP7: inertial wrench at theta = 90 deg for Q_C = (10, 0, 0.4)
close('GP7: Q_O at 90 deg = (0, 10, 0.4)', Jmat(np.pi/2) @ Q, [0, 10, 0.4], tol=1e-9)


# ================= 9. Extra numbers used in worked examples =================
close('N_z = n_z + y f_x - x f_y, x_G=0.05, f=(10,4), n_z=0.4 -> 0.2 N m', 0.4 + (0*10 - 0.05*4), 0.2)
close('offset COM with u=2, thetadot=0.5 only: (xdd_G, ydd_G) = (-x_G w^2, u w) = (-0.0125, 1.0)', [-0.05*0.5**2, 2*0.5], [-0.0125, 1.0])
th_, z = np.pi/2, np.array([1.0, 0.0, 2.0])
J, Jd = Jmat(th_), Jdot(th_, 2.0); xi_d = J @ z; M3 = np.diag([10.0, 10.0, 0.25]); cC = np.array([0.0, 20.0, 0.0])
close('p.34 check case (C=G, theta=90 deg, u=1, v=0, w=2): correct c_O = 0',
      np.linalg.inv(J.T) @ (cC - M3 @ np.linalg.inv(J) @ Jd @ np.linalg.inv(J) @ xi_d), np.zeros(3), tol=1e-9)
close('p.34 check case: printed formula gives c_O = (-20, 20, 0)',
      np.linalg.inv(J.T) @ (cC - M3 @ Jd @ np.linalg.inv(J) @ xi_d), [-20.0, 20.0, 0.0], tol=1e-9)
close('GP1 wrong answers: no lever arm -> 12.5; lever 2a -> 3.75', [(Fr2-Fl2)/I2, (2*a2)*(Fr2-Fl2)/I2], [12.5, 3.75])
close('Section 8 P11.3 circle radius u/w = 4 m, m u^2/R = 10 N', [2/0.5, 10*2**2/(2/0.5)], [4.0, 10.0])

json.dump(out, open(os.path.join(os.path.dirname(__file__), 'w11_vectors.json'), 'w'), indent=1, default=float)
report()
