"""W13 revision questions Q1-Q10: verifies every model answer on the "Sample Q solved" tab.
Run: python3 site/src/checks/w13.py"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *

out = {}
t1, t2, t3 = sp.symbols('theta1 theta2 theta3', real=True)
L1, L2, L3, L4 = sp.symbols('L1 L2 L3 L4', positive=True)

# ===== Q1: FK of the RRR arm (lecturer's ETS, W13 p.4) =====
T_ST = chain(sRz(t1), stz(L1 + L2), sRy(-t2), stx(L3), sRy(-t3), stx(L4), sRx(sp.pi/2))
T_ST = sp.simplify(T_ST)
p = [sp.simplify(sp.trigsimp(e)) for e in T_ST[:3, 3]]
print('Q1 position of {T} in {S}:', p)
c23 = sp.cos(t2 + t3); s23 = sp.sin(t2 + t3)
r_expr = L3*sp.cos(t2) + L4*c23
check('Q1: x = cos(t1)(L3 c2 + L4 c23)', sp.simplify(p[0] - sp.cos(t1)*r_expr) == 0)
check('Q1: y = sin(t1)(L3 c2 + L4 c23)', sp.simplify(p[1] - sp.sin(t1)*r_expr) == 0)
check('Q1: z = L1+L2 + L3 s2 + L4 s23 (matches lecturer IK, p.7)', sp.simplify(p[2] - (L1 + L2 + L3*sp.sin(t2) + L4*s23)) == 0)
Rst = T_ST[:3, :3]
print('Q1 rotation S_R_T =')
sp.pprint(sp.simplify(Rst))
# home pose: tool frame axes: x_T = x_S, y_T = z_S, z_T = -y_S  (figure: Y_T up)
R0 = np.array(Rst.subs({t1: 0, t2: 0, t3: 0}), dtype=float)
close('Q1: at home, x_T=x_S, y_T=z_S, z_T=-y_S', R0, [[1, 0, 0], [0, 0, -1], [0, 1, 0]])
# intermediate frames (one frame per joint)
T_S1 = sRz(t1); T_12 = stz(L1 + L2) * sRy(-t2); T_23 = stx(L3) * sRy(-t3); T_3T = stx(L4) * sRx(sp.pi/2)
check('Q1: S_T_1 1_T_2 2_T_3 3_T_T == full ETS', sp.simplify(T_S1*T_12*T_23*T_3T - T_ST) == sp.zeros(4, 4))
num = {L1: 0.6, L2: 0.5, L3: 0.4, L4: 0.3}
qn = (30*deg, 20*deg, 40*deg)
Tn = np.array(T_ST.subs(num).subs({t1: qn[0], t2: qn[1], t3: qn[2]}).evalf(), dtype=float)
print('Q1 numeric (L=0.6,0.5,0.4,0.3; θ=30,20,40°):\n', np.round(Tn, 4))
check('Q1 numeric: R proper', is_rotation(Tn[:3, :3]))
out['q1'] = dict(L=[0.6, 0.5, 0.4, 0.3], th_deg=[30, 20, 40], T=np.round(Tn, 4).tolist())

# ===== Q3: IK (all branches) and round trip =====
def fk_num(L, th):
    l1, l2, l3, l4 = L
    rr = l3*np.cos(th[1]) + l4*np.cos(th[1]+th[2])
    return np.array([np.cos(th[0])*rr, np.sin(th[0])*rr, l1+l2 + l3*np.sin(th[1]) + l4*np.sin(th[1]+th[2])])

def ik_num(L, P):
    l1, l2, l3, l4 = L
    x, y, z = P
    sols = []
    for s1 in (+1, -1):                    # shoulder: facing the point (r>0) or reaching over the top (r<0)
        th1 = np.arctan2(s1*y, s1*x)
        rr = s1*np.hypot(x, y)
        zz = z - (l1 + l2)
        c3 = (rr**2 + zz**2 - l3**2 - l4**2) / (2*l3*l4)
        if abs(c3) > 1: continue
        for s3 in (+1, -1):                # elbow branch
            th3 = np.arctan2(s3*np.sqrt(1 - c3**2), c3)
            th2 = np.arctan2(zz, rr) - np.arctan2(l4*np.sin(th3), l3 + l4*np.cos(th3))
            sols.append((th1, th2, th3))
    return sols

Ln = (0.6, 0.5, 0.4, 0.3)
P = fk_num(Ln, qn)
print('Q3 target from Q1 numbers: P =', np.round(P, 4))
sols = ik_num(Ln, P)
check('Q3: four IK solutions for an interior point', len(sols) == 4)
ok = all(np.allclose(fk_num(Ln, s), P, atol=1e-12) for s in sols)
check('Q3: every IK branch reproduces P (IK->FK round trip)', ok)
check('Q3: original angles are among the solutions', any(np.allclose(s, qn) for s in sols))
for s in sols: print('   branch θ(deg) =', np.round(np.array(s)/deg, 3))
out['q3'] = dict(P=np.round(P, 4).tolist(), sols_deg=[np.round(np.array(s)/deg, 3).tolist() for s in sols])
# random round trips
rng = np.random.default_rng(1); okr = True
for _ in range(200):
    th = rng.uniform(-np.pi, np.pi, 3); Pp = fk_num(Ln, th)
    okr &= all(np.allclose(fk_num(Ln, s), Pp, atol=1e-9) for s in ik_num(Ln, Pp)) and len(ik_num(Ln, Pp)) >= 2
check('Q3: 200 random poses, all branches round-trip', okr)

# ===== Q2: workspace = spherical shell about the shoulder point =====
pts = np.array([fk_num(Ln, th) for th in rng.uniform(-np.pi, np.pi, (20000, 3))])
dist = np.linalg.norm(pts - np.array([0, 0, Ln[0] + Ln[1]]), axis=1)
rin, rout = abs(Ln[2] - Ln[3]), Ln[2] + Ln[3]
check('Q2: all reachable points lie in the shell |L3-L4| <= |P - shoulder| <= L3+L4', dist.min() >= rin - 1e-9 and dist.max() <= rout + 1e-9,
      f'min {dist.min():.4f} (bound {rin}), max {dist.max():.4f} (bound {rout})')
# distance depends only on theta3: |P-sh|^2 = L3^2+L4^2+2L3L4 cos(theta3)
th = rng.uniform(-np.pi, np.pi, (50, 3))
d2 = np.array([np.sum((fk_num(Ln, t) - [0, 0, 1.1])**2) for t in th])
close('Q2: |P-shoulder|^2 = L3²+L4²+2L3L4cosθ3', d2, Ln[2]**2 + Ln[3]**2 + 2*Ln[2]*Ln[3]*np.cos(th[:, 2]), show=False)
out['q2'] = dict(centre=[0, 0, Ln[0] + Ln[1]], r_in=rin, r_out=rout)

# ===== Q4: equations of motion (Lagrange), general link model =====
t = sp.symbols('t')
q = [sp.Function(f'q{i}')(t) for i in (1, 2, 3)]
qd = [sp.diff(qi, t) for qi in q]
m2, m3, l2, l3, I1, I2, I3, g = sp.symbols('m2 m3 l2 l3 I1 I2 I3 g', positive=True)
# frames along the chain
T1 = sRz(q[0]) * stz(L1 + L2)
T2 = T1 * sRy(-q[1])                  # link 2 frame at shoulder, x along link 2
T3 = T2 * stx(L3) * sRy(-q[2])        # link 3 frame at elbow, x along link 3
pc2 = (T2 * sp.Matrix([l2, 0, 0, 1]))[:3, 0]
pc3 = (T3 * sp.Matrix([l3, 0, 0, 1]))[:3, 0]
def omega(T):
    R = T[:3, :3]; W = sp.simplify(sp.diff(R, t) * R.T)
    return sp.Matrix([W[2, 1], W[0, 2], W[1, 0]])
w2, w3 = omega(T2), omega(T3)
Irod = lambda I: sp.diag(0, I, I)     # slender link along its own x-axis (inertia about COM)
def rotKE(T, w, I):
    R = T[:3, :3]; return sp.Rational(1, 2) * (w.T * R * Irod(I) * R.T * w)[0]
v2 = sp.diff(pc2, t); v3 = sp.diff(pc3, t)
KE = (sp.Rational(1, 2)*I1*qd[0]**2 + sp.Rational(1, 2)*m2*(v2.T*v2)[0] + rotKE(T2, w2, I2)
      + sp.Rational(1, 2)*m3*(v3.T*v3)[0] + rotKE(T3, w3, I3))
PE = m2*g*pc2[2] + m3*g*pc3[2]
Lag = KE - PE
tau = [sp.simplify(sp.diff(sp.diff(Lag, qd[i]), t) - sp.diff(Lag, q[i])) for i in range(3)]
# extract M, G
qdd = [sp.diff(qi, t, 2) for qi in q]
M = sp.Matrix(3, 3, lambda i, j: sp.simplify(sp.expand(tau[i]).coeff(qdd[j])))
G = sp.Matrix([sp.simplify(tau[i].subs({**{qdd[j]: 0 for j in range(3)}}).subs({qd[j]: 0 for j in range(3)})) for i in range(3)])
c2, c3_, c23 = sp.cos(q[1]), sp.cos(q[2]), sp.cos(q[1] + q[2])
M_hand = sp.Matrix([
    [I1 + m2*(l2*c2)**2 + m3*(L3*c2 + l3*c23)**2 + I2*c2**2 + I3*c23**2, 0, 0],
    [0, m2*l2**2 + I2 + m3*(L3**2 + l3**2 + 2*L3*l3*c3_) + I3, m3*(l3**2 + L3*l3*c3_) + I3],
    [0, m3*(l3**2 + L3*l3*c3_) + I3, m3*l3**2 + I3]])
check('Q4: M(q) from Lagrange == hand form', sp.simplify(M - M_hand) == sp.zeros(3, 3))
G_hand = sp.Matrix([0, g*((m2*l2 + m3*L3)*c2 + m3*l3*c23), g*m3*l3*c23])
check('Q4: G(q) == hand form', sp.simplify(G - G_hand) == sp.zeros(3, 3)[:, 0])
# Coriolis/centrifugal V = tau - M qdd - G
V = sp.simplify(sp.Matrix(tau) - M*sp.Matrix(qdd) - G)
print('Q4 V(q,qd) =')
for i in range(3): print(f'   V{i+1} =', sp.factor(sp.simplify(V[i])))
# hand form of V
s2, s23, s3 = sp.sin(q[1]), sp.sin(q[1]+q[2]), sp.sin(q[2])
r2, r3 = l2*c2, L3*c2 + l3*c23
d1, d2_, d3 = qd
V_hand = sp.Matrix([
    -2*d1*(m2*r2*l2*s2*d2_ + m3*r3*(L3*s2*d2_ + l3*s23*(d2_ + d3)) + I2*c2*s2*d2_ + I3*c23*s23*(d2_ + d3)),
    d1**2*(m2*r2*l2*s2 + m3*r3*(L3*s2 + l3*s23) + I2*c2*s2 + I3*c23*s23) - m3*L3*l3*s3*(2*d2_*d3 + d3**2),
    d1**2*(m3*r3*l3*s23 + I3*c23*s23) + m3*L3*l3*s3*d2_**2])
check('Q4: V(q,qd) == hand form', sp.simplify(V - V_hand) == sp.zeros(3, 1))
# properties: M symmetric PD; Mdot - 2C skew (use Christoffel C)
qs = sp.symbols('a1 a2 a3'); qds = sp.symbols('b1 b2 b3')
sub = {**{q[i]: qs[i] for i in range(3)}}
Ms = M.subs(sub)
C = sp.zeros(3, 3)
for k in range(3):
    for j in range(3):
        C[k, j] = sum(sp.Rational(1, 2)*(sp.diff(Ms[k, j], qs[i]) + sp.diff(Ms[k, i], qs[j]) - sp.diff(Ms[i, j], qs[k]))*qds[i] for i in range(3))
Mdot = sp.Matrix(3, 3, lambda i, j: sum(sp.diff(Ms[i, j], qs[k])*qds[k] for k in range(3)))
check('Q4: Mdot - 2C is skew-symmetric', sp.simplify(Mdot - 2*C + (Mdot - 2*C).T) == sp.zeros(3, 3))
Vs = V.subs({qd[i]: qds[i] for i in range(3)}).subs(sub)
check('Q4: V == C(q,qd) qd (Christoffel)', sp.simplify(Vs - C*sp.Matrix(qds)) == sp.zeros(3, 1))
numv = {m2: 2.0, m3: 1.0, l2: 0.2, l3: 0.15, I1: 0.05, I2: 0.03, I3: 0.01, g: 9.81, L3: 0.4, L1: 0.6, L2: 0.5}
Mn = np.array(Ms.subs(numv).subs({qs[0]: 0.3, qs[1]: 0.4, qs[2]: -0.7}), dtype=float)
check('Q4: M symmetric positive-definite at a test pose', np.allclose(Mn, Mn.T) and np.all(np.linalg.eigvalsh(Mn) > 0))
# energy conservation in free motion (tau=0) — numeric integration
f_M = sp.lambdify([qs], Ms.subs(numv), 'numpy')
f_V = sp.lambdify([qs, qds], Vs.subs(numv), 'numpy')
f_G = sp.lambdify([qs], G.subs(sub).subs(numv), 'numpy')
KEs = KE.subs({qd[i]: qds[i] for i in range(3)}).subs(sub).subs(numv)
PEs = PE.subs(sub).subs(numv)
f_E = sp.lambdify([qs, qds], KEs + PEs, 'numpy')
from scipy.integrate import solve_ivp
def rhs(_, s):
    qq, vv = s[:3], s[3:]
    a = np.linalg.solve(np.array(f_M(qq), float), -np.array(f_V(qq, vv), float).ravel() - np.array(f_G(qq), float).ravel())
    return np.concatenate([vv, a])
s0 = np.array([0.2, 0.5, -0.3, 1.0, 0.0, 0.5])
sol = solve_ivp(rhs, [0, 3], s0, rtol=1e-10, atol=1e-10)
E = [f_E(sol.y[:3, k], sol.y[3:, k]) for k in range(sol.y.shape[1])]
check('Q4: total energy conserved in free motion (3 s)', np.ptp(E) < 1e-6, f'drift {np.ptp(E):.2e} J')
out['q4'] = 'symbolic (see tab)'

# ===== Q5: four-bar IK (W13 pp. 9-11) =====
L0n, L1n, L3n, a_, b_ = 1.0, 0.5, 0.6, 0.2, 0.25
L2n = a_ + b_
# build a consistent configuration: choose theta1, solve closure for phi (coupler angle)
th1 = 70*deg
A = L1n*np.array([np.cos(th1), np.sin(th1)]); D = np.array([L0n, 0])
# B on circle(A, L2) and circle(D, L3): two assembly modes
dAD = np.linalg.norm(D - A); aa = (L2n**2 - L3n**2 + dAD**2)/(2*dAD); hh = np.sqrt(L2n**2 - aa**2)
mid = A + aa*(D - A)/dAD; perp = np.array([-(D - A)[1], (D - A)[0]])/dAD
Bs = [mid + hh*perp, mid - hh*perp]
B = max(Bs, key=lambda v: v[1])        # "elbow up" assembly as drawn
phi = np.arctan2(B[1] - A[1], B[0] - A[0])
K = A + a_*np.array([np.cos(phi), np.sin(phi)])
print(f'Q5 data: K=({K[0]:.4f}, {K[1]:.4f}), phi={phi/deg:.4f} deg')
# IK as on the slides
th1_ik = np.arctan2(K[1] - a_*np.sin(phi), K[0] - a_*np.cos(phi))
ph2 = (phi - th1_ik) % (2*np.pi)
Bik = K + b_*np.array([np.cos(phi), np.sin(phi)])
ph1 = np.arctan2(Bik[1], Bik[0] - L0n)
ph3 = (phi + np.pi - ph1) % (2*np.pi)
close('Q5: theta1 recovered', th1_ik, th1)
check('Q5: |B - D| = L3 (data consistent)', abs(np.linalg.norm(Bik - D) - L3n) < 1e-12)
xr = L0n + L3n*np.cos(ph1) + b_*np.cos(ph1 + ph3); yr = L3n*np.sin(ph1) + b_*np.sin(ph1 + ph3)
close('Q5: slide position equation (p.11) satisfied with phi3 = phi + pi - phi1', [xr, yr], K)
ph3_slide = (ph1 - phi + np.pi) % (2*np.pi)
xs = L0n + L3n*np.cos(ph1) + b_*np.cos(ph1 + ph3_slide)
check('Q5: the printed relation phi1 - phi = phi3 - pi does NOT satisfy the p.11 position equation here', abs(xs - K[0]) > 1e-3)
xl = L1n*np.cos(th1_ik) + a_*np.cos(th1_ik + ph2); yl = L1n*np.sin(th1_ik) + a_*np.sin(th1_ik + ph2)
close('Q5: left-chain equation (p.10) satisfied', [xl, yl], K)
print(f'   θ1={th1_ik/deg:.4f}°, φ1={ph1/deg:.4f}°, φ2={ph2/deg:.4f}°, φ3={ph3/deg:.4f}°')
out['q5'] = dict(L0=L0n, L1=L1n, L3=L3n, a=a_, b=b_, K=np.round(K, 4).tolist(), phi_deg=r(phi/deg),
                 th1_deg=r(th1_ik/deg), ph1_deg=r(ph1/deg), ph2_deg=r(ph2/deg), ph3_deg=r(ph3/deg))

# ===== Q6, Q7: mobility =====
GK = lambda D_, N, J, F, Rc=0, Rm=0: D_*(N - J - 1) + F + Rc - Rm
check('Q6: planar 3-PRR, M = 3(8-9-1)+9 = 3', GK(3, 8, 9, 9) == 3)
check('Q7: delta, plain G-K: 6(11-15-1)+39 = 9', GK(6, 11, 15, 3*(1 + 4*3)) == 9)
check('Q7: delta, extended (Rc=0, Rm=6): 3', GK(6, 11, 15, 39, 0, 6) == 3)
# link/joint bookkeeping for the delta: base + 3 upper arms + 6 rods + platform
check('Q7: N = 1 + 3 + 6 + 1 = 11; J = 3 R + 12 S = 15', (1 + 3 + 6 + 1, 3 + 12) == (11, 15))

# ===== Q8: two-segment cubic spline with continuous acceleration =====
T1s, T2s, th0, thv, thg = sp.symbols('T1 T2 theta0 theta_v theta_g', positive=True)
a10, a11, a12, a13, a20, a21, a22, a23 = sp.symbols('a10 a11 a12 a13 a20 a21 a22 a23')
tt = sp.symbols('t')
p1 = a10 + a11*tt + a12*tt**2 + a13*tt**3
p2 = a20 + a21*tt + a22*tt**2 + a23*tt**3
eqs = [p1.subs(tt, 0) - th0, sp.diff(p1, tt).subs(tt, 0), p1.subs(tt, T1s) - thv, p2.subs(tt, 0) - thv,
       p2.subs(tt, T2s) - thg, sp.diff(p2, tt).subs(tt, T2s),
       sp.diff(p1, tt).subs(tt, T1s) - sp.diff(p2, tt).subs(tt, 0),
       sp.diff(p1, tt, 2).subs(tt, T1s) - sp.diff(p2, tt, 2).subs(tt, 0)]
solq8 = sp.solve(eqs, [a10, a11, a12, a13, a20, a21, a22, a23], dict=True)[0]
Tf = sp.symbols('t_f', positive=True)
eqT = {k: sp.factor(sp.simplify(v.subs({T1s: Tf, T2s: Tf}))) for k, v in solq8.items()}
craig = {a10: th0, a11: 0, a12: (12*thv - 3*thg - 9*th0)/(4*Tf**2), a13: (-8*thv + 3*thg + 5*th0)/(4*Tf**3),
         a20: thv, a21: (3*thg - 3*th0)/(4*Tf), a22: (-12*thv + 6*thg + 6*th0)/(4*Tf**2), a23: (8*thv - 5*thg - 3*th0)/(4*Tf**3)}
check('Q8: equal-duration solution matches the closed form', all(sp.simplify(eqT[k] - craig[k]) == 0 for k in craig))
for k in [a10, a11, a12, a13, a20, a21, a22, a23]: print(f'   {k} =', eqT[k])
# numeric example: theta0=10°, thv=40°, thg=20°, tf=1 s each
nq8 = {th0: 10, thv: 40, thg: 20, Tf: 1}
out['q8'] = {str(k): float(sp.sympify(craig[k]).subs(nq8)) for k in craig}
print('   numeric (10°, 40°, 20°, tf=1 s):', {k: round(v, 4) for k, v in out['q8'].items()})
vv = float((3*thg - 3*th0)/(4*Tf)).__class__ if False else float(craig[a21].subs(nq8))
check('Q8 numeric: via velocity = 3(θg-θ0)/(4tf) = 7.5 °/s', abs(vv - 7.5) < 1e-12)

# ===== Q9: prismatic cubic =====
d0, dfin, tf9 = 0.02, 0.15, 0.8
a0 = d0; a1 = 0; a2 = 3*(dfin - d0)/tf9**2; a3 = -2*(dfin - d0)/tf9**3
close('Q9: coefficients', [a0, a1, a2, a3], [0.02, 0, 0.609375, -0.5078125])
dd_ = lambda tt_: a0 + a1*tt_ + a2*tt_**2 + a3*tt_**3
vel = lambda tt_: a1 + 2*a2*tt_ + 3*a3*tt_**2
close('Q9: boundary conditions d(0), d(tf), v(0), v(tf)', [dd_(0), dd_(tf9), vel(0), vel(tf9)], [0.02, 0.15, 0, 0])
vmax = vel(tf9/2); acc0 = 2*a2
print(f'   Q9 peak velocity at t=0.4 s: {vmax:.6f} m/s (= 1.5·Δd/tf); initial accel {acc0:.5f} m/s²')
close('Q9: peak velocity = 1.5 Δd / tf', vmax, 1.5*(dfin - d0)/tf9)
out['q9'] = dict(a=[a0, a1, a2, a3], vmax=r(vmax, 6), acc0=r(acc0, 6))

# ===== Q10: three-wheel omni robot =====
d_, r_ = sp.symbols('d r', positive=True)
u, v, w = sp.symbols('u v thetadot', real=True)
alphas = [sp.pi/2, -sp.pi/6, 7*sp.pi/6]        # wheel positions (W1 top, W2 lower right, W3 lower left)
betas = [0, -2*sp.pi/3, 2*sp.pi/3]             # drive direction of each wheel (x of {W_i}); clockwise-tangent convention
rows = []
for al_, be in zip(alphas, betas):
    xw, yw = d_*sp.cos(al_), d_*sp.sin(al_)
    gen = (sp.Matrix([[1/r_, 0]]) * sp.Matrix([[sp.cos(be), sp.sin(be)], [-sp.sin(be), sp.cos(be)]])
           * sp.Matrix([[1, 0, -yw], [0, 1, xw]]) * sp.Matrix([u, v, w]))[0]
    # direct rigid-body check: contact-point velocity projected on drive direction / r
    vc = sp.Matrix([u - w*yw, v + w*xw]); drive = sp.Matrix([sp.cos(be), sp.sin(be)])
    check(f'Q10: generalised model (p.21, γ=0) == rigid-body projection for wheel at {al_}', sp.simplify(gen - (vc.T*drive)[0]/r_) == 0)
    rows.append([sp.simplify(sp.diff(gen, s)*r_) for s in (u, v, w)])
Jw = sp.Matrix(rows)
print('Q10: r·φ̇ = J [u v θ̇]^T with J =', Jw.tolist())
check('Q10: J == [[1,0,-d],[-1/2,-√3/2,-d],[-1/2,√3/2,-d]]',
      sp.simplify(Jw - sp.Matrix([[1, 0, -d_], [-sp.Rational(1, 2), -sp.sqrt(3)/2, -d_], [-sp.Rational(1, 2), sp.sqrt(3)/2, -d_]])) == sp.zeros(3, 3))
check('Q10: J invertible (det != 0) -> holonomic, any (u,v,θ̇) achievable', sp.simplify(Jw.det()) != 0, f'det = {sp.simplify(Jw.det())}')
Jinv = sp.simplify(Jw.inv())
print('Q10: [u v θ̇]^T = r J^-1 φ̇ with J^-1 =', Jinv.tolist())
# numeric example: d=0.1 m, r=0.05 m, u=0.2, v=0, θ̇=0
phid = (Jw.subs({d_: 0.1}) * sp.Matrix([0.2, 0, 0]) / 0.05).evalf()
print('   example d=0.1, r=0.05, (u,v,θ̇)=(0.2,0,0): φ̇ =', [r(x_, 4) for x_ in phid])
out['q10'] = dict(J='[[1,0,-d],[-1/2,-sqrt3/2,-d],[-1/2,sqrt3/2,-d]]', example_phid=[r(x_, 4) for x_ in phid])

json.dump(out, open(os.path.join(os.path.dirname(__file__), 'w13_vectors.json'), 'w'), indent=1, default=str)
report()
