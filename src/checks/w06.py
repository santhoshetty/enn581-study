"""W6 Velocity kinematics: verifies every number on the W6 tab and the widget test vectors.
Run: python3 site/src/checks/w06.py   (prints PASS/FAIL, writes w06_vectors.json)
Method (lecture pp. 13-39): J columns by differentiating the ETS; ANALYTIC derivative per elementary transform,
checked against finite differences, against the geometric column rule and against Craig's outward recursion."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *

out = {}
rng = np.random.default_rng(6)
np.set_printoptions(precision=4, suppress=True)

# ====================================================================================
# 0. The lecture's method as code (mirrors the JS in frag/06-w06.html)
#    op = {'op': 'Rz', 'q': j, 'sgn': +-1, 'off': c}  (joint)   or   {'op': 'tx', 'v': c}  (constant)
# ====================================================================================
AX = {'Rx': 0, 'Ry': 1, 'Rz': 2, 'tx': 0, 'ty': 1, 'tz': 2}
def E(o, q):
    val = (o.get('sgn', 1) * q[o['q']] + o.get('off', 0)) if 'q' in o else o['v']
    return {'Rx': Rx, 'Ry': Ry, 'Rz': Rz, 'tx': tx, 'ty': ty, 'tz': tz}[o['op']](val)

# slide p.31 (rotations) and p.29 (translations): the constant matrices R-hat and t-hat
RHAT = {'Rx': np.array([[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0.]]),
        'Ry': np.array([[0, 0, 1, 0], [0, 0, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0.]]),
        'Rz': np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.]])}
def THAT(ax):
    m = np.zeros((4, 4)); m[ax, 3] = 1; return m
def dE(o, q):
    """dE/dq_j. For a joint whose positive sense is opposite to the axis (sgn = -1) the derivative changes sign."""
    s = o.get('sgn', 1)
    if o['op'][0] == 'R':
        return s * RHAT[o['op']] @ E(o, q)
    return s * THAT(AX[o['op']])

def fk(ops, q):
    T = np.eye(4)
    for o in ops: T = T @ E(o, q)
    return T

def vex(S): return np.array([S[2, 1], S[0, 2], S[1, 0]])
def skew(s): return np.array([[0, -s[2], s[1]], [s[2], 0, -s[0]], [-s[1], s[0], 0.]])

def n_joints(ops): return 1 + max(o['q'] for o in ops if 'q' in o)

def dT_dq(ops, j, q):
    """slide p.32: product of E_i before mu(j), dE_mu/dq_j, product of E_i after mu(j)"""
    T = np.eye(4)
    for o in ops:
        T = T @ (dE(o, q) if o.get('q') == j else E(o, q))
    return T

def jac0(ops, q):
    """slides pp.17-26: J_v col = tau(dT/dq_j), J_w col = vex(rho(dT/dq_j) rho(T)^T)"""
    n = n_joints(ops); T = fk(ops, q); R = T[:3, :3]
    J = np.zeros((6, n))
    for j in range(n):
        dT = dT_dq(ops, j, q)
        J[:3, j] = dT[:3, 3]
        J[3:, j] = vex(dT[:3, :3] @ R.T)
    return J

def jac_fd(ops, q, h=1e-6):
    n = n_joints(ops); T0 = fk(ops, q); J = np.zeros((6, n))
    for j in range(n):
        dq = np.zeros(n); dq[j] = h
        Tp, Tm = fk(ops, q + dq), fk(ops, q - dq)
        J[:3, j] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
        S = (Tp[:3, :3] - Tm[:3, :3]) / (2 * h) @ T0[:3, :3].T
        J[3:, j] = vex(S)
    return J

def jac_geom(ops, q):
    """added: column j = [z x (p_E - o); z] (revolute), [z; 0] (prismatic); z, o = axis and point of joint j in {0}"""
    n = n_joints(ops); T = np.eye(4); cols = {}
    frames = []
    for o in ops:
        if 'q' in o: frames.append((o, T.copy()))
        T = T @ E(o, q)
    pE = T[:3, 3]; J = np.zeros((6, n))
    for o, Tb in frames:
        j = o['q']; z = o.get('sgn', 1) * Tb[:3, AX[o['op']]]; p = Tb[:3, 3]
        if o['op'][0] == 'R': J[:3, j] = np.cross(z, pE - p); J[3:, j] = z
        else: J[:3, j] = z
    return J

def Rz2(a): return Rz(a)[:3, :3]

R_, T_ = 'R', 't'
def rot(op, j, sgn=1): return {'op': op, 'q': j, 'sgn': sgn}
def con(op, v): return {'op': op, 'v': v}

# ---------------- widget presets (identical to the JS) ----------------
L1w, L2w = 0.6, 0.4
PRESETS = {
    'rr': dict(ops=[rot('Rz', 0), con('tx', L1w), rot('Rz', 1), con('tx', L2w)], types='RR', rows=[0, 1]),
    'rp': dict(ops=[rot('Rz', 0), con('tx', 0.5), rot('ty', 1)], types='RP', rows=[0, 1]),
    'w13': dict(ops=[rot('Rz', 0), con('tz', 1.1), rot('Ry', 1, -1), con('tx', 0.4), rot('Ry', 2, -1), con('tx', 0.3), con('Rx', np.pi / 2)],
                types='RRR', rows=[0, 1, 2]),
}

# ====================================================================================
# 1. Cross product and skew matrix (cross-product note) + R-dot = [w]x R
# ====================================================================================
w_, P_ = sp.symbols('wx wy wz'), sp.symbols('Px Py Pz')
wv, Pv = sp.Matrix(w_), sp.Matrix(P_)
i_, j_, k_ = sp.symbols('i j k')
det_form = sp.Matrix([[i_, j_, k_], [w_[0], w_[1], w_[2]], [P_[0], P_[1], P_[2]]]).det()
coef = [sp.expand(det_form).coeff(s) for s in (i_, j_, k_)]
Sk = sp.Matrix([[0, -w_[2], w_[1]], [w_[2], 0, -w_[0]], [-w_[1], w_[0], 0]])
check('cross: determinant expansion == skew-matrix product (symbolic)', all(sp.expand(a - b) == 0 for a, b in zip(coef, Sk * Pv)))
check('cross: j-coefficient is wz*Px - wx*Pz (note prints wx*Pz - wz*Px: wrong sign)',
      sp.expand(coef[1] - (w_[2] * P_[0] - w_[0] * P_[2])) == 0 and sp.expand(coef[1] + (w_[0] * P_[2] - w_[2] * P_[0])) == 0)
w3, P3 = np.array([1, 2, 3.]), np.array([4, 5, 6.])
cr = np.cross(w3, P3)
close('cross example (1,2,3)x(4,5,6) = (-3,6,-3)', cr, [-3, 6, -3])
close('cross example: skew form agrees', skew(w3) @ P3, cr)
note_j = w3[0] * P3[2] - w3[2] * P3[0]
check('cross example: the note printed j-term would give -6 instead of +6', note_j == -6 and cr[1] == 6)
out['cross'] = dict(w=[1, 2, 3], P=[4, 5, 6], wxP=cr.tolist(), wrong_j=note_j)

# skew/vex round trip, and R-dot R^T is skew; Rz example
s_ = rng.normal(size=3); close('vex(skew(s)) = s', vex(skew(s_)), s_)
th = sp.symbols('th'); thd = sp.symbols('thd')
Rzs = sRz(th)[:3, :3]
Rdot = Rzs.diff(th) * thd
S = sp.simplify(Rdot * Rzs.T)
check('Rz: Rdot R^T = thd [z]x (symbolic)', sp.simplify(S - thd * sp.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])) == sp.zeros(3, 3))
# R R^T = I  =>  Rdot R^T skew, random path
def Rrand(t):
    return Rx(0.3 + t)[:3, :3] @ Ry(-0.7 + 2 * t)[:3, :3] @ Rz(1.1 - t)[:3, :3]
h = 1e-6
Rd = (Rrand(h) - Rrand(-h)) / (2 * h)
Sm = Rd @ Rrand(0).T
close('Rdot R^T is skew-symmetric (random rotation path)', Sm, -Sm.T, 1e-8)
# point velocity v = w x r
wv0 = np.array([0.0, 0.0, 2.0]); r0 = np.array([0.3, 0.4, 0])
close('point velocity w x r, w=(0,0,2), r=(0.3,0.4,0) = (-0.8,0.6,0)', np.cross(wv0, r0), [-0.8, 0.6, 0])

# elementary derivatives (slides pp.29, 31): analytic vs symbolic
a = sp.symbols('a')
for nm, fn in (('Rx', sRx), ('Ry', sRy), ('Rz', sRz)):
    lhs = fn(a).diff(a); rhs = sp.Matrix(RHAT[nm].tolist()) * fn(a)
    check(f'd{nm}/da = R-hat {nm} * {nm}(a) (p.31), symbolic', sp.simplify(lhs - rhs) == sp.zeros(4, 4))
for nm, fn, ax in (('tx', stx, 0), ('ty', sty, 1), ('tz', stz, 2)):
    check(f'd{nm}/dd = t-hat (p.29), symbolic', sp.simplify(fn(a).diff(a) - sp.Matrix(THAT(ax).tolist())) == sp.zeros(4, 4))
# sign flip: d/dq Ry(-q) = sgn * Rhat_y Ry(-q)
qq = 0.7
fd = (Ry(-(qq + 1e-6)) - Ry(-(qq - 1e-6))) / 2e-6
close('sgn=-1 rule: d Ry(-q)/dq = -Rhat_y Ry(-q) = Rhat_y^T Ry(-q)', fd, dE({'op': 'Ry', 'q': 0, 'sgn': -1}, [qq]), 1e-6)
close('sgn=-1 rule equals notebook form Rhat.T @ T.T', dE({'op': 'Ry', 'q': 0, 'sgn': -1}, [qq]), RHAT['Ry'].T @ Ry(qq).T)

# ====================================================================================
# 2. Planar RR: three routes agree (pp. 33-39, 45), notebook Q1
# ====================================================================================
q1s, q2s, l1s, l2s = sp.symbols('q1 q2 L1 L2', real=True)
Trr = chain(sRz(q1s), stx(l1s), sRz(q2s), stx(l2s))
xy = sp.Matrix([Trr[0, 3], Trr[1, 3]])
Jxy = xy.jacobian([q1s, q2s])            # route 1: differentiate FK (p.34)
hand = sp.Matrix([[-l1s * sp.sin(q1s) - l2s * sp.sin(q1s + q2s), -l2s * sp.sin(q1s + q2s)],
                  [l1s * sp.cos(q1s) + l2s * sp.cos(q1s + q2s), l2s * sp.cos(q1s + q2s)]])
check('RR: Jacobian from differentiating FK == hand formula (p.34)', sp.simplify(Jxy - hand) == sp.zeros(2, 2))
# route 2: ETS method, symbolic, translations tau and rotations vex(rho(dT) rho(T)^T)
RZh = sp.Matrix(RHAT['Rz'].tolist())
dT1 = RZh * sRz(q1s) * stx(l1s) * sRz(q2s) * stx(l2s)
dT2 = sRz(q1s) * stx(l1s) * RZh * sRz(q2s) * stx(l2s)
Rm = Trr[:3, :3]
Jv_ets = sp.Matrix.hstack(dT1[:3, 3], dT2[:3, 3])
def svex(S): return sp.Matrix([S[2, 1], S[0, 2], S[1, 0]])
Jw_ets = sp.Matrix.hstack(svex(dT1[:3, :3] * Rm.T), svex(dT2[:3, :3] * Rm.T))
check('RR: ETS method J_v rows x,y == differentiating FK', sp.simplify(Jv_ets[:2, :] - Jxy) == sp.zeros(2, 2))
check('RR: ETS method J_v z row = 0', sp.simplify(Jv_ets[2, :]) == sp.zeros(1, 2))
check('RR: ETS method J_w = (0,0,1) for both joints', sp.simplify(Jw_ets - sp.Matrix([[0, 0], [0, 0], [1, 1]])) == sp.zeros(3, 2))
detJ = sp.simplify(Jxy.det())
check('RR: det J = L1 L2 sin q2 (p.45)', sp.simplify(detJ - l1s * l2s * sp.sin(q2s)) == 0)
# frame {3} Jacobian (p.45): rotate rows by R^T
R03 = Trr[:3, :3]
J0 = sp.Matrix.vstack(Jv_ets, Jw_ets)
Rb = sp.diag(R03.T, R03.T)
J3 = sp.simplify(Rb * J0)
slideJ3 = sp.Matrix([[l1s * sp.sin(q2s), 0], [l1s * sp.cos(q2s) + l2s, l2s], [0, 0], [0, 0], [0, 0], [1, 1]])
check('RR: ^3J = diag(R^T,R^T) ^0J equals slide p.45 matrix', sp.simplify(J3 - slideJ3) == sp.zeros(6, 2))
check('RR: det of the 2x2 block is the same in {0} and {3}', sp.simplify(slideJ3[:2, :].det() - detJ) == 0)

# notebook Q1: L1=3, L2=4, q=(30,15) deg
rr_ops = lambda L1, L2: [rot('Rz', 0), con('tx', L1), rot('Rz', 1), con('tx', L2)]
qQ1 = np.array([30, 15]) * deg
JQ1 = jac0(rr_ops(3, 4), qQ1)
print('Q1 J0=\n', JQ1)
close('Q1: J0 first two rows by hand', JQ1[:2], [[-3 * .5 - 4 * np.sin(np.pi / 4), -4 * np.sin(np.pi / 4)], [3 * np.cos(np.pi / 6) + 4 * np.cos(np.pi / 4), 4 * np.cos(np.pi / 4)]])
close('Q1: rows 3,4,5 are zero, row 6 = (1,1)', JQ1[2:], [[0, 0], [0, 0], [0, 0], [1, 1]])
JQ1e = np.block([[Rz2(qQ1.sum()).T, np.zeros((3, 3))], [np.zeros((3, 3)), Rz2(qQ1.sum()).T]]) @ JQ1
close('Q1: J in end-effector frame = slide p.45 form',
      JQ1e[:2], [[3 * np.sin(qQ1[1]), 0], [3 * np.cos(qQ1[1]) + 4, 4]])
dQ1 = np.linalg.det(JQ1[:2])
check('Q1: det = L1 L2 sin q2 = 12 sin 15 deg = 3.1058', abs(dQ1 - 12 * np.sin(15 * deg)) < 1e-12 and abs(dQ1 - 3.1058) < 5e-5, f'{dQ1:.4f}')
print('Q1 numbers: J0 =', np.round(JQ1[:2], 4).tolist(), 'JE =', np.round(JQ1e[:2], 4).tolist(), 'det', round(dQ1, 4))
out['Q1'] = dict(L=[3, 4], q_deg=[30, 15], J0_xy=np.round(JQ1[:2], 4).tolist(), JE_xy=np.round(JQ1e[:2], 4).tolist(), det=r(dQ1))

# numeric tool velocity for Q1 with qdot = (0.2, -0.1) rad/s
qd = np.array([0.2, -0.1])
v0 = JQ1[:2] @ qd
print('Q1 v0 (qd=(0.2,-0.1)) =', v0, ' E-frame', JQ1e[:2] @ qd, ' omega_z', JQ1[5] @ qd)
out['Q1']['qdot'] = qd.tolist(); out['Q1']['v0'] = np.round(v0, 4).tolist(); out['Q1']['vE'] = np.round(JQ1e[:2] @ qd, 4).tolist(); out['Q1']['wz'] = r(JQ1[5] @ qd)
close('Q1: v in {0} rotated by R^T = v in {E}', Rz2(qQ1.sum()).T @ np.r_[v0, 0], np.r_[JQ1e[:2] @ qd, 0])
check('Q1: |v| same in both frames', abs(np.linalg.norm(v0) - np.linalg.norm(JQ1e[:2] @ qd)) < 1e-12)

# ====================================================================================
# 3. The lecture method == finite differences == geometric rule, for every preset + random arms
# ====================================================================================
vectors = {}
def vex_sigma(Jt):
    s = np.linalg.svd(Jt, compute_uv=False)
    return s
for key, P in PRESETS.items():
    ops, types, rows = P['ops'], P['types'], P['rows']
    n = n_joints(ops)
    worst_fd = worst_geo = 0
    for _ in range(40):
        q = rng.uniform(-3, 3, n)
        Ja = jac0(ops, q)
        worst_fd = max(worst_fd, np.abs(Ja - jac_fd(ops, q)).max())
        worst_geo = max(worst_geo, np.abs(Ja - jac_geom(ops, q)).max())
    check(f'{key}: analytic ETS Jacobian == central finite difference (40 random q), max err {worst_fd:.1e}', worst_fd < 5e-8)
    check(f'{key}: ETS Jacobian == geometric column rule (40 random q), max err {worst_geo:.1e}', worst_geo < 1e-10)

# Random 6-joint arms with mixed axes and sign flips
axes = ['Rx', 'Ry', 'Rz']
for trial in range(5):
    ops = []; j = 0
    for k in range(6):
        ops.append(con(rng.choice(['tx', 'ty', 'tz']), float(rng.uniform(0.1, 0.5))))
        if rng.random() < 0.25: ops.append(rot(rng.choice(['tx', 'ty', 'tz']), j, int(rng.choice([1, -1]))))
        else: ops.append(rot(rng.choice(axes), j, int(rng.choice([1, -1]))))
        j += 1
    q = rng.uniform(-3, 3, j)
    Ja = jac0(ops, q)
    check(f'random mixed 6-joint arm #{trial} (R and P, sign flips): ETS == FD == geometric',
          np.abs(Ja - jac_fd(ops, q)).max() < 1e-7 and np.abs(Ja - jac_geom(ops, q)).max() < 1e-9)

# ====================================================================================
# 4. W13 RRR arm: det, singularities (boundary / interior)
# ====================================================================================
ops13 = PRESETS['w13']['ops']; L3_, L4_ = 0.4, 0.3
def det13(q): return np.linalg.det(jac0(ops13, q)[:3])
def r13(q): return L3_ * np.cos(q[1]) + L4_ * np.cos(q[1] + q[2])
rel = []
for _ in range(30):
    q = rng.uniform(-3, 3, 3)
    rel.append(det13(q) - (-L3_ * L4_ * np.sin(q[2]) * r13(q)) )
sgn13 = None
q = np.array([0.4, 0.5, 0.9]); d_ = det13(q); f_ = L3_ * L4_ * np.sin(q[2]) * r13(q)
sgn13 = np.sign(d_ / f_)
for _ in range(30):
    q = rng.uniform(-3, 3, 3)
    check_ok = abs(det13(q) - sgn13 * L3_ * L4_ * np.sin(q[2]) * r13(q)) < 1e-10
    if not check_ok: break
check(f'W13 arm: det J_v = {int(sgn13):+d} L3 L4 sin(th3) (L3 cos th2 + L4 cos(th2+th3)), 30 random q', check_ok, f'sign {int(sgn13)}')
q_bnd1 = np.array([0.5, 0.3, 0.0]); q_bnd2 = np.array([0.5, 0.3, np.pi])
check('W13 arm: det = 0 at th3 = 0 (stretched out, boundary)', abs(det13(q_bnd1)) < 1e-12)
check('W13 arm: det = 0 at th3 = 180 deg (folded, boundary)', abs(det13(q_bnd2)) < 1e-12)
th2_i = 60 * deg; th23_i = np.arccos(-2 / 3); th3_i = th23_i - th2_i
q_int = np.array([0.5, th2_i, th3_i])
check('W13 arm interior singularity: th2 = 60 deg, th3 = 71.81 deg gives r = 0 and det = 0',
      abs(r13(q_int)) < 1e-12 and abs(det13(q_int)) < 1e-12, f'th3 = {th3_i / deg:.2f} deg')
tip = fk(ops13, q_int)[:3, 3]
close('W13 arm interior singularity: tool on the vertical axis through the shoulder (x = y = 0)', tip[:2], [0, 0])
# null direction: the tool cannot move horizontally-sideways: J_v has rank 2
Jint = jac0(ops13, q_int)[:3]
check('W13 arm interior singularity: rank J_v = 2', np.linalg.matrix_rank(Jint, tol=1e-9) == 2)
# sideways (perpendicular to the arm plane) is impossible at the interior singularity
sv = np.linalg.svd(Jint)
nullL = sv[0][:, -1]
print('   interior singular pose: left null vector (direction the tool cannot move) =', np.round(nullL, 4), ' th1=0.5 rad -> perpendicular to the arm plane would be', np.round([-np.sin(0.5), np.cos(0.5), 0], 4))
close('W13 interior singularity: lost direction is perpendicular to the arm plane (-sin th1, cos th1, 0)', np.abs(nullL), np.abs([-np.sin(0.5), np.cos(0.5), 0]), 1e-9)
# stretched boundary: lost direction is along the arm
Jb = jac0(ops13, q_bnd1)[:3]
nullb = np.linalg.svd(Jb)[0][:, -1]
arm_dir = np.array([np.cos(q_bnd1[0]) * np.cos(q_bnd1[1]), np.sin(q_bnd1[0]) * np.cos(q_bnd1[1]), np.sin(q_bnd1[1])])
close('W13 stretched pose: lost direction is along the arm (cos th1 cos th2, sin th1 cos th2, sin th2)', np.abs(nullb), np.abs(arm_dir), 1e-9)
out['w13'] = dict(sign=int(sgn13), q_bnd_deg=[r(v / deg, 2) for v in q_bnd1], q_int_deg=[r(v / deg, 4) for v in q_int], th3_int_deg=r(th3_i / deg, 2))
# worked numbers at (30, 20, 40) deg (the widget default)
qd13 = np.array([30, 20, 40]) * deg
J13 = jac0(ops13, qd13); print('W13 J at default =\n', np.round(J13, 4)); print('det', det13(qd13))
out['w13']['default_Jv'] = np.round(J13[:3], 4).tolist(); out['w13']['default_det'] = r(det13(qd13), 5)

# ====================================================================================
# 5. Offset RP arm (W8 folder): det = -d2
# ====================================================================================
opsrp = PRESETS['rp']['ops']
for _ in range(10):
    q = np.array([rng.uniform(-3, 3), rng.uniform(-0.5, 0.5)])
    check_ok = abs(np.linalg.det(jac0(opsrp, q)[:2]) - (-q[1])) < 1e-12
    if not check_ok: break
check('offset RP arm: det of the 2x2 block = -d2 (matches offset_rp_jacobian.py)', check_ok)
J_rp0 = jac0(opsrp, np.array([0.6, 0.0]))
close('offset RP at d2 = 0: revolute and prismatic columns are parallel (col1 = l1 * col2)', J_rp0[:2, 0], 0.5 * J_rp0[:2, 1])

# ====================================================================================
# 6. Craig outward velocity propagation (added) for the planar RR, vs slide p.45
# ====================================================================================
def craig_rr(L1, L2, q, qd):
    R01 = Rz2(q[0]); R12 = Rz2(q[1])
    w1 = np.array([0, 0, qd[0]]); v1 = np.zeros(3)                 # frame {1}
    P12 = np.array([L1, 0, 0])
    w2 = R12.T @ w1 + np.array([0, 0, qd[1]])
    v2 = R12.T @ (v1 + np.cross(w1, P12))
    P23 = np.array([L2, 0, 0])
    v3 = v2 + np.cross(w2, P23)                                     # frame {2} = {3} orientation (tip frame has same axes)
    return v3, w2
qC = np.array([30, 15]) * deg; qdC = np.array([0.2, -0.1])
v3, w2 = craig_rr(3, 4, qC, qdC)
close('Craig recursion: ^3v_3 = ^3J (slide p.45) * qdot, Q1 numbers', v3[:2], JQ1e[:2] @ qdC)
close('Craig recursion: omega_z = q1dot + q2dot', w2, [0, 0, qdC.sum()])
sq, cq = sp.symbols('s2 c2')
v1v = sp.Matrix([0, 0, 0]); wd1, wd2 = sp.symbols('qd1 qd2')
wv1 = sp.Matrix([0, 0, wd1]); Rt = sp.Matrix([[cq, sq, 0], [-sq, cq, 0], [0, 0, 1]])   # R12^T
v2s = Rt * (v1v + wv1.cross(sp.Matrix([l1s, 0, 0])))
w2s = Rt * wv1 + sp.Matrix([0, 0, wd2])
v3s = v2s + w2s.cross(sp.Matrix([l2s, 0, 0]))
check('Craig recursion symbolic: ^3v_3 = (L1 s2 qd1, (L1 c2 + L2) qd1 + L2 qd2, 0)',
      sp.simplify(v3s - sp.Matrix([l1s * sq * wd1, (l1s * cq + l2s) * wd1 + l2s * wd2, 0])) == sp.zeros(3, 1))
# frame {1} velocity of origin of {2} before the cross term: (0, L1 qd1, 0)
close('Craig step: ^1w_1 x ^1P_2 = (0, L1 qd1, 0)', np.cross([0, 0, 0.2], [3, 0, 0]), [0, 0.6, 0])

# ====================================================================================
# 7. Change of frame: rotation of the Jacobian, velocity transform
# ====================================================================================
q = np.array([0.3, -0.9, 1.2]); J0_ = jac0(ops13, q); T_ = fk(ops13, q); RE = T_[:3, :3]
JE_ = np.block([[RE.T, np.zeros((3, 3))], [np.zeros((3, 3)), RE.T]]) @ J0_
qd_ = np.array([0.1, -0.2, 0.3])
close('frame change: ^E v = R^T ^0 v (W13 arm)', JE_[:3] @ qd_, RE.T @ (J0_[:3] @ qd_))
check('frame change: sigma of J_v unchanged (rotation is orthogonal)', np.allclose(np.linalg.svd(J0_[:3], compute_uv=False), np.linalg.svd(JE_[:3], compute_uv=False)))
check('frame change: det J_v unchanged', abs(np.linalg.det(J0_[:3]) - np.linalg.det(JE_[:3])) < 1e-12)
# a numeric frame-change example on the RR Q1: R(45 deg)^T
R45 = Rz2(45 * deg)
close('frame-change example: R(45 deg)^T (1,0,0) = (0.7071, -0.7071, 0)', R45.T @ [1, 0, 0], [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0])

# ====================================================================================
# 8. Inverse velocity, singularity blow-up, resolved rate (notebook Q4)
# ====================================================================================
L1, L2 = 3.0, 4.0
def Jrr(q1, q2, L1=3.0, L2=4.0):
    s1, c1, s12, c12 = np.sin(q1), np.cos(q1), np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12], [L1 * c1 + L2 * c12, L2 * c12]])
# inverse formula for the 2x2 block
q = np.array([30, 60]) * deg
Jq = Jrr(*q); detq = np.linalg.det(Jq)
Jinv_formula = np.array([[L2 * np.cos(q.sum()), L2 * np.sin(q.sum())], [-L1 * np.cos(q[0]) - L2 * np.cos(q.sum()), -L1 * np.sin(q[0]) - L2 * np.sin(q.sum())]]) / (L1 * L2 * np.sin(q[1]))
close('RR: J^-1 = 1/(L1 L2 s2) [[L2 c12, L2 s12],[-L1 c1 - L2 c12, -L1 s1 - L2 s12]]', Jinv_formula, np.linalg.inv(Jq))
# blow-up: tip speed 0.1 m/s perpendicular to the arm (frame E x-axis): q1dot = v/(L1 s2)
rows = []
for q2d in (90, 30, 15, 5, 1):
    JE = np.array([[L1 * np.sin(q2d * deg), 0], [L1 * np.cos(q2d * deg) + L2, L2]])
    qdv = np.linalg.solve(JE, [0.1, 0.0])
    rows.append((q2d, qdv[0], qdv[1]))
    check(f'blow-up: q2 = {q2d} deg, q1dot = 0.1/(L1 sin q2) = {qdv[0]:.4f} rad/s', abs(qdv[0] - 0.1 / (L1 * np.sin(q2d * deg))) < 1e-12)
print('blow-up table (q2, q1dot, q2dot):', [(a, r(b), r(c)) for a, b, c in rows])
out['blowup'] = [dict(q2_deg=a, q1dot=r(b), q2dot=r(c)) for a, b, c in rows]

# Q4 resolved rate
nu = np.array([0.2, -0.3, -0.2]); dt = 0.1
q0 = np.array([30, 60]) * deg
J6 = jac0(rr_ops(3, 4), q0)
check('Q4: 6x2 J rows 3-5 zero so only (vx, vy, wz) matter', np.allclose(J6[2:5], 0))
J3 = J6[[0, 1, 5]]
print('Q4 J (vx,vy,wz) =\n', J3, '\nrank', np.linalg.matrix_rank(J3))
qdot_xy = np.linalg.solve(J3[:2], nu[:2])
wz_xy = J3[2] @ qdot_xy
qdot_pi = np.linalg.pinv(J3) @ nu
res = J3 @ qdot_pi - nu
# normal-equations form of the pseudo-inverse (tall J): (J^T J)^-1 J^T
close('tall-J pseudo-inverse = (J^T J)^-1 J^T', np.linalg.inv(J3.T @ J3) @ J3.T, np.linalg.pinv(J3))
print('Q4 (a) detJ_xy =', np.linalg.det(J3[:2]), ' (b-xy) qdot =', qdot_xy, 'wz achieved', wz_xy, ' (b-pinv) qdot', qdot_pi, 'residual', res)
check('Q4: demanded wz = -0.2 cannot be met together with (vx,vy): achieved wz', abs(wz_xy - (-0.00669873)) < 1e-7, f'{wz_xy:.5f}')
check('Q4: tall J has 3 equations, 2 unknowns: residual nonzero', np.linalg.norm(res) > 0.1)
# normal equations check: J^T residual = 0
close('Q4: least-squares residual is orthogonal to the columns of J (J^T r = 0)', J3.T @ res, [0, 0], 1e-12)
q_new_xy = q0 + qdot_xy * dt; q_new_pi = q0 + qdot_pi * dt
print('Q4 (c) new q xy', q_new_xy / deg, ' pinv', q_new_pi / deg)
def simulate(mode, T=5.0):
    q = q0.copy(); hist = [q.copy()]; steps = int(round(T / dt))
    for _ in range(steps):
        J6 = jac0(rr_ops(3, 4), q); J3 = J6[[0, 1, 5]]
        qdot = np.linalg.solve(J3[:2], nu[:2]) if mode == 'xy' else np.linalg.pinv(J3) @ nu
        q = q + qdot * dt; hist.append(q.copy())
    return np.array(hist)
Hxy, Hpi = simulate('xy'), simulate('pinv')
def tip(Hq): return np.array([[3 * np.cos(a) + 4 * np.cos(a + b), 3 * np.sin(a) + 4 * np.sin(a + b)] for a, b in Hq])
txy, tpi = tip(Hxy), tip(Hpi)
p0 = tip(np.array([q0]))[0]
print('Q4 start tip', p0, ' final xy-mode q', Hxy[-1] / deg, 'tip', txy[-1], ' final pinv q', Hpi[-1] / deg, 'tip', tpi[-1])
print('Q4 min q2 over run (deg): xy', Hxy[:, 1].min() / deg, ' pinv', Hpi[:, 1].min() / deg, ' max', Hxy[:, 1].max() / deg, Hpi[:, 1].max() / deg)
# ideal straight line would arrive at p0 + nu_xy * 5
print('ideal straight-line end', p0 + nu[:2] * 5)
out['Q4'] = dict(q0_deg=[30, 60], nu=nu.tolist(), dt=dt, J=np.round(J3, 4).tolist(), detJ=r(np.linalg.det(J3[:2]), 4),
                 qdot_xy=np.round(qdot_xy, 4).tolist(), wz_xy=r(wz_xy, 4), qdot_pinv=np.round(qdot_pi, 4).tolist(), resid=np.round(res, 4).tolist(),
                 q1_xy_deg=np.round(q_new_xy / deg, 3).tolist(), q1_pinv_deg=np.round(q_new_pi / deg, 3).tolist(),
                 p0=np.round(p0, 4).tolist(), end_xy=dict(q_deg=np.round(Hxy[-1] / deg, 2).tolist(), tip=np.round(txy[-1], 3).tolist()),
                 end_pinv=dict(q_deg=np.round(Hpi[-1] / deg, 2).tolist(), tip=np.round(tpi[-1], 3).tolist()))
# is the xy-mode tip moving close to the commanded constant velocity? (Euler error)
vel_err = np.linalg.norm((txy[1] - txy[0]) / dt - nu[:2])
check('Q4 xy mode: first-step tip velocity = commanded (0.2,-0.3) to Euler accuracy (< 1e-2 m/s)', vel_err < 1e-2, f'{vel_err:.4f}')

# ====================================================================================
# 9. Manipulability: ellipsoid, m, w. RR with L1=3, L2=4, then the 2D check of the ellipsoid equation
# ====================================================================================
def manip(J):
    A = J @ J.T; lam = np.sort(np.linalg.eigvalsh(A))[::-1]; sig = np.sqrt(np.clip(lam, 0, None))
    return A, lam, sig
rows_m = []
for q2d in (15, 60, 90, 150):
    J = Jrr(30 * deg, q2d * deg)
    A, lam, sig = manip(J)
    m_ = sig[-1] / sig[0]; w_ = np.sqrt(np.linalg.det(A))
    check(f'RR L=(3,4), q2={q2d}: w = sqrt(det J J^T) = L1 L2 |sin q2| = {w_:.4f}', abs(w_ - 12 * abs(np.sin(q2d * deg))) < 1e-9)
    check(f'RR q2={q2d}: sigma_i = sqrt(lambda_i) = singular values of J', np.allclose(sig, np.linalg.svd(J, compute_uv=False)))
    check(f'RR q2={q2d}: sigma1 sigma2 = |det J|', abs(sig[0] * sig[1] - abs(np.linalg.det(J))) < 1e-9)
    rows_m.append((q2d, sig[0], sig[1], m_, w_))
print('manip table (q2, s1, s2, m, w):', [tuple(round(float(x), 4) for x in r_) for r_ in rows_m])
out['manip'] = [dict(q2_deg=a, s1=r(b), s2=r(c), m=r(d), w=r(e)) for a, b, c, d, e in rows_m]
# w is independent of q1 for the RR
check('RR: manipulability w independent of q1', abs(np.sqrt(np.linalg.det(Jrr(0.1, 1.0) @ Jrr(0.1, 1.0).T)) - np.sqrt(np.linalg.det(Jrr(2.0, 1.0) @ Jrr(2.0, 1.0).T))) < 1e-9)
# ellipsoid boundary: unit joint-velocity circle -> nu^T (J J^T)^-1 nu = 1
J = Jrr(30 * deg, 60 * deg); A = J @ J.T
ok = True
for t in np.linspace(0, 2 * np.pi, 13):
    qd = np.array([np.cos(t), np.sin(t)]); nuv = J @ qd
    ok &= abs(nuv @ np.linalg.inv(A) @ nuv - 1) < 1e-12
check('velocity ellipsoid: unit qdot circle maps onto nu^T (J J^T)^-1 nu = 1', ok)
# non-square fat J: min-norm qdot gives the same ellipsoid
Jf = rng.normal(size=(3, 5)); Af = Jf @ Jf.T
ok = True
for _ in range(10):
    u = rng.normal(size=5); u /= np.linalg.norm(u); nuv = Jf @ u
    ok &= nuv @ np.linalg.inv(Af) @ nuv <= 1 + 1e-12
check('velocity ellipsoid: fat J (3x5): every unit-norm qdot lands inside nu^T (J J^T)^-1 nu <= 1', ok)
# slide typo: qdot qdot^T is a matrix
qdv = np.array([[0.6], [0.8]])
check('slide p.51: qdot qdot^T is a 2x2 matrix, qdot^T qdot = 1 is the scalar norm condition', (qdv @ qdv.T).shape == (2, 2) and abs((qdv.T @ qdv)[0, 0] - 1) < 1e-12)
# m for stretch: m -> 0
for q2d in (15, 5, 1, 0):
    J = Jrr(30 * deg, q2d * deg); _, lam, sig = manip(J)
    print(f'   RR q2={q2d} deg: m = {sig[-1] / sig[0]:.5f}, w = {np.sqrt(abs(np.linalg.det(J @ J.T))):.5f}')
Jz = Jrr(30 * deg, 0.0); check('RR at q2 = 0: lambda_min = 0 so m = 0', abs(np.linalg.eigvalsh(Jz @ Jz.T)[0]) < 1e-12)
# maximum m for RR with L1=L2 over q2 (is m = 1 reachable?)
q2g = np.linspace(0.01, np.pi - 0.01, 3000)
mm = [(lambda s: s[-1] / s[0])(np.linalg.svd(Jrr(0.3, a, 1, 1), compute_uv=False)) for a in q2g]
print('equal links L1=L2: max m =', max(mm), 'at q2 =', q2g[int(np.argmax(mm))] / deg)
mm2 = [(lambda s: s[-1] / s[0])(np.linalg.svd(Jrr(0.3, a, 3, 4), compute_uv=False)) for a in q2g]
print('L=(3,4): max m =', max(mm2), 'at q2 =', q2g[int(np.argmax(mm2))] / deg, 'w max at 90 deg =', 12)
out['m_max_equal'] = dict(m=r(max(mm)), q2_deg=r(q2g[int(np.argmax(mm))] / deg, 1))
out['m_max_34'] = dict(m=r(max(mm2)), q2_deg=r(q2g[int(np.argmax(mm2))] / deg, 1))
# Hand case for exam: L1=L2=1, q2=90: JJ^T closed form -> sigma
Jh = Jrr(0.0, np.pi / 2, 1, 1)
print('L=1,1 q=(0,90): J =', Jh, 'sig', np.linalg.svd(Jh, compute_uv=False))

# ====================================================================================
# 10. Statics tau = J^T F, force ellipsoid
# ====================================================================================
La, Lb = 0.5, 0.4; qs = np.array([30, 60]) * deg; Fv = np.array([10.0, -20.0])
Js = Jrr(qs[0], qs[1], La, Lb)
tau = Js.T @ Fv
close('statics: J (L=0.5,0.4; q=(30,60)) = [[-0.65,-0.4],[0.4330,0]]', Js, [[-0.65, -0.4], [0.5 * np.cos(30 * deg), 0]])
close('statics: tau = J^T F = (-15.1603, -4.0) N m for F = (10,-20) N', tau, [-0.65 * 10 + 0.5 * np.cos(30 * deg) * -20, -4.0])
print('statics tau =', tau)
close('statics trap: J F (wrong, not J^T F) = (1.5, 4.3301)', Js @ Fv, [1.5, 0.5 * np.cos(30 * deg) * 10])
# virtual work: tau . dq = F . dx
dq = np.array([0.01, -0.02]); dx = Js @ dq
check('virtual work: tau^T dq = F^T dx', abs(tau @ dq - Fv @ dx) < 1e-14)
# force expressed in {E}: same tau when both J and F rotate
RE2 = Rz2(qs.sum())
JsE = RE2.T[:2, :2] @ Js; FvE = RE2.T[:2, :2] @ Fv
close('statics: tau from (^E J, ^E F) equals tau from (^0 J, ^0 F)', JsE.T @ FvE, tau)
# stretched arm: F along the arm needs zero torque
qst = np.array([30 * deg, 0.0]); Fa = 50 * np.array([np.cos(30 * deg), np.sin(30 * deg)])
close('statics at q2 = 0: 50 N pushing along the stretched arm needs tau = (0, 0)', Jrr(*qst, La, Lb).T @ Fa, [0, 0], 1e-12)
Fp = 50 * np.array([-np.sin(30 * deg), np.cos(30 * deg)])
print('perpendicular 50 N at stretched arm: tau =', Jrr(*qst, La, Lb).T @ Fp)
close('statics at q2 = 0: 50 N perpendicular needs tau = (45, 20) N m (L1+L2 = 0.9, L2 = 0.4)', Jrr(*qst, La, Lb).T @ Fp, [45.0, 20.0], 1e-9)
# force ellipsoid F^T J J^T F <= 1 : derive from |tau| <= 1 with F = J^-T tau
Af = Js @ Js.T
ok = True
for t in np.linspace(0, 2 * np.pi, 13):
    tv = np.array([np.cos(t), np.sin(t)]); F = np.linalg.solve(Js.T, tv)
    ok &= abs(F @ Af @ F - 1) < 1e-12
check('force ellipsoid: unit joint-torque circle maps onto F^T (J J^T) F = 1', ok)
lam, V = np.linalg.eigh(Af);
vel_ax, force_ax = np.sqrt(lam), 1 / np.sqrt(lam)
check('velocity semi-axes sqrt(lambda), force semi-axes 1/sqrt(lambda): reciprocal, same eigenvectors', np.allclose(vel_ax * force_ax, 1))
print('force/velocity ellipsoid axes (L=0.5,0.4; q=(30,60)): lam', lam, 'vel', vel_ax, 'force', force_ax)
# the velocity ellipsoid using (J J^T)^-1 and the WRONG force ellipsoid using (J J^T)^-1: different shapes
Fwrong = V[:, 0] * vel_ax[0]
check('the force ellipsoid is NOT nu^T (JJ^T)^-1 nu: the long velocity axis is the SHORT force axis', force_ax[0] > force_ax[1] and vel_ax[0] < vel_ax[1])
out['statics'] = dict(L=[0.5, 0.4], q_deg=[30, 60], F=[10, -20], J=np.round(Js, 4).tolist(), tau=np.round(tau, 4).tolist(),
                      lam=np.round(lam, 5).tolist(), vel_axes=np.round(vel_ax, 4).tolist(), force_axes=np.round(force_ax, 4).tolist())
# and in a 3x? wrench: planar moment n_z enters the 6th row
Jw6 = jac0(rr_ops(La, Lb), qs)
tau6 = Jw6.T @ np.array([10, -20, 0, 0, 0, 2.0])
close('statics: wrench (10,-20,0,0,0,2): tau = J_xy^T F + n_z (1,1)', tau6, tau + 2 * np.array([1, 1]))
# tall pseudo-inverse and fat pseudo-inverse formulas
Jt_ = rng.normal(size=(6, 2)); Jf_ = rng.normal(size=(2, 6))
close('pseudo-inverse of a tall J: (J^T J)^-1 J^T', np.linalg.inv(Jt_.T @ Jt_) @ Jt_.T, np.linalg.pinv(Jt_), 1e-10)
close('pseudo-inverse of a fat J: J^T (J J^T)^-1', Jf_.T @ np.linalg.inv(Jf_ @ Jf_.T), np.linalg.pinv(Jf_), 1e-10)
# min-norm property
nu2 = rng.normal(size=2); qa = np.linalg.pinv(Jf_) @ nu2; nullv = np.linalg.svd(Jf_)[2][-1]
check('min-norm: J^+ nu has smaller norm than J^+ nu + null-space motion', np.linalg.norm(qa) < np.linalg.norm(qa + 0.5 * nullv) and np.allclose(Jf_ @ nullv, 0, atol=1e-12))

# ====================================================================================
# 11. PUMA 560: slide p.24 matrix, notebook Q2/Q3
# ====================================================================================
slideJ = np.array([[0.13, 0, 0.306, 0, 0, 0], [0.664, 0, 0, -0.038, 0, 0], [0, 0.664, 0.359, 0, 0.053, 0],
                   [0, 0, 0, 0.707, 0, 1], [0, -1, -1, 0, -1, 0], [1, 0, 0, -0.707, 0, 0]])
try:
    import roboticstoolbox as rtb
    puma = rtb.models.URDF.Puma560(); qn = puma.configs['qn']
    Jn = puma.jacob0(qn)
    close('PUMA: slide p.24 matrix == roboticstoolbox URDF Puma560 jacob0 at qn = (0,45,180,0,45,0) deg (to 3 dp)', np.round(Jn, 3), slideJ, 1e-9)
    check('PUMA qn configuration is (0, 45, 180, 0, 45, 0) deg', np.allclose(qn, np.array([0, 45, 180, 0, 45, 0]) * deg))
    q5z = qn.copy(); q5z[4] = 0
    check('PUMA: q5 = 0 -> rank J = 5 (wrist singularity)', np.linalg.matrix_rank(puma.jacob0(q5z), tol=1e-8) == 5)
    check('PUMA: at qn rank J = 6', np.linalg.matrix_rank(Jn, tol=1e-8) == 6)
    Jq5 = puma.jacob0(q5z)
    close('PUMA: at q5 = 0 the angular parts of columns 4 and 6 coincide', Jq5[3:, 3], Jq5[3:, 5], 1e-9)
    tipn = puma.fkine(qn).t
    print('PUMA tool point at qn =', tipn)
    close('PUMA: column 1 linear part = z0 x p => p_x = 0.664, p_y = -0.130 (tool point of the URDF model)', [tipn[0], tipn[1]], [0.664, -0.130], 1e-3)
    close('PUMA: J_v col 1 = (-p_y, p_x, 0)', Jn[:3, 0], [-tipn[1], tipn[0], 0], 1e-6)
    check('PUMA: col 4 angular (0.707,0,-0.707) and col 6 (1,0,0) make angle q5 = 45 deg', abs(np.degrees(np.arccos(Jn[3:, 3] @ Jn[3:, 5])) - 45) < 1e-6)
    print('PUMA det at qn =', np.linalg.det(Jn))
    out['puma'] = dict(q_deg=[0, 45, 180, 0, 45, 0], det_qn=r(np.linalg.det(Jn), 5), rank_q5z=5)
except Exception as e:
    check('roboticstoolbox available for the PUMA reproduction', False, repr(e))

# angular columns of slide matrix: read the axes
ang = slideJ[3:]
close('PUMA slide: angle between wrist axes z4 and z6 columns = 45 deg (cos = 0.707)', [ang[:, 3] @ ang[:, 5]], [0.707], 1e-9)
check('PUMA slide: angular columns 2, 3, 5 equal (0,-1,0): three parallel horizontal axes', np.allclose(ang[:, 1], [0, -1, 0]) and np.allclose(ang[:, 2], [0, -1, 0]) and np.allclose(ang[:, 4], [0, -1, 0]))
# Zero-offset PUMA ETS (W2 notebook Q5 answer): det J = +- Lu Lf sin q3 (Lu sin q2 + Lf sin(q2+q3)) sin q5
Lb_, Lu_, Lf_ = 1.0, 0.5, 0.3
opsp = [con('tz', Lb_), rot('Rz', 0), rot('Ry', 1), con('tz', Lu_), rot('Ry', 2), con('tz', Lf_), rot('Rz', 3), rot('Ry', 4), rot('Rz', 5)]
def detp(q): return np.linalg.det(jac0(opsp, q))
def formula(q): return Lu_ * Lf_ * np.sin(q[2]) * (Lu_ * np.sin(q[1]) + Lf_ * np.sin(q[1] + q[2])) * np.sin(q[4])
ratios = []
for _ in range(30):
    q = rng.uniform(-3, 3, 6); ratios.append(detp(q) / formula(q))
check('PUMA-like (zero offset) ETS: |det J| = Lu Lf |sin q3| |Lu sin q2 + Lf sin(q2+q3)| |sin q5|, constant ratio over 30 random q',
      np.allclose(np.abs(ratios), 1, atol=1e-9) and np.allclose(ratios, ratios[0]), f'ratio {ratios[0]:+.3f}')
q3z = np.array([0.3, 0.9, 0.0, 0.2, 0.7, 0.1]); check('PUMA-like: q3 = 0 (elbow stretched) -> det = 0', abs(detp(q3z)) < 1e-12)
q5z = np.array([0.3, 0.9, 1.0, 0.2, 0.0, 0.1]); check('PUMA-like: q5 = 0 (wrist) -> det = 0', abs(detp(q5z)) < 1e-12)
# shoulder: wrist centre on waist axis: Lu sin q2 + Lf sin(q2+q3) = 0
q2s_ = 0.4; q23s = -np.arcsin(Lu_ * np.sin(q2s_) / Lf_)  # sin(q2+q3) = -Lu sin q2 / Lf
q3s = q23s - q2s_
qsh = np.array([0.3, q2s_, q3s, 0.2, 0.7, 0.1])
check('PUMA-like: wrist centre on the waist axis (Lu sin q2 + Lf sin q23 = 0) -> det = 0', abs(detp(qsh)) < 1e-12)
wc = fk(opsp[:5] + [con('tz', Lf_)], qsh)[:3, 3]
close('PUMA-like shoulder singularity: wrist centre x = y = 0', wc[:2], [0, 0], 1e-12)
out['puma_like'] = dict(Lb=Lb_, Lu=Lu_, Lf=Lf_, sign=r(ratios[0], 0), q2_shoulder_deg=r(q2s_ / deg, 3), q3_shoulder_deg=r(q3s / deg, 3))
print('shoulder singularity example q2, q3 (deg):', q2s_ / deg, q3s / deg)
# number of free parameters = zero-offset PUMA with q5 = 0: z4 = z6
Jz5 = jac0(opsp, q5z)
close('PUMA-like at q5 = 0: angular columns 4 and 6 identical (axes aligned)', Jz5[3:, 3], Jz5[3:, 5], 1e-12)
check('PUMA-like at q5 = 0: rank J = 5', np.linalg.matrix_rank(Jz5, tol=1e-9) == 5)
# at the wrist singularity, which instantaneous rotation is lost: about the axis perpendicular to z4 (= z6) and y5
Jz5w = Jz5[3:]; print('wrist singular angular part rank =', np.linalg.matrix_rank(Jz5w, tol=1e-9))


# ====================================================================================
# 13. Extra numbers quoted in the text (guided problems, singular poses)
# ====================================================================================
check('trap value: 12 sin 45 deg = 8.4853', abs(12 * np.sin(45 * deg) - 8.4853) < 5e-5)
check('Q2 quiz: L1=L2=1, det at q2=30 deg is 0.5', abs(np.linalg.det(Jrr(0.3, 30 * deg, 1, 1)) - 0.5) < 1e-12)
check('ratio of q1dot between 30 and 5 deg = sin30/sin5 = 5.7', abs(np.sin(30 * deg) / np.sin(5 * deg) - 5.737) < 5e-3)
J0s = Jrr(30 * deg, 0.0)
close('RR stretched, q1=30: frame-0 block = [[-3.5,-2],[6.0622,3.4641]]', J0s, [[-3.5, -2], [6.0622, 3.4641]], 1e-4)
close('RR stretched: col 1 = 1.75 col 2, both along (-0.5, 0.866)', J0s[:, 0], 1.75 * J0s[:, 1], 1e-12)
for q2d, want in ((0, [[0, 0], [7, 4]]), (180, [[0, 0], [1, 4]])):
    JE2 = np.array([[3 * np.sin(q2d * deg), 0], [3 * np.cos(q2d * deg) + 4, 4]])
    close(f'RR in tool frame at q2={q2d}: {want}', JE2, want, 1e-12)
    check(f'RR at q2={q2d}: rank 1', np.linalg.matrix_rank(JE2, tol=1e-9) == 1)
A15 = Jrr(30 * deg, 15 * deg) @ Jrr(30 * deg, 15 * deg).T
close('P6.7: at q2=15 tr A = 64.182, det A = 9.646, lambda = 64.032, 0.1506', [np.trace(A15), np.linalg.det(A15)] + list(np.linalg.eigvalsh(A15)[::-1]), [64.182, 9.646, 64.032, 0.1506], 1e-3)
A90 = Jrr(30 * deg, 90 * deg) @ Jrr(30 * deg, 90 * deg).T
close('P6.7: at q2=90 tr A = 41, det A = 144, lambda = 37.12, 3.88', [np.trace(A90), np.linalg.det(A90)] + list(np.linalg.eigvalsh(A90)[::-1]), [41, 144, 37.12, 3.88], 5e-3)
check('speed is the same in both frames: 0.9918 m/s', abs(np.linalg.norm([0.1553, 0.9796]) - 0.9918) < 1e-3)
check('radius 3x? ellipsoid scale: det of RR in tool frame at 30,15 equals 3.1058', abs(np.linalg.det(np.array([[3 * np.sin(15 * deg), 0], [3 * np.cos(15 * deg) + 4, 4]])) - 3.1058) < 5e-5)

# ====================================================================================
# 12. Test vectors for the widget (identical definitions to the JS)
# ====================================================================================
def widget_state(key, qv, qdot):
    P = PRESETS[key]; ops, rows = P['ops'], P['rows']
    qv = np.array(qv, float); J = jac0(ops, qv); Jt = J[rows]
    A = Jt @ Jt.T; lam = np.clip(np.linalg.eigvalsh(A), 0, None)[::-1]; sig = np.sqrt(lam)
    m_eff = len(rows); n = J.shape[1]
    sg = sig[:min(m_eff, n)]
    det = float(np.linalg.det(Jt)) if Jt.shape[0] == Jt.shape[1] else None
    v = Jt @ np.array(qdot, float)
    return dict(q=qv.tolist(), J=J.tolist(), sigma=sg.tolist(), m=float(sg[-1] / sg[0]) if sg[0] > 0 else 0.0,
                w=float(np.prod(sg)), det=det, tip=fk(ops, qv)[:3, 3].tolist(), v=v.tolist(), qdot=list(qdot))
TV = {
    'rr': [([30 * deg, 60 * deg], [0.5, -0.3]), ([0.0, 0.0], [0.5, -0.3]), ([30 * deg, 15 * deg], [0.2, -0.1]), ([-40 * deg, 150 * deg], [1, 1])],
    'rp': [([35 * deg, 0.35], [0.5, 0.1]), ([35 * deg, 0.0], [0.5, 0.1]), ([-1.0, -0.3], [-0.4, 0.2])],
    'w13': [([30 * deg, 20 * deg, 40 * deg], [0.4, 0.2, -0.3]), (q_bnd1, [0.4, 0.2, -0.3]), (q_int, [0.4, 0.2, -0.3]), ([-70 * deg, 100 * deg, -50 * deg], [1, -1, 1])],
}
vec = {}
for key, lst in TV.items():
    vec[key] = {'ops': [{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in o.items()} for o in PRESETS[key]['ops']], 'rows': PRESETS[key]['rows'], 'cases': []}
    for qv, qdt in lst:
        s = widget_state(key, qv, qdt)
        # cross-check the widget's number against finite differences
        Jfd = jac_fd(PRESETS[key]['ops'], np.array(qv, float))
        check(f'widget vector {key} q={np.round(qv, 3).tolist()}: J == finite difference', np.abs(np.array(s['J']) - Jfd).max() < 5e-8)
        vec[key]['cases'].append(s)
# prismatic example: the q for rp is [rad, m]
with open(os.path.join(os.path.dirname(__file__), 'w06_vectors.json'), 'w') as f:
    json.dump(dict(vectors=vec, notes=out), f, indent=1)
print('wrote w06_vectors.json')
print(json.dumps(out, indent=None)[:3000])
report()
