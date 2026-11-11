"""W10 Wheeled robot kinematics — verifies every number on the W10 tab. Run: python3 site/src/checks/w10.py"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *

out = {}
u, v, w, r, a, b, d, Lb = sp.symbols('u v thetadot r a b d L', real=True)
def wheel_row(xw, yw, beta, gamma=0):
    """generalised model (slide p.36): phidot_i = [1/r, tan(g)/r] R(beta)^T [[1,0,-y],[0,1,x]] zeta"""
    Rt = sp.Matrix([[sp.cos(beta), sp.sin(beta)], [-sp.sin(beta), sp.cos(beta)]])
    return (sp.Matrix([[1/r, sp.tan(gamma)/r]]) * Rt * sp.Matrix([[1, 0, -yw], [0, 1, xw]]))
def noslide_row(xw, yw, beta):
    return sp.Matrix([[-sp.sin(beta), sp.cos(beta), yw*sp.sin(beta) + xw*sp.cos(beta)]])

# ---- F10 (p.68): rolling + no-sliding rows equal the generalised model with gamma = 0 ----
xs, ys, bs = sp.symbols('x_w y_w beta', real=True)
roll = sp.Matrix([[sp.cos(bs), sp.sin(bs), -ys*sp.cos(bs) + xs*sp.sin(bs)]])
check('rolling row (p.68) == r * generalised row (p.36) with gamma = 0', sp.simplify(roll - r*wheel_row(xs, ys, bs)) == sp.zeros(1, 3))
# no-slide row is the component of the contact velocity along the axle
contact = sp.Matrix([u - w*ys, v + w*xs]); axle = sp.Matrix([-sp.sin(bs), sp.cos(bs)])
check('no-slide row (p.68) == contact velocity · axle direction', sp.simplify((noslide_row(xs, ys, bs)*sp.Matrix([u, v, w]))[0] - (contact.T*axle)[0]) == 0)

# ---- differential drive (pp.43-47): W1 at (0, +a), W2 at (0, -a) ----
Winv = sp.Matrix.vstack(wheel_row(0, a, 0), wheel_row(0, -a, 0))
check('diff drive: phid1 = (u - a thd)/r, phid2 = (u + a thd)/r', sp.simplify(Winv - sp.Matrix([[1/r, 0, -a/r], [1/r, 0, a/r]])) == sp.zeros(2, 3))
p1, p2 = sp.symbols('phid1 phid2')
zeta = sp.Matrix([[r/2, r/2], [0, 0], [-r/(2*a), r/(2*a)]]) * sp.Matrix([p1, p2])
check('diff drive: W (slide p.47) inverts the rows (with v = 0 from no-slide)', sp.simplify(Winv*zeta - sp.Matrix([p1, p2])) == sp.zeros(2, 1))
rn, an, f1, f2 = 0.05, 0.15, 4.0, 6.0
un = rn*(f1 + f2)/2; wn = rn*(f2 - f1)/(2*an)
print(f'diff drive r=0.05, a=0.15, phid=(4,6): u = {un:.4f} m/s, thetadot = {wn:.4f} rad/s, ICR y = {un/wn:.4f} m')
out['dd'] = dict(r=rn, a=an, phid=[f1, f2], u=r_(un) if (r_ := (lambda x: float(np.round(x, 6)))) else 0, thd=r_(wn), icr=r_(un/wn))
# ICR (practical Q3): on the axle at y = u/thd = a (phid1 + phid2)/(phid2 - phid1)
icr = sp.simplify((zeta[0]/zeta[2]))
check('ICR y = a(phid1+phid2)/(phid2-phid1)', sp.simplify(icr - a*(p1 + p2)/(p2 - p1)) == 0)
check('ICR example a=0.3, phid=(2,5): y = 0.7 m', np.isclose(0.3*(2 + 5)/(5 - 2), 0.7))
# check the ICR is the zero-velocity point: velocity of chassis point (0, y) = (u - y thd, v) = 0
check('ICR is the point of zero velocity', sp.simplify((zeta[0] - icr*zeta[2])) == 0)

# ---- skid-steer, 4 standard wheels (pp.48-52) ----
S4 = sp.Matrix.vstack(*[wheel_row(x_, y_, 0) for x_, y_ in [(-b, a), (b, a), (-b, -a), (b, -a)]])
check('skid-steer rows: (1/r)[1,0,-a] (left) and (1/r)[1,0,+a] (right); b drops out', sp.simplify(S4 - sp.Matrix([[1, 0, -a], [1, 0, -a], [1, 0, a], [1, 0, a]])/r) == sp.zeros(4, 3))
S4u = S4[:, [0, 2]]   # v = 0 from no-slide
pinv = sp.simplify((S4u.T*S4u).inv()*S4u.T)
check('skid-steer: u = (r/4)Σphid, thd = (r/4a)(-φ1-φ2+φ3+φ4) is the least-squares inverse', sp.simplify(pinv - sp.Matrix([[r/4, r/4, r/4, r/4], [-r/(4*a), -r/(4*a), r/(4*a), r/(4*a)]])) == sp.zeros(2, 4))

# ---- tricycle (pp.56-58): front W1 at (b, 0) driven + steered; rear axle through C, so v = 0 ----
be = sp.symbols('beta1', real=True)
cv = sp.Matrix([u, b*w])   # step 2 with v = 0
sol_t = sp.solve([sp.Eq(r*sp.Symbol('P')*sp.cos(be), u), sp.Eq(r*sp.Symbol('P')*sp.sin(be), b*w)], [sp.Symbol('P'), be], dict=True)
check('tricycle: r phid1 (cos b1, sin b1) = (u, b thd) has phid1 = ±sqrt(u²+b²thd²)/r', any(sp.simplify(x_[sp.Symbol('P')]**2 - (u**2 + b**2*w**2)/r**2) == 0 for x_ in sol_t))
un2, wn2, bn, rn2 = 0.4, 0.5, 0.6, 0.1
phid_t = np.hypot(un2, bn*wn2)/rn2; b1 = np.arctan2(bn*wn2, un2)
row = wheel_row(bn, 0, b1).subs({r: rn2})
check('tricycle numbers satisfy the generalised model', np.isclose(float((row*sp.Matrix([un2, 0, wn2]))[0]), phid_t))
check('tricycle numbers satisfy the front wheel no-slide row', abs(float((noslide_row(bn, 0, b1)*sp.Matrix([un2, 0, wn2]))[0])) < 1e-12)
print(f'tricycle u=0.4, thd=0.5, b=0.6, r=0.1: phid1 = {phid_t:.4f} rad/s, beta1 = {np.degrees(b1):.3f}°')
out['tri'] = dict(u=un2, thd=wn2, b=bn, r=rn2, phid=r_(phid_t), beta_deg=r_(np.degrees(b1)))

# ---- bicycle (practical Q4): rear fixed at C (0,0), front steered at (L, 0) ----
sol = sp.solve([(noslide_row(0, 0, 0)*sp.Matrix([u, v, w]))[0], (noslide_row(Lb, 0, bs)*sp.Matrix([u, v, w]))[0]], [v, w], dict=True)[0]
check('bicycle: v = 0, thd = u tan(beta)/L', sp.simplify(sol[v]) == 0 and sp.simplify(sol[w] - u*sp.tan(bs)/Lb) == 0)
check('bicycle example u=2, beta=20°, L=1.2: thd = 0.6066 rad/s, turn radius 3.297 m', np.isclose(2*np.tan(20*deg)/1.2, 0.6066, atol=1e-4) and np.isclose(1.2/np.tan(20*deg), 3.297, atol=1e-3))

# ---- Ackermann (p.61, added): both front axle lines through the rear-axle ICR: cot(bo) - cot(bi) = track/wheelbase ----
Lw, tw, R_ = 2.5, 1.5, 6.0
bi = np.arctan(Lw/(R_ - tw/2)); bo = np.arctan(Lw/(R_ + tw/2))
check('Ackermann: cot(beta_out) - cot(beta_in) = track / wheelbase', np.isclose(1/np.tan(bo) - 1/np.tan(bi), tw/Lw))
print(f'Ackermann wheelbase 2.5, track 1.5, turn radius 6 m: inner {np.degrees(bi):.3f}°, outer {np.degrees(bo):.3f}°')
out['ack'] = dict(L=Lw, track=tw, R=R_, inner=r_(np.degrees(bi)), outer=r_(np.degrees(bo)))

# ---- omni triangle (practical Q1 / W13 Q10) and practical Q2: motion in the inertial frame ----
J = sp.Matrix([[1, 0, -d], [-sp.Rational(1, 2), -sp.sqrt(3)/2, -d], [-sp.Rational(1, 2), sp.sqrt(3)/2, -d]])
rows = sp.Matrix.vstack(*[wheel_row(d*sp.cos(al), d*sp.sin(al), be_)*r for al, be_ in [(sp.pi/2, 0), (-sp.pi/6, -2*sp.pi/3), (7*sp.pi/6, 2*sp.pi/3)]])
check('omni triangle rows from the generalised model == W13 Q10 matrix', sp.simplify(rows - J) == sp.zeros(3, 3))
dn, rn3 = 0.10, 0.05
Jn = np.array(J.subs(d, dn), float)
phid = np.array([4.0, -1.0, -1.0])
zet = rn3*np.linalg.solve(Jn, phid)
print(f'omni d=0.1, r=0.05, phid=(4,-1,-1): (u, v, thd) = {np.round(zet, 4)}')
def exact_pose(z, T, x0=np.zeros(3)):
    uu, vv, ww = z; th = x0[2] + ww*T
    if abs(ww) < 1e-12: return np.array([x0[0] + uu*T*np.cos(x0[2]) - vv*T*np.sin(x0[2]), x0[1] + uu*T*np.sin(x0[2]) + vv*T*np.cos(x0[2]), th])
    sx = (np.sin(th) - np.sin(x0[2]))/ww; cx = (np.cos(th) - np.cos(x0[2]))/ww
    return np.array([x0[0] + uu*sx + vv*cx, x0[1] - uu*cx + vv*sx, th])
def euler(z, T, dt):
    p = np.zeros(3)
    for _ in range(int(round(T/dt))):
        th = p[2]; p = p + dt*np.array([z[0]*np.cos(th) - z[1]*np.sin(th), z[0]*np.sin(th) + z[1]*np.cos(th), z[2]])
    return p
pe = exact_pose(zet, 2.0); p1e = euler(zet, 2.0, 0.01)
print(f'   pose after 2 s: exact {np.round(pe, 4)}, Euler dt=0.01 {np.round(p1e, 4)}, error {np.linalg.norm(pe[:2] - p1e[:2])*1000:.2f} mm')
check('Euler (dt = 0.01) converges to the exact arc (error < 1 cm)', np.linalg.norm(pe[:2] - p1e[:2]) < 0.01)
icr_c = np.array([-zet[1]/zet[2], zet[0]/zet[2]])           # ICR in the chassis frame at t = 0 (= inertial, pose 0)
pts = [exact_pose(zet, T_) for T_ in np.linspace(0, 6, 25)]
rad = [np.hypot(p_[0] - icr_c[0], p_[1] - icr_c[1]) for p_ in pts]
check(f'constant wheel speeds: path is a circle of radius {np.hypot(*icr_c):.4f} m about the ICR', np.allclose(rad, np.hypot(*icr_c)))
out['omni'] = dict(d=dn, r=rn3, phid=phid.tolist(), zeta=np.round(zet, 4).tolist(), pose2=np.round(pe, 4).tolist(), euler2=np.round(p1e, 4).tolist())

# ---- manoeuvrability (pp.62-64): check the slides' four examples against a rank definition (added: Siegwart) ----
def deg_mob(fixed, steered):
    rows_ = [noslide_row(x_, y_, b_) for x_, y_, b_ in fixed + steered]
    C1 = sp.Matrix.vstack(*rows_) if rows_ else sp.zeros(0, 3)
    dm = 3 - (C1.rank() if rows_ else 0)
    ds = sp.Matrix.vstack(*[noslide_row(*s) for s in steered]).rank() if steered else 0
    return dm, ds, dm + ds
exs = {'2 fixed + castor': deg_mob([(0, 0.2, 0), (0, -0.2, 0)], []),
       '2 fixed + 1 steered': deg_mob([(0, 0.2, 0), (0, -0.2, 0)], [(0.3, 0, 0.4)]),
       '2 steered + 1 Swedish': deg_mob([], [(0.3, 0.1, 0.4), (-0.3, -0.1, 1.1)]),
       '3 Swedish': deg_mob([], [])}
check('slide p.64 values (dm, ds, dM) = (2,0,2), (1,1,2), (1,2,3), (3,0,3) match dm = 3 − rank C1, ds = rank C1s',
      list(exs.values()) == [(2, 0, 2), (1, 1, 2), (1, 2, 3), (3, 0, 3)], exs)
out['manoeuvrability'] = {k: list(map(int, v_)) for k, v_ in exs.items()}
json.dump(out, open(os.path.join(os.path.dirname(__file__), 'w10_vectors.json'), 'w'), indent=1)
report()
