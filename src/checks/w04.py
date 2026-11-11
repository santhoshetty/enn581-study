"""W4 Parallel manipulators — verifies every number on the W4 tab. Run: python3 site/src/checks/w04.py"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *

out = {}
GK = lambda D, N, J, F, Rc=0, Rm=0: D*(N - J - 1) + F + Rc - Rm
# ---------- mobility counts (slides pp.25-43) ----------
check('3-RPR planar (pp.25-27): 3(8-9-1)+9 = 3', GK(3, 8, 9, 9) == 3)
check('simplest delta R-U-U legs (pp.29-30): 6(8-9-1)+(1+2+2)*3 = 3', GK(6, 8, 9, 15) == 3)
check('common delta plain (p.33): 6(11-15-1)+39 = 9', GK(6, 11, 15, 39) == 9)
check('common delta extended (p.36): +0-6 = 3', GK(6, 11, 15, 39, 0, 6) == 3)
Rc_4rrr = 12 - len({'Ty', 'Tz', 'Rx'} | {'Tx', 'Tz', 'Ry'})
check('4RRR (pp.37-43): union of loop DoFs = 5, R_C = 12-5 = 7', Rc_4rrr == 7)
check('4RRR: plain G-K = -6, extended = -6+7-0 = 1', GK(6, 10, 12, 12) == -6 and GK(6, 10, 12, 12, 7, 0) == 1)
check('Gough-Stewart 6-UPS (added): 6(14-18-1)+6*(2+1+3) = 6', GK(6, 14, 18, 36) == 6)
check('planar four-bar: 3(4-4-1)+4 = 1', GK(3, 4, 4, 4) == 1)

# ---------- four-bar: loop closure, FK (assembly modes), slide formulas ----------
L0, L1, L3, a, b = 1.0, 0.5, 0.6, 0.2, 0.25
L2 = a + b
def fk_fourbar(th1, mode):
    """given crank angle th1, return (A, B, K, phi, phi1) for assembly mode +1/-1 (B above/below line AD)"""
    A = L1*np.array([np.cos(th1), np.sin(th1)]); D = np.array([L0, 0.0])
    d = np.linalg.norm(D - A)
    if d > L2 + L3 or d < abs(L2 - L3): return None
    x_ = (L2**2 - L3**2 + d**2)/(2*d); h = np.sqrt(max(L2**2 - x_**2, 0))
    u = (D - A)/d; n = np.array([-u[1], u[0]])
    B = A + x_*u + mode*h*n
    phi = np.arctan2(B[1] - A[1], B[0] - A[0])
    K = A + a*np.array([np.cos(phi), np.sin(phi)])
    phi1 = np.arctan2(B[1], B[0] - L0)
    return A, B, K, phi, phi1
th1 = 70*deg
res = {m: fk_fourbar(th1, m) for m in (+1, -1)}
for m, (A, B, K, phi, phi1) in res.items():
    eta = (L1*np.cos(th1) - L0 - L3*np.cos(phi1))**2 + (L1*np.sin(th1) - L3*np.sin(phi1))**2 - L2**2
    check(f'four-bar mode {m:+d}: Freudenstein-type eta1(theta1, phi1) = 0 (p.57)', abs(eta) < 1e-12)
    phi2 = phi - th1; phi3 = th1 + phi2 - phi1 - np.pi
    lhs = [L1*np.cos(th1) + a*np.cos(th1 + phi2), L1*np.sin(th1) + a*np.sin(th1 + phi2)]
    rhs = [L0 + L3*np.cos(phi1) + b*np.cos(phi1 + phi3), L3*np.sin(phi1) + b*np.sin(phi1 + phi3)]
    close(f'four-bar mode {m:+d}: three loop-closure equations (p.54) hold with theta1+phi2 = phi1+phi3+pi', lhs, rhs, show=False)
    print(f'   mode {m:+d}: K = {np.round(K,4)}, phi = {phi/deg:.3f}°, phi1 = {phi1/deg:.3f}°')
out['fourbar'] = dict(L0=L0, L1=L1, L2=L2, L3=L3, a=a, b=b, th1_deg=70,
                      modes={str(m): dict(K=np.round(v[2], 4).tolist(), phi_deg=r(v[3]/deg, 3), phi1_deg=r(v[4]/deg, 3)) for m, v in res.items()})
# slide FK (pp.62-66): given K = (x, y), find delta = theta1 + phi2
A_, B_, K, phi, phi1 = res[+1]
x, y = K
A1, B1, C1 = x - L0, y, ((x - L0)**2 + y**2 + b**2 - L3**2)/(2*b)
A2, B2, C2 = x, y, (L1**2 - a**2 - x**2 - y**2)/(2*a)
check('slide p.62: A1 cos(delta)+B1 sin(delta)+C1 = 0 at the true delta', abs(A1*np.cos(phi) + B1*np.sin(phi) + C1) < 1e-12)
check('slide p.64: A2 cos(delta)+B2 sin(delta)+C2 = 0 at the true delta', abs(A2*np.cos(phi) + B2*np.sin(phi) + C2) < 1e-12)
Dd = A1*B2 - A2*B1
cd_, sd_ = (B1*C2 - B2*C1)/Dd, (A2*C1 - A1*C2)/Dd
close('linear solve: cos(delta), sin(delta) from Cramer', [cd_, sd_], [np.cos(phi), np.sin(phi)], show=False)
delta_ok = 2*np.arctan((A2*C1 - A1*C2)/((B1*C2 - B2*C1) + Dd))
delta_slide = np.arctan((A1*C2 - A2*C1)/((B1*C2 - B2*C1) + Dd))
check('corrected half-angle form 2·atan[(A2C1−A1C2)/((B1C2−B2C1)+(A1B2−A2B1))] recovers delta', np.isclose(np.arctan2(np.sin(delta_ok), np.cos(delta_ok)), phi))
check('slide p.65 form atan[(A1C2−A2C1)/(...)] does NOT recover delta here', not np.isclose(delta_slide, phi, atol=1e-6), f'slide gives {delta_slide/deg:.3f}°, true {phi/deg:.3f}°')
check('slide p.65 consistency identity (A1B2−A2B1)² = (A1C2−A2C1)² + (B1C2−B2C1)²', np.isclose(Dd**2, (A1*C2 - A2*C1)**2 + (B1*C2 - B2*C1)**2))
out['fk_slide'] = dict(delta_true=r(phi/deg, 4), delta_slide=r(delta_slide/deg, 4))
# Grashof classification
s_, l_ = sorted([L0, L1, L2, L3])[0], sorted([L0, L1, L2, L3])[-1]
p_q = sum(sorted([L0, L1, L2, L3])[1:3])
check('this four-bar is non-Grashof (s+l > p+q): no link turns fully', s_ + l_ > p_q, f'{s_}+{l_} vs {p_q}')
ok_angles = np.array([t for t in np.arange(-180, 180, 0.05) if fk_fourbar(t*deg, 1) is not None])
lim = np.degrees(np.arccos((L0**2 + L1**2 - (L2 + L3)**2)/(2*L0*L1)))
check(f'crank closes only for |theta1| <= {lim:.2f}° (|AD| <= L2+L3), i.e. it rocks', abs(ok_angles.min() + lim) < 0.06 and abs(ok_angles.max() - lim) < 0.06 and np.all(np.diff(ok_angles) < 0.06),
      f'{ok_angles.min():.2f}..{ok_angles.max():.2f}')
rng_ = [float(-lim), float(lim)]
out['fourbar']['th1_range_deg'] = [r(rng_[0], 2), r(rng_[1], 2)]

# ---------- delta robot IK (added) ----------
Rb, Rp, La, Lr = 0.20, 0.05, 0.20, 0.45
ang = np.array([0, 120, 240])*deg
def delta_ik(p):
    th = []
    for g in ang:
        u = np.array([np.cos(g), np.sin(g), 0]); t = np.array([-np.sin(g), np.cos(g), 0])
        d = p + Rp*u - Rb*u
        du, dt, dz = d @ u, d @ t, d[2]
        K = (d @ d + La**2 - Lr**2)/(2*La)
        A, B = du, -dz
        rho = np.hypot(A, B)
        if abs(K) > rho: return None
        th.append(np.arctan2(B, A) - np.arccos(K/rho))   # elbow-out branch: arm hangs outward/down (theta measured down from horizontal)
    return np.array(th)
def delta_elbows(th):
    return [Rb*np.array([np.cos(g), np.sin(g), 0]) + La*np.array([np.cos(t)*np.cos(g), np.cos(t)*np.sin(g), -np.sin(t)]) for g, t in zip(ang, th)]
pt = np.array([0.03, -0.02, -0.38])
thd = delta_ik(pt)
E = delta_elbows(thd)
lens = [np.linalg.norm(E[i] - (pt + Rp*np.array([np.cos(g), np.sin(g), 0]))) for i, g in enumerate(ang)]
close('delta IK: every rod length = Lr (closure)', lens, [Lr]*3)
print('   delta IK at p = (0.03, -0.02, -0.38): theta (deg) =', np.round(thd/deg, 3))
check('delta: chosen branch has elbows outside the base circle and below the base (physical)', all(np.hypot(*e_[:2]) > Rb and e_[2] < 0 for e_ in delta_elbows(thd)))
out['delta'] = dict(Rb=Rb, Rp=Rp, La=La, Lr=Lr, p=pt.tolist(), th_deg=np.round(thd/deg, 4).tolist())
okd = True; rg = np.random.default_rng(3)
for _ in range(300):
    p = np.array([rg.uniform(-0.12, 0.12), rg.uniform(-0.12, 0.12), rg.uniform(-0.5, -0.25)]); t_ = delta_ik(p)
    if t_ is None: continue
    E = delta_elbows(t_)
    okd &= np.allclose([np.linalg.norm(E[i] - (p + Rp*np.array([np.cos(g), np.sin(g), 0]))) for i, g in enumerate(ang)], Lr)
check('delta IK closure holds at 300 random reachable points', okd)

# ---------- Gough-Stewart leg IK (homework p.73) ----------
bang = np.array([-10, 10, 110, 130, 230, 250])*deg; pang = np.array([-50, 50, 70, 170, 190, 290])*deg
RB, RP = 1.0, 0.5
Bp = np.array([[RB*np.cos(t), RB*np.sin(t), 0] for t in bang])
Pp = np.array([[RP*np.cos(t), RP*np.sin(t), 0] for t in pang])
BTP = Tr(0.05, -0.02, 1.0) @ Rz(5*deg) @ Ry(3*deg)
legs = np.array([(BTP @ np.append(Pp[i], 1))[:3] - Bp[i] for i in range(6)])
lens = np.linalg.norm(legs, axis=1)
print('   Stewart leg lengths (m):', np.round(lens, 4))
lens0 = np.linalg.norm(np.array([(Tr(0, 0, 1.0) @ np.append(Pp[i], 1))[:3] - Bp[i] for i in range(6)]), axis=1)
check('Stewart: at the symmetric home pose all six legs are equal', np.allclose(lens0, lens0[0]))
print(f'   home leg length = {lens0[0]:.4f} m')
out['stewart'] = dict(base_deg=[-10, 10, 110, 130, 230, 250], plat_deg=[-50, 50, 70, 170, 190, 290], RB=RB, RP=RP,
                      pose='Tr(0.05,-0.02,1.0)·Rz(5°)·Ry(3°)', lens=np.round(lens, 4).tolist(), home=r(lens0[0]))
json.dump(out, open(os.path.join(os.path.dirname(__file__), 'w04_vectors.json'), 'w'), indent=1)
report()
