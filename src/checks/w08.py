"""W8 Control part 1 - verifies every number on the W8 tab and writes widget test vectors.
Run: python3 site/src/checks/w08.py   (ends with ALL CHECKS PASSED)"""
import json, os, sys, re
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import sympy as sp
from scipy.integrate import solve_ivp

HERE = os.path.dirname(__file__)
FRAG = os.path.join(HERE, '..', 'frag', '08-w08.html')
out = {}

def roots_of(m, b, k):
    return np.roots([m, b, k])

# ---------- 1. the three deck examples (pp. 19, 22, 25): m xdd + b xd + k x = 0, x(0) = -1, xd(0) = 0 ----------
def simulate_smd(m, b, k, x0, v0, T, n=2001):
    t = np.linspace(0, T, n)
    s = solve_ivp(lambda t, y: [y[1], -(b*y[1] + k*y[0])/m], [0, T], [x0, v0], t_eval=t, rtol=1e-12, atol=1e-13)
    return t, s.y[0]

# case 1 (p. 19)
r1 = np.sort(roots_of(1, 3, 2).real)
close('p19 roots of s^2+3s+2', r1, [-2, -1])
A = np.linalg.solve([[1, 1], [-1, -2]], [-1, 0])      # A1+A2=-1 ; -A1-2A2=0
close('p19 A1,A2 = -2, 1', A, [-2, 1])
t, xs = simulate_smd(1, 3, 2, -1, 0, 12)
close('p19 x(t) = -2e^-t + e^-2t matches ODE', -2*np.exp(-t) + np.exp(-2*t), xs, tol=1e-8, show=False)
check('p19 over-damped: x rises monotonically, no overshoot', np.all(np.diff(xs) >= -1e-12) and xs.max() <= 1e-9)
disc1 = 3**2 - 4*1*2
check('p19 b^2-4mk = 1 > 0', disc1 == 1)
out['ex1'] = dict(roots=[-1, -2], A=[-2, 1], x1=r(-2*np.exp(-1) + np.exp(-2)), x3=r(-2*np.exp(-3) + np.exp(-6)))
close('p19 x(1) = -2/e + 1/e^2 = -0.6004', -2*np.exp(-1) + np.exp(-2), -0.6004, tol=5e-5)

# case 2 (p. 22)
rr = roots_of(1, 1, 1)
close('p22 roots re = -1/2', rr.real, [-0.5, -0.5]); close('p22 roots im = +-sqrt3/2', np.sort(rr.imag), [-np.sqrt(3)/2, np.sqrt(3)/2])
A1 = -1.0; A2 = -1/np.sqrt(3)     # -A1/2 + sqrt3 A2/2 = 0  -> A2 = A1/sqrt3
close('p22 A2 = A1/sqrt3 = -1/sqrt3', A2, A1/np.sqrt(3))
close('p22 A2 numeric -0.5774', A2, -0.5774, tol=5e-5)
f2 = lambda t: -np.exp(-t/2)*(np.cos(np.sqrt(3)*t/2) + np.sin(np.sqrt(3)*t/2)/np.sqrt(3))
t2, xs2 = simulate_smd(1, 1, 1, -1, 0, 14, 14001)
close('p22 closed form matches ODE', f2(t2), xs2, tol=1e-8, show=False)
ipk = np.argmax(xs2); tpk = t2[ipk]; xpk = xs2[ipk]
# exact first peak: xd = 0 -> tan(sqrt3 t/2) ... the first positive maximum occurs at t = 2pi/sqrt3 (=3.628) by symmetry of the damped sinusoid
tp_exact = 2*np.pi/np.sqrt(3)
close('p22 first peak at t = 2pi/sqrt3 = 3.628 s', tpk, tp_exact, tol=2e-3)
close('p22 first peak value = e^{-pi/sqrt3} = 0.1633', xpk, np.exp(-np.pi/np.sqrt(3)), tol=1e-4)
print('   p22 peak', xpk, 'at', tpk)
out['ex2'] = dict(peak=r(xpk), t_peak=r(tp_exact, 3), A2=r(A2))
# overshoot relative to final value 0 and initial step 1: 16.3 % of the initial offset
check('p22 undershoot after the peak exists (x goes back below 0)', xs2[ipk:].min() < -1e-3)
imin = ipk + np.argmin(xs2[ipk:int(ipk*2.2)]); print('   p22 first trough', xs2[imin], 'at', t2[imin])
# damping ratio of the same system
close('p22 zeta = b/(2 sqrt(mk)) = 0.5', 1/(2*np.sqrt(1*1)), 0.5)
close('p22 wn = sqrt(k/m) = 1', np.sqrt(1/1), 1.0)
check('p22 e^{-pi zeta/sqrt(1-zeta^2)} with zeta=.5 equals peak 0.1633', abs(np.exp(-np.pi*0.5/np.sqrt(0.75)) - xpk) < 1e-4)

# case 3 (p. 25)
close('p25 roots -2,-2', np.roots([1, 4, 4]).real, [-2, -2], tol=1e-6)
A = np.linalg.solve([[1, 0], [-2, 1]], [-1, 0])    # A1=-1 ; -2A1 + A2 = 0
close('p25 A1,A2 = -1,-2', A, [-1, -2])
t3, xs3 = simulate_smd(1, 4, 4, -1, 0, 12)
close('p25 x(t) = (-1-2t)e^-2t matches ODE', (-1 - 2*t3)*np.exp(-2*t3), xs3, tol=1e-8, show=False)
check('p25 critically damped: no overshoot', xs3.max() <= 1e-9 and np.all(np.diff(xs3) >= -1e-12))
out['ex3'] = dict(x1=r((-1-2)*np.exp(-2)))
# settling times (2 % of the initial offset, i.e. |x| <= 0.02) for the comparison table
def t_settle(t, x, tol=0.02):
    idx = np.where(np.abs(x) > tol)[0]
    return t[idx[-1]] if len(idx) else 0.0
ts = [t_settle(t, xs), t_settle(t2, xs2), t_settle(t3, xs3)]
print('   2% settling times (over, under, crit):', np.round(ts, 2))
check('settling order for the three deck examples: critical (2.92) < over (4.60) < under (8.08)', ts[2] < ts[0] < ts[1], f'{ts}')
close('2% settling times rounded: 4.60, 8.08, 2.92', [round(v, 2) for v in ts], [4.60, 8.08, 2.92], tol=1e-9)
out['settle'] = dict(over=r(ts[0], 2), under=r(ts[1], 2), crit=r(ts[2], 2))
# the fastest-decaying example is the critical one (slowest pole -2 vs -1 and -0.5)
close('slowest decay rates: over -1, under -0.5, crit -2', [max(r1), -0.5, -2], [-1, -0.5, -2])

# ---------- SVG polylines for the three-response figure (viewBox 0 0 440 230; X=40+38 t, Y=55-150 x) ----------
tt = np.linspace(0, 10, 101)
def path(xv):
    return ' '.join(f'{40+38*a:.1f},{55-150*b:.1f}' for a, b in zip(tt, xv))
paths = dict(OVER=path(-2*np.exp(-tt) + np.exp(-2*tt)), UNDER=path(f2(tt)), CRIT=path((-1-2*tt)*np.exp(-2*tt)))
out['svg_paths'] = paths

# ---------- 2. PD law on the plant (pp. 29-32) ----------
m, b, k = 2.0, 3.0, 8.0
check('plant m=2,b=3,k=8 is under-damped (b^2-4mk = -55)', b*b - 4*m*k == -55)
kp_ = 10.0
kprime = k + kp_
bprime = np.sqrt(4*m*kprime)
kv_ = bprime - b
close("PD: k' = 18, b' = 12, kv = 9 for critical damping", [kprime, bprime, kv_], [18, 12, 9])
close('PD: closed-loop double root -b\'/2m = -3', np.roots([m, b + kv_, k + kp_]).real, [-3, -3], tol=1e-6)
out['pd'] = dict(m=m, b=b, k=k, kp=kp_, kv=kv_, root=-3)
# the (wrong) servo formula applied to the raw plant
close('wrong shortcut kv = 2 sqrt(kp) = 6.3246 (P8.2 trap)', 2*np.sqrt(kp_), 6.3246, tol=5e-5)
# with kv = 6.3246 the plant is then not critical:
disc_w = (b + 2*np.sqrt(kp_))**2 - 4*m*kprime
check('with kv=6.3246 the PD loop is still under-damped (disc = -57.05 < 0)', disc_w < 0 and abs(disc_w + 57.053) < 1e-2, f'{disc_w:.3f}')

# ---------- 3. partitioned control (pp. 33-37) ----------
xx, xd_ = 0.5, 1.0
kp, kv = 25.0, 10.0
fprime = -kv*xd_ - kp*xx
f = m*fprime + b*xd_ + k*xx
close('partition: f\' = -22.5', fprime, -22.5)
close('partition: f = m f\' + b xd + k x = -38', f, -38)
xdd = (f - b*xd_ - k*xx)/m
close('partition: plant xdd = f\' (unit-mass behaviour)', xdd, fprime)
close('partition: critical roots of s^2+10s+25 -> -5,-5', np.roots([1, kv, kp]).real, [-5, -5], tol=1e-6)
check('critical damping condition is kv^2 = 4 kp (100 = 100)', kv**2 == 4*kp)
check('slide p.36 version kp^2 = 4 kv fails for these gains (625 != 40)', kp**2 != 4*kv)
# slide's version applied: choose kp=16, kv=64 from kp^2=4kv
kp_s, kv_s = 16.0, 64.0
check('slide p.36 formula kp^2=4kv holds for (16,64)', kp_s**2 == 4*kv_s)
rs = np.sort(np.roots([1, kv_s, kp_s]).real)
close('(16,64): roots -63.749, -0.251 (heavily over-damped)', rs, [-63.7490, -0.2510], tol=1e-3)
check('(16,64): kv^2 = 4096 != 4 kp = 64', kv_s**2 != 4*kp_s)
out['slide_err'] = dict(kp=kp_s, kv=kv_s, roots=[r(v, 3) for v in rs])
# general: kv = 2 sqrt(kp)
for kpv in [1, 4, 25, 100, 400]:
    kvv = 2*np.sqrt(kpv)
    assert abs(kvv**2 - 4*kpv) < 1e-9
check('kv = 2 sqrt(kp) satisfies kv^2 = 4kp for kp in 1,4,25,100,400', True)
# pole placement: s^2 + kv s + kp = (s-p1)(s-p2)
p1, p2 = -2.0, -8.0
kv_pp, kp_pp = -(p1 + p2), p1*p2
close('pole placement poles -2,-8 -> kv=10, kp=16', [kv_pp, kp_pp], [10, 16])
zz = kv_pp/(2*np.sqrt(kp_pp)); close('same gains: zeta = 1.25 (over-damped)', zz, 1.25)

# model mismatch (added): controller uses b_hat != b, same m
bh = 1.0
eff_kv = kv + (b - bh)/m
close('model error b_hat=1 (true 3): effective kv = kv + (b - b_hat)/m = 11', eff_kv, 11.0)
# verify by simulation: plant m xdd + b xd + k x = f, f = m f' + bh xd + kh x with kh=k
def mismatch_roots(kv, kp, m, b, k, mh, bh, kh):
    # m xdd + b xd + k x = mh*(-kv xd - kp x) + bh xd + kh x
    return np.roots([m, b + mh*kv - bh, k + mh*kp - kh])
rm_ = mismatch_roots(kv, kp, m, b, k, m, bh, k)
close('mismatch b_hat=1: char. poly s^2 + 11 s + 25', np.poly(rm_).real, [1, 11, 25], tol=1e-9)
# mass error: m_hat = 1.5 vs m = 2 scales the gains by m_hat/m
rmm = mismatch_roots(kv, kp, m, b, k, 1.5, b, k)
close('mismatch m_hat=1.5 (true 2): s^2 + (kv m_hat/m) s + kp m_hat/m = s^2 + 7.5 s + 18.75', np.poly(rmm).real/np.poly(rmm).real[0], [1, 7.5, 18.75], tol=1e-9)
out['mismatch'] = dict(eff_kv=eff_kv, scale=0.75, kv_eff_m=7.5, kp_eff_m=18.75)
# 7.5^2 - 4*18.75 = 56.25 - 75 < 0 -> under-damped
check('m_hat=1.5: now under-damped (56.25 - 75 < 0)', 7.5**2 - 4*18.75 < 0)

# ---------- 4. trajectory following (pp. 38-40) ----------
# unit-mass error dynamics, desired x_d = a t^2/2, with and without the xdd_d feed-forward
a_, kp_t, kv_t = 2.0, 100.0, 20.0
# with feed-forward: E'' + kv E' + kp E = 0, E(0)=0, E'(0)=0 -> E == 0
s = solve_ivp(lambda t, y: [y[1], -kv_t*y[1] - kp_t*y[0]], [0, 5], [0, 0], rtol=1e-12, atol=1e-14)
check('feed-forward: zero initial error stays zero for any x_d', np.max(np.abs(s.y[0])) < 1e-12)
# without feed-forward: E'' + kv E' + kp E = a  -> E_ss = a/kp
s = solve_ivp(lambda t, y: [y[1], -kv_t*y[1] - kp_t*y[0] + a_], [0, 5], [0, 0], rtol=1e-12, atol=1e-14)
close('no feed-forward on x_d = t^2: E_ss = a/kp = 0.02 m', s.y[0][-1], a_/kp_t, tol=1e-6)
out['ff'] = dict(a=a_, kp=kp_t, kv=kv_t, ess=a_/kp_t)
# E(0) = 0.5 decays with the critical-damped pair
s = solve_ivp(lambda t, y: [y[1], -kv_t*y[1] - kp_t*y[0]], [0, 5], [0.5, 0], rtol=1e-12, atol=1e-14, t_eval=[0.5])
w = np.sqrt(kp_t)
close('E(t) = E0 (1+wt)e^{-wt} for kp=100, kv=20 at t=0.5: 0.5*(6)*e^-5 = 0.0202', s.y[0][0], 0.5*(1+w*0.5)*np.exp(-w*0.5), tol=1e-9)
close('numeric value 0.0202', 0.5*(1+w*0.5)*np.exp(-w*0.5), 0.0202, tol=5e-5)
check('kp=100,kv=20: kv^2 = 400 = 4 kp, critical, wn = 10', kv_t**2 == 4*kp_t and w == 10)

# ---------- 5. full partitioned loop on the real plant (diagram of p.41 with its signs) ----------
# plant m xdd + b xd + k x = f - f_dist ; f = m f' + b xd + k x ; f' = xdd_d + kv E' + kp E
def full_loop(m, b, k, kp, kv, ki, fd, xdes, T=40.0, x0=0.0):
    def rhs(t, y):
        x, v, I = y   # I = integral of E
        E = xdes(t)[0] - x; Ed = xdes(t)[1] - v
        fp = xdes(t)[2] + kv*Ed + kp*E + ki*I
        f = m*fp + b*v + k*x
        a = (f - fd - b*v - k*x)/m
        return [v, a, E]
    return solve_ivp(rhs, [0, T], [x0, 0.0, 0.0], rtol=1e-11, atol=1e-13, dense_output=True)
const1 = lambda t: (1.0, 0.0, 0.0)
fd_, m_, kp_d, kv_d = 10.0, 2.0, 25.0, 10.0
sol = full_loop(2.0, 3.0, 8.0, kp_d, kv_d, 0.0, fd_, const1)
Ess = 1.0 - sol.y[0][-1]
close('full loop (m=2,b=3,k=8), f_dist=10 N, kp=25: E_ss = f/(kp m) = 0.2 m', Ess, fd_/(kp_d*m_), tol=1e-7)
# the steady error does not depend on b, k, only on m: change b,k
sol2 = full_loop(2.0, 7.0, 1.0, kp_d, kv_d, 0.0, fd_, const1)
close('E_ss unchanged when plant b,k change', 1.0 - sol2.y[0][-1], 0.2, tol=1e-7)
# depends on m
sol3 = full_loop(5.0, 3.0, 8.0, kp_d, kv_d, 0.0, fd_, const1)
close('E_ss halves-ish with m: m=5 -> 10/(25*5) = 0.08 m', 1.0 - sol3.y[0][-1], 0.08, tol=1e-7)
# 5 mm requirement
kp_req = fd_/(m_*0.005); kv_req = 2*np.sqrt(kp_req)
close('E_ss <= 5 mm needs kp >= 1000 (f=10 N, m=2 kg)', kp_req, 1000.0)
close('then critical kv = 2 sqrt(1000) = 63.25', kv_req, 63.2456, tol=5e-5)
out['dist'] = dict(f=fd_, m=m_, kp=kp_d, ess=0.2, ess_m5=0.08, kp_req=kp_req, kv_req=r(kv_req, 3))
# PID on the plant: E_ss -> 0 and the integral settles at d/ki
ki_ = 40.0
solp = full_loop(2.0, 3.0, 8.0, kp_d, kv_d, ki_, fd_, const1, T=60)
close('PID: E -> 0 with constant disturbance', 1.0 - solp.y[0][-1], 0.0, tol=1e-7)
close('PID: integral of E settles at f_dist/(m ki) = 0.125', solp.y[2][-1], fd_/(m_*ki_), tol=1e-7)
out['pid'] = dict(ki=ki_, int_final=fd_/(m_*ki_))

# ---------- 6. PID third-order equation and its stability (pp. 44-45) ----------
s_ = sp.symbols('s'); kp_s_, kv_s_, ki_s_ = sp.symbols('kp kv ki', positive=True)
cub = sp.expand(s_**3 + kv_s_*s_**2 + kp_s_*s_ + ki_s_)
check('third-order char. poly is s^3 + kv s^2 + kp s + ki', cub.coeff(s_, 2) == kv_s_ and cub.coeff(s_, 1) == kp_s_ and cub.coeff(s_, 0) == ki_s_)
# Routh: stable iff kv kp > ki. at kv kp = ki: (s+kv)(s^2+kp)
fac = sp.factor(cub.subs(ki_s_, kp_s_*kv_s_))
check('at ki = kv kp the polynomial factors as (s+kv)(s^2+kp)', sp.simplify(fac - (s_ + kv_s_)*(s_**2 + kp_s_)) == 0)
for kiv, expect in [(40, True), (250, False), (300, False), (200, True)]:
    rts = np.roots([1, 10, 25, kiv]); st = bool(np.all(rts.real < -1e-9))
    check(f'kp=25,kv=10,ki={kiv}: stable={expect} (limit kv*kp = 250)', st == expect, f'max Re = {rts.real.max():.4f}')
r3 = np.roots([1, 10, 25, 40]); r3 = r3[np.argsort(r3.real)]
print('   roots kp=25,kv=10,ki=40:', np.round(r3, 4))
out['pid_roots'] = [[r(v.real, 4), r(v.imag, 4)] for v in r3]
rr250 = np.roots([1, 10, 25, 250]); close('ki=250 real parts -10, 0, 0 (marginal)', np.sort(rr250.real), [-10, 0, 0], tol=1e-6); close('ki=250 imaginary parts 0, +-5', np.sort(rr250.imag), [-5, 0, 5], tol=1e-6)

# ---------- 7. zeta / omega_n links (added) ----------
def zw(kp, kv): return kv/(2*np.sqrt(kp)), np.sqrt(kp)
for kp_, kv_, ze, wn in [(100, 20, 1.0, 10.0), (25, 6, 0.6, 5.0), (100, 10, 0.5, 10.0), (16, 10, 1.25, 4.0)]:
    z, w = zw(kp_, kv_); close(f'kp={kp_}, kv={kv_}: zeta={ze}, wn={wn}', [z, w], [ze, wn])
rt = np.roots([1, 6, 25]); close('kp=25,kv=6: roots re = -3 (= -zeta wn)', rt.real, [-3, -3]); close('kp=25,kv=6: roots im = +-4 (= wn sqrt(1-zeta^2))', np.sort(rt.imag), [-4, 4])
close('overshoot for zeta=0.6: exp(-pi*0.6/0.8) = 0.0948', np.exp(-np.pi*0.6/np.sqrt(1-0.36)), 0.0948, tol=5e-5)
s_ov = solve_ivp(lambda t, y: [y[1], -6*y[1] - 25*y[0]], [0, 3], [-1, 0], rtol=1e-12, atol=1e-14, dense_output=True)
tt_ = np.linspace(0, 3, 30001); close('overshoot confirmed by simulation (from x0=-1)', s_ov.sol(tt_)[0].max(), np.exp(-np.pi*0.6/0.8), tol=2e-5)
# design example: wn = 10 rad/s critical -> kp=100, kv=20
close('design: critical with wn=10 rad/s -> kp=wn^2=100, kv=2 wn=20', [10**2, 2*10], [100, 20])
# design: zeta=0.7, wn=8: kp=64, kv=2*0.7*8=11.2
close('design: zeta=0.7, wn=8 -> kp=64, kv=11.2', [8**2, 2*0.7*8], [64, 11.2])
out['design'] = dict(crit=[100, 20], z07=[64, 11.2])
# real plant (not unit mass), damping ratio of m x'' + b x' + k x
close('plant m=2,b=3,k=8: zeta = b/(2 sqrt(mk)) = 0.375', 3/(2*np.sqrt(2*8)), 0.375)
close('plant m=2,b=3,k=8: wn = sqrt(k/m) = 2', np.sqrt(8/2), 2.0)

# ---------- 8. DC motor and gearing (pp. 48-50) ----------
q, qm = sp.symbols('q q_m'); I_, Im_, c_, cm_, eta, t_ = sp.symbols('I I_m c c_m eta t', positive=True)
qa = sp.Function('qa')(t_)
# motor shaft: Im qm'' + cm qm' + (reaction torque from the load) = Qm ; reaction at the motor = Q_axle / eta ; q_m = eta q_a
qm_t = eta*qa
Qaxle = I_*qa.diff(t_, 2) + c_*qa.diff(t_)                     # what the load needs at the axle
Qm_needed = Im_*qm_t.diff(t_, 2) + cm_*qm_t.diff(t_) + Qaxle/eta
slide = (I_ + eta**2*Im_)*qa.diff(t_, 2) + (c_ + eta**2*cm_)*qa.diff(t_)
check('derivation: eta * Qm = (I + eta^2 Im) qa\'\' + (c + eta^2 cm) qa\'  (p.50)', sp.simplify(eta*Qm_needed - slide) == 0)
eta_v, Im_v, I_v, c_v, cm_v = 50.0, 2e-5, 0.04, 0.1, 1e-4
Ieff = I_v + eta_v**2*Im_v; ceff = c_v + eta_v**2*cm_v
close('gear example: I_eff = 0.04 + 2500*2e-5 = 0.09 kg m^2', Ieff, 0.09)
close('gear example: c_eff = 0.1 + 2500*1e-4 = 0.35 N m s/rad', ceff, 0.35)
qdd, qd = 10.0, 2.0
Qax = Ieff*qdd + ceff*qd
close('gear example: Q = 0.09*10 + 0.35*2 = 1.6 N m (axle side)', Qax, 1.6)
Qm_v = Qax/eta_v
close('gear example: Qm = Q/eta = 0.032 N m', Qm_v, 0.032)
k4 = 0.04
close('gear example: i_a = Qm/k4 = 0.8 A', Qm_v/k4, 0.8)
check('gear example: motor-side inertia 0.05 exceeds the link 0.04 (eta^2 Im > I)', eta_v**2*Im_v > I_v)
out['gear'] = dict(eta=eta_v, Im=Im_v, I=I_v, c=c_v, cm=cm_v, Ieff=Ieff, ceff=ceff, Q=Qax, Qm=Qm_v, ia=Qm_v/k4)
# motor speed
close('gear example: motor speed = eta * 2 rad/s = 100 rad/s', eta_v*qd, 100.0)
# P8.8 style: eta = 100, Im=1e-5 ; I=0.02
close('quiz: eta=100, Im=1e-5 -> reflected 0.1', 100**2*1e-5, 0.1)

# ---------- 8b. guided problems and quiz numbers ----------
# P8.1: m=1,b=5,k=6, x(0)=2, xd(0)=0
rts = np.sort(np.roots([1, 5, 6]).real); close('P8.1 roots -3, -2', rts, [-3, -2])
A = np.linalg.solve([[1, 1], [-2, -3]], [2, 0]); close('P8.1 A1=6, A2=-4', A, [6, -4])
tq, xq = simulate_smd(1, 5, 6, 2, 0, 3, 3001)
close('P8.1 closed form matches ODE', 6*np.exp(-2*tq) - 4*np.exp(-3*tq), xq, tol=1e-8, show=False)
close('P8.1 x(1) = 6e^-2 - 4e^-3 = 0.6129', 6*np.exp(-2) - 4*np.exp(-3), 0.6129, tol=5e-5)
check('P8.1 b^2-4mk = 1 > 0 (over-damped)', 25 - 24 == 1)
# P8.4: student used the slide formula; correct kv for kp = 16
close('P8.4 correct kv = 2 sqrt(16) = 8', 2*np.sqrt(16), 8.0)
close('P8.4 (16, 8): double root -4', np.roots([1, 8, 16]).real, [-4, -4], tol=1e-6)
# P8.7: kp = 25 with kv = 4, 10, 20
rr4 = np.roots([1, 4, 25]); close('P8.7 kv=4: -2 +- 4.583i', [rr4.real[0], abs(rr4.imag[0])], [-2, np.sqrt(21)], tol=1e-9)
close('P8.7 sqrt(21) = 4.583', np.sqrt(21), 4.583, tol=5e-4)
close('P8.7 kv=4: zeta = 0.4', 4/(2*5), 0.4)
close('P8.7 kv=4: overshoot exp(-pi 0.4/sqrt(0.84)) = 0.2538', np.exp(-np.pi*0.4/np.sqrt(0.84)), 0.2538, tol=5e-5)
rr20 = np.sort(np.roots([1, 20, 25]).real); close('P8.7 kv=20: -18.66, -1.34', rr20, [-(10 + np.sqrt(75)), -(10 - np.sqrt(75))])
close('P8.7 kv=20 numeric -18.660, -1.340', rr20, [-18.660, -1.340], tol=5e-4)
close('P8.7 kv=20: zeta = 2', 20/(2*5), 2.0)
# P8.8: wrong answers
close('P8.8 wrong (no eta^2): Q_m = 0.012', ((0.04 + 2e-5)*10 + (0.1 + 1e-4)*2)/50, 0.012, tol=2e-5)
close('warn-box: slide formula gives kv = kp^2/4 = 156.25 = 625/4 for kp=25', 25**2/4, 625/4)
close('(16,64) slow pole time constant 1/0.251 = 3.98 s (about 4 s)', 1/0.250984, 3.98, tol=0.01)
rr9 = np.sort(np.roots([1, 11, 25]).real); close('P8.9 poles -5.5 +- sqrt(21)/2', rr9, [-5.5 - np.sqrt(21)/2, -5.5 + np.sqrt(21)/2])
check('P8.9 11^2 = 121 > 4 kp = 100: slightly over-damped', 11**2 > 100)
# Q2 quiz: steady-state error
close('Q2 E_ss = 6/(50*3) = 0.04 m', 6/(50*3), 0.04)
# open loop: an initial offset never decays (E'' = 0)
sol_o = solve_ivp(lambda t, y: [y[1], 0.0], [0, 5], [0.1, 0.0], rtol=1e-12, atol=1e-14)
close('open loop: E(5) = E(0) = 0.1 rad (no correction)', sol_o.y[0][-1], 0.1)
# integral limit
check('kp=25, kv=10: integral gain limit kv*kp = 250', 25*10 == 250)
# overdamped example: kp=25,kv=20 etc. used in widget presets
close('preset kp=25,kv=4: zeta = 0.4; kp=25,kv=20: zeta = 2', [4/(2*5), 20/(2*5)], [0.4, 2.0])
# the quoted overshoot of the ELI5-free text: 16.3 % for the p.22 example
close('p22 overshoot percent 16.3', 100*np.exp(-np.pi/np.sqrt(3)), 16.3, tol=0.05)

# ---------- 9. widget test vectors ----------
def servo_ref(kp, kv, ki, f, m_):
    d = f/m_
    disc = kv*kv - 4*kp
    zeta = kv/(2*np.sqrt(kp)); wn = np.sqrt(kp)
    if kv == 0: regime = 'undamped'
    elif abs(disc) < 1e-9: regime = 'critically damped'
    elif disc > 0: regime = 'overdamped'
    else: regime = 'underdamped'
    r2 = np.roots([1, kv, kp])
    r2 = r2[np.lexsort((r2.imag, r2.real))]
    ess_pd = d/kp
    stable3 = bool(kv > 0 and kp > 0 and ki > 0 and kv*kp > ki) if ki > 0 else True
    r3 = np.roots([1, kv, kp, ki]) if ki > 0 else np.array([])
    if len(r3): r3 = r3[np.lexsort((r3.imag, r3.real))]
    def run(kiv):
        s = solve_ivp(lambda t, y: [y[1], -kv*y[1] - kp*y[0] - kiv*y[2] + d, y[0]], [0, 5], [1.0, 0.0, 0.0],
                      rtol=1e-11, atol=1e-13, t_eval=[1.0, 2.5, 5.0])
        return s.y[0].tolist()
    os_pct = float(np.exp(-np.pi*zeta/np.sqrt(1-zeta**2))*100) if 0 < zeta < 1 else 0.0
    return dict(kp=kp, kv=kv, ki=ki, f=f, m=m_, d=d, zeta=r(zeta, 6), wn=r(wn, 6), regime=regime,
                roots2=[[r(z.real, 6), r(z.imag, 6)] for z in r2], ess_pd=r(ess_pd, 6),
                stable3=stable3, roots3=[[r(z.real, 6), r(z.imag, 6)] for z in r3],
                e_pd=[r(v, 6) for v in run(0.0)], e_pid=[r(v, 6) for v in run(ki)], os_pct=r(os_pct, 4))
cases = [(25, 10, 40, 10, 2), (100, 20, 0, 0, 1), (25, 6, 100, 5, 1), (16, 10, 20, 4, 2), (25, 10, 300, 10, 2), (9, 0, 0, 0, 1), (400, 40, 1000, 0, 1)]
vecs = [servo_ref(*c) for c in cases]
for v in vecs: print('   vec', v['kp'], v['kv'], v['ki'], v['regime'], 'ess_pd', v['ess_pd'], 'stable3', v['stable3'], 'e_pid(5)', v['e_pid'][-1])
json.dump(dict(out, vectors=vecs), open(os.path.join(HERE, 'w08_vectors.json'), 'w'), indent=1)
# default widget state used in the page text
dv = vecs[0]
close('widget default: kp=25,kv=10 critical, ess_pd = 10/(25*2) = 0.2', dv['ess_pd'], 0.2)
check('widget default regime = critically damped', dv['regime'] == 'critically damped')

# ---------- 10. the page must contain the exact SVG polylines computed here ----------
if os.path.exists(FRAG):
    html = open(FRAG, encoding='utf-8').read()
    for k_, p_ in paths.items():
        check(f'frag contains the {k_} response polyline', p_ in html)
else:
    print('   (frag not written yet; skipping polyline check)')
report()
