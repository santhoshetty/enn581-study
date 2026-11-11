"""W5 Trajectories: verifies every number on the W5 tab. Run: python3 site/src/checks/w05.py
Also writes checks/w05_vectors.json (test vectors for the W3D 'traj' widget) and, if the fragment exists,
runs the widget's JavaScript core in node against those vectors."""
import json, os, subprocess, sys, shutil, re
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}

# ------------------------------------------------------------------ helpers
def cubic_coeffs(q0, qf, v0, vf, T):
    d = qf - q0
    return np.array([q0, v0, (3*d - (2*v0 + vf)*T)/T**2, (-2*d + (v0 + vf)*T)/T**3])

def poly(c, t, der=0):
    c = np.array(c, float); n = len(c)
    t = np.asarray(t, float)
    s = np.zeros_like(t)
    for k in range(der, n):
        f = 1.0
        for j in range(der): f *= (k - j)
        s = s + c[k]*f*t**(k - der)
    return s

# ------------------------------------------------------------------ 1. cubic: symbolic derivation
a0, a1, a2, a3, T, q0s, qfs, v0s, vfs, t = sp.symbols('a0 a1 a2 a3 T q0 qf v0 vf t')
qs = a0 + a1*t + a2*t**2 + a3*t**3
qds = sp.diff(qs, t)
sol = sp.solve([qs.subs(t, 0) - q0s, qs.subs(t, T) - qfs, qds.subs(t, 0) - v0s, qds.subs(t, T) - vfs], [a0, a1, a2, a3], dict=True)[0]
d = qfs - q0s
check('cubic a2 = [3d - (2v0+vf)T]/T^2', sp.simplify(sol[a2] - (3*d - (2*v0s + vfs)*T)/T**2) == 0)
check('cubic a3 = [-2d + (v0+vf)T]/T^3', sp.simplify(sol[a3] - (-2*d + (v0s + vfs)*T)/T**3) == 0)
rr = {k: sp.simplify(v.subs({v0s: 0, vfs: 0})) for k, v in sol.items()}
check('rest-to-rest a2 = 3d/T^2, a3 = -2d/T^3', sp.simplify(rr[a2] - 3*d/T**2) == 0 and sp.simplify(rr[a3] + 2*d/T**3) == 0)

# Example 1 (slides pp.17-19, notebook cell 15)
c = cubic_coeffs(10, 50, 0, 0, 2)
close('Example 1 coefficients (10, 0, 30, -10)', c, [10, 0, 30, -10])
A = np.array([[1, 0, 0, 0], [1, 2, 4, 8], [0, 1, 0, 0], [0, 1, 4, 12.]])
close('Example 1 via notebook matrix solve', np.linalg.solve(A, [10, 50, 0, 0]), [10, 0, 30, -10])
tt = np.linspace(0, 2, 2001)
check('Example 1 peak velocity 30 deg/s at t = 1 s (= 1.5 d/T)', np.isclose(poly(c, tt, 1).max(), 30) and np.isclose(tt[poly(c, tt, 1).argmax()], 1.0) and np.isclose(1.5*40/2, 30))
close('Example 1 acceleration +60 at t=0, -60 at t=2 (p.19)', [poly(c, 0, 2), poly(c, 2, 2)], [60, -60])
check('Example 1 zero acceleration at the midpoint t = 1', np.isclose(poly(c, 1, 2), 0))
out['cubic_rest'] = dict(q0=10, qf=50, T=2, v0=0, vf=0, coeffs=c.tolist())

# P5.1: 20 -> 80 deg in 3 s rest to rest
c51 = cubic_coeffs(20, 80, 0, 0, 3)
close('P5.1 coefficients', c51, [20, 0, 20, -4.444444444])
tt3 = np.linspace(0, 3, 3001)
check('P5.1 peak velocity 30 deg/s at 1.5 s', np.isclose(poly(c51, tt3, 1).max(), 30) and np.isclose(tt3[poly(c51, tt3, 1).argmax()], 1.5))
close('P5.1 end accelerations +40, -40', [poly(c51, 0, 2), poly(c51, 3, 2)], [40, -40])
close('P5.1 q(1) = 20 + 20 - 4.4444 = 35.5556', poly(c51, 1.0), 35.5555556, 1e-6)

# non-zero end velocities
c_g = cubic_coeffs(20, 80, 0, 10, 3)
a2g = (3*60 - (2*0 + 10)*3)/9; a3g = (-2*60 + (0 + 10)*3)/27
close('cubic general coefficients', c_g, [20, 0, a2g, a3g])
print(f'   general cubic: a2 = {a2g:.6f}, a3 = {a3g:.6f}')
check('general cubic hits all four conditions', np.isclose(poly(c_g, 0), 20) and np.isclose(poly(c_g, 3), 80) and np.isclose(poly(c_g, 0, 1), 0) and np.isclose(poly(c_g, 3, 1), 10))
out['cubic_general'] = dict(q0=20, qf=80, T=3, v0=0, vf=10, coeffs=c_g.tolist())

# clock shift: t0 != 0 (use tau = t - t0)
c_sh = cubic_coeffs(10, 50, 0, 0, 2)  # same motion, t0 = 1.5, tF = 3.5
check('clock shift: q(t0 + tau) with tau = t - t0 gives the same numbers', np.isclose(poly(c_sh, (3.5 - 1.5)), 50) and np.isclose(poly(c_sh, 0), 10))

# ------------------------------------------------------------------ 2. via points (Q2): 10, 50, 20, 40, 45 at 0, 5, 10, 15, 20
qv = np.array([10, 50, 20, 40, 45.]); tv = np.array([0, 5, 10, 15, 20.])
slopes = np.diff(qv)/np.diff(tv)
close('Q2 slopes 8, -6, 4, 1 deg/s', slopes, [8, -6, 4, 1])
vel = np.zeros(5)
for i in range(1, 4):
    vel[i] = 0.0 if slopes[i-1]*slopes[i] < 0 else 0.5*(slopes[i-1] + slopes[i])
close('Q2 heuristic via velocities (0, 0, 0, 2.5, 0)', vel, [0, 0, 0, 2.5, 0])
segs = [cubic_coeffs(qv[i], qv[i+1], vel[i], vel[i+1], 5) for i in range(4)]
for i, s in enumerate(segs): print(f'   Q2 segment {i+1} (local time): a = {np.round(s, 6).tolist()}')
close('Q2 segment 1 = 10 + 0t + 6t^2 - 0.8t^3 (a2 = 3*40/25 = 4.8)', segs[0], [10, 0, 4.8, -0.64])
close('Q2 segment 4 (C to F) a2 = -0.4, a3 = 0.02', segs[3], [40, 2.5, -0.4, 0.02])
ok = all(np.isclose(poly(segs[i], 5), qv[i+1]) and np.isclose(poly(segs[i], 5, 1), vel[i+1]) and np.isclose(poly(segs[i+1], 0, 1), vel[i+1]) for i in range(3))
check('Q2 segments continuous in q and in velocity at all via points', ok)
jumps = [poly(segs[i], 5, 2) - poly(segs[i+1], 0, 2) for i in range(3)]
print('   Q2 acceleration jumps at A, B, C:', np.round(jumps, 4).tolist())
check('Q2 acceleration is NOT continuous at the via points (jumps non-zero)', all(abs(j) > 1e-6 for j in jumps))
out['q2'] = dict(q=qv.tolist(), t=tv.tolist(), slopes=slopes.tolist(), vel=vel.tolist(), seg_coeffs=[s.tolist() for s in segs], acc_jumps=[float(j) for j in jumps])

# ------------------------------------------------------------------ 3. Q3: continuous acceleration at the via point
T1, T2, q0, qvia, qF, vv = sp.symbols('T1 T2 q0 qvia qF vv', positive=True)
def cub_sym(qa, qb, va, vb, TT):
    dd = qb - qa
    return (qa, va, (3*dd - (2*va + vb)*TT)/TT**2, (-2*dd + (va + vb)*TT)/TT**3)
c1 = cub_sym(q0, qvia, 0, vv, T1); c2 = cub_sym(qvia, qF, vv, 0, T2)
acc1 = 2*c1[2] + 6*c1[3]*T1       # end of segment 1
acc2 = 2*c2[2]                     # start of segment 2
vv_sol = sp.solve(sp.Eq(acc1, acc2), vv)[0]
vv_form = 3*(T1**2*(qF - qvia) + T2**2*(qvia - q0))/(2*T1*T2*(T1 + T2))
check('Q3 symbolic: v_via = 3[T1^2(qF-qv) + T2^2(qv-q0)] / [2 T1 T2 (T1+T2)]', sp.simplify(vv_sol - vv_form) == 0)
Tp = sp.Symbol('T', positive=True)
check('Q3 symbolic: equal T gives 3(qF-q0)/(4T)', sp.simplify(vv_sol.subs({T1: Tp, T2: Tp}) - 3*(qF - q0)/(4*Tp)) == 0)
# the common acceleration at the via point
a_via = sp.simplify(acc2.subs(vv, vv_sol))
print('   Q3 acceleration at via point (symbolic):', a_via)
a_via_eq = sp.simplify(a_via.subs({T1: Tp, T2: Tp}))
print('   Q3 equal-T acceleration at via point:', a_via_eq)

def vvia_cont(q0_, qv_, qf_, T1_, T2_):
    return 3*(T1_**2*(qf_ - qv_) + T2_**2*(qv_ - q0_))/(2*T1_*T2_*(T1_ + T2_))
v_eq = vvia_cont(10, 40, 20, 1, 1)
close('Q3 numeric (sample Q8): 10, 40, 20, T = 1: v_via = 7.5 deg/s', v_eq, 7.5)
close('Q3 equal T closed form 3(20-10)/(4*1) = 7.5', 3*(20 - 10)/4, 7.5)
v_un = vvia_cont(10, 40, 20, 1, 2)
close('Q3 unequal T1 = 1, T2 = 2: v_via = 25 deg/s', v_un, 25.0)
# full numeric check with an 8x8 linear solve (8 unknowns, 8 conditions), as in the sample-question solution
def solve8(q0_, qv_, qf_, T1_, T2_):
    A = np.zeros((8, 8)); b = np.zeros(8)
    A[0, 0] = 1; b[0] = q0_                                        # q1(0)=q0
    A[1, 1] = 1; b[1] = 0                                          # qd1(0)=0
    A[2, :4] = [1, T1_, T1_**2, T1_**3]; b[2] = qv_                # q1(T1)=qv
    A[3, 4] = 1; b[3] = qv_                                        # q2(0)=qv
    A[4, 4:] = [1, T2_, T2_**2, T2_**3]; b[4] = qf_                # q2(T2)=qf
    A[5, 4:] = [0, 1, 2*T2_, 3*T2_**2]; b[5] = 0                   # qd2(T2)=0
    A[6, :4] = [0, 1, 2*T1_, 3*T1_**2]; A[6, 5] = -1               # velocity continuity
    A[7, :4] = [0, 0, 2, 6*T1_]; A[7, 6] = -2                      # acceleration continuity
    return np.linalg.solve(A, b)
x8 = solve8(10, 40, 20, 1, 1)
close('Q3 8x8 solve (equal T): matches sample Q8 coefficients', x8, [10, 0, 82.5, -52.5, 40, 7.5, -75, 47.5])
x8u = solve8(10, 40, 20, 1, 2)
close('Q3 8x8 solve (T1=1, T2=2): a21 = 25', x8u[5], 25.0)
cA = cubic_coeffs(10, 40, 0, v_un, 1); cB = cubic_coeffs(40, 20, v_un, 0, 2)
close('Q3 closed-form segments match the 8x8 solve (unequal T)', np.r_[cA, cB], x8u)
check('Q3 unequal T: accelerations equal at the via point', np.isclose(poly(cA, 1, 2), poly(cB, 0, 2)))
print(f'   Q3 unequal T acceleration at via point = {poly(cA, 1, 2):.4f} deg/s^2; equal T (sample Q8) = {poly(np.array(x8[:4]), 1, 2):.4f}')
out['q3'] = dict(q0=10, qv=40, qf=20, equalT=dict(T1=1, T2=1, vv=v_eq, coeffs=x8.tolist()), unequalT=dict(T1=1, T2=2, vv=v_un, coeffs=np.r_[cA, cB].tolist()))

# continuous-acceleration spline through the five Q2 points, rest at both ends (option 3 on p.26)
n = 5; h = np.diff(tv)
A = np.zeros((n, n)); b = np.zeros(n)
A[0, 0] = 1; A[-1, -1] = 1
for i in range(1, n - 1):
    A[i, i-1] = 1/h[i-1]; A[i, i] = 2*(1/h[i-1] + 1/h[i]); A[i, i+1] = 1/h[i]
    b[i] = 3*(slopes[i-1]/h[i-1] + slopes[i]/h[i])
vcc = np.linalg.solve(A, b)
segcc = [cubic_coeffs(qv[i], qv[i+1], vcc[i], vcc[i+1], h[i]) for i in range(4)]
acc_ok = all(np.isclose(poly(segcc[i], h[i], 2), poly(segcc[i+1], 0, 2)) for i in range(3))
check('Q2 data, option 3: continuous-acceleration velocities give matching accelerations at A, B, C', acc_ok)
print('   Q2 data, continuous-acceleration via velocities (deg/s):', np.round(vcc, 4).tolist())
out['q2_contacc_vel'] = [float(v) for v in vcc]

# ------------------------------------------------------------------ 4. quintic
b0, b1, b2, b3, b4, b5 = sp.symbols('b0:6')
qq = sum(bk*t**k for k, bk in enumerate([b0, b1, b2, b3, b4, b5]))
conds = [qq.subs(t, 0) - q0s, sp.diff(qq, t).subs(t, 0) - v0s, sp.diff(qq, t, 2).subs(t, 0),
         qq.subs(t, T) - qfs, sp.diff(qq, t).subs(t, T) - vfs, sp.diff(qq, t, 2).subs(t, T)]
sq = sp.solve(conds, [b0, b1, b2, b3, b4, b5], dict=True)[0]
hh = qfs - q0s
form = {b3: (20*hh - (8*vfs + 12*v0s)*T)/(2*T**3), b4: (-30*hh + (14*vfs + 16*v0s)*T)/(2*T**4), b5: (12*hh - 6*(vfs + v0s)*T)/(2*T**5)}
check('quintic (zero end accelerations) closed forms a3, a4, a5', all(sp.simplify(sq[k] - v) == 0 for k, v in form.items()) and sq[b2] == 0)
sq_rr = {k: sp.simplify(v.subs({v0s: 0, vfs: 0})) for k, v in sq.items()}
check('quintic rest-to-rest: a3 = 10d/T^3, a4 = -15d/T^4, a5 = 6d/T^5', sp.simplify(sq_rr[b3] - 10*hh/T**3) == 0 and sp.simplify(sq_rr[b4] + 15*hh/T**4) == 0 and sp.simplify(sq_rr[b5] - 6*hh/T**5) == 0)

def quintic_coeffs(q0_, qf_, v0_, vf_, T_):
    h_ = qf_ - q0_
    return np.array([q0_, v0_, 0, (20*h_ - (8*vf_ + 12*v0_)*T_)/(2*T_**3), (-30*h_ + (14*vf_ + 16*v0_)*T_)/(2*T_**4), (12*h_ - 6*(vf_ + v0_)*T_)/(2*T_**5)])
cq = quintic_coeffs(0, 15, 0, 0, 1)
close('P5.12 quintic 0 -> 15 deg in 1 s: (0, 0, 0, 150, -225, 90)', cq, [0, 0, 0, 150, -225, 90])
t1 = np.linspace(0, 1, 100001)
vq = poly(cq, t1, 1); aq = poly(cq, t1, 2)
check('quintic peak velocity 1.875 d/T = 28.125 deg/s at T/2', np.isclose(vq.max(), 28.125, atol=1e-6) and np.isclose(t1[vq.argmax()], 0.5, atol=1e-4))
cc3 = cubic_coeffs(0, 15, 0, 0, 1)
check('same move with a cubic: peak velocity 1.5 d/T = 22.5 deg/s', np.isclose(poly(cc3, t1, 1).max(), 22.5, atol=1e-6))
check('quintic: zero acceleration at both ends; cubic: +/-90 deg/s^2', np.isclose(aq[0], 0) and np.isclose(aq[-1], 0, atol=1e-6) and np.isclose(poly(cc3, 0, 2), 90) and np.isclose(poly(cc3, 1, 2), -90))
check('quintic peak |acceleration| = 10/sqrt(3) d/T^2 = 86.60 deg/s^2', np.isclose(np.abs(aq).max(), 10*15/np.sqrt(3), atol=1e-3))
close('quintic ratio mean/peak speed = 1/1.875 = 53.3 %', 100*(15/1)/28.125, 53.3333333, 1e-4)
close('cubic ratio mean/peak speed = 1/1.5 = 66.7 %', 100*(15/1)/22.5, 66.6666667, 1e-4)
out['quintic_rest'] = dict(q0=0, qf=15, T=1, v0=0, vf=0, coeffs=cq.tolist())
cqg = quintic_coeffs(0, 15, 10, 5, 1)
check('quintic with end velocities 10 and 5 meets all six conditions', np.isclose(poly(cqg, 0), 0) and np.isclose(poly(cqg, 1), 15) and np.isclose(poly(cqg, 0, 1), 10) and np.isclose(poly(cqg, 1, 1), 5) and np.isclose(poly(cqg, 0, 2), 0) and np.isclose(poly(cqg, 1, 2), 0))

# ------------------------------------------------------------------ 5. LSPB, single segment
def lspb(q0_, qf_, tF, a, t0=0.0):
    """linear segment with parabolic blends; returns dict. Uses duration T = tF - t0 (clock shift)."""
    Tt = tF - t0; D = abs(qf_ - q0_); s = np.sign(qf_ - q0_)
    amin = 4*D/Tt**2
    if a < amin - 1e-12: return dict(ok=False, amin=amin)
    tB = Tt/2 - np.sqrt(max(a*a*Tt*Tt - 4*a*D, 0))/(2*a)
    v = a*tB
    return dict(ok=True, amin=amin, tB=tB, v=v, coast=Tt - 2*tB, s=s, tB_plus=Tt/2 + np.sqrt(max(a*a*Tt*Tt - 4*a*D, 0))/(2*a))

def lspb_profile(q0_, qf_, tF, a, n=40001):
    L = lspb(q0_, qf_, tF, a); s = L['s']; tB, v = L['tB'], L['v']
    tt = np.linspace(0, tF, n)
    acc = np.where(tt < tB, s*a, np.where(tt > tF - tB, -s*a, 0.0))
    vel = np.concatenate([[0], np.cumsum((acc[1:] + acc[:-1])/2*np.diff(tt))])
    pos = q0_ + np.concatenate([[0], np.cumsum((vel[1:] + vel[:-1])/2*np.diff(tt))])
    return tt, pos, vel, acc

# the slide equations, solved symbolically (t0 = 0)
a_, tB_, tF_, dq_, v_ = sp.symbols('a t_B t_F dq v', positive=True)
qB1 = a_*tB_**2/2
qB2 = qB1 + (a_*tB_)*(tF_ - 2*tB_)
qFe = qB2 + (a_*tB_)*tB_ - a_*tB_**2/2
quad = sp.expand(qFe - dq_)
check('slide eqs (t0 = 0) reduce to a tB^2 - a tF tB + dq = 0', sp.simplify(quad + (a_*tB_**2 - a_*tF_*tB_ + dq_) - 0) == 0 or sp.simplify(-quad - (a_*tB_**2 - a_*tF_*tB_ + dq_)) == 0 or sp.expand(quad - (-(a_*tB_**2 - a_*tF_*tB_ + dq_))) == 0)
disc = a_**2*tF_**2 - 4*a_*dq_
check('discriminant a^2 tF^2 - 4 a dq >= 0  <=>  a >= 4 dq / tF^2', sp.simplify(sp.solve(sp.Ge(disc, 0), a_)) is not None and sp.simplify(disc.subs(a_, 4*dq_/tF_**2)) == 0 and sp.simplify(disc.subs(a_, 5*dq_/tF_**2)) > 0)

# p.37 example
L = lspb(0, 120, 2, 300)
print(f"   LSPB example: amin = {L['amin']}, tB = {L['tB']:.6f}, v = {L['v']:.6f}, coast = {L['coast']:.6f}, other root = {L['tB_plus']:.6f}")
close('p.37: minimum acceleration 4*120/4 = 120 deg/s^2, and 300 >= 120', [L['amin'], float(300 >= L['amin'])], [120, 1])
close('p.37: tB = 1 - sqrt(216000)/600 = 0.2254 s', L['tB'], 1 - np.sqrt(216000)/600, 1e-12)
close('p.37: tB = 0.22540 s, v = a tB = 67.62 deg/s', [L['tB'], L['v']], [0.225403, 67.6209], 1e-3)
close('p.37: closed form v = [a tF - sqrt(a^2 tF^2 - 4 a dq)]/2', L['v'], (300*2 - np.sqrt(300**2*4 - 4*300*120))/2, 1e-9)
close('p.37: coast duration tF - 2 tB = 1.5492 s', L['coast'], 1.549194, 1e-5)
check('p.37: the + root tB = 1.7746 s exceeds tF/2 and is rejected', np.isclose(L['tB_plus'], 1.774597, atol=1e-5) and L['tB_plus'] > 1)
check('p.37: coast velocity lies between mean 60 and 120 deg/s (dq/tF < v <= 2 dq/tF)', 60 < L['v'] <= 120)
tt, pos, vel, acc = lspb_profile(0, 120, 2, 300)
check('p.37: numerical integration of the trapezoid ends at 120 deg with zero velocity', np.isclose(pos[-1], 120, atol=1e-2) and np.isclose(vel[-1], 0, atol=1e-2) and np.isclose(vel.max(), L['v'], atol=1e-2))
close('p.37: blend covers a tB^2/2 = 7.6209 deg; coast covers v*coast = 104.7582', [300*L['tB']**2/2, L['v']*L['coast']], [7.6209, 104.7582], 1e-3)
close('p.37: 2 * 7.6209 + 104.7582 = 120', 2*300*L['tB']**2/2 + L['v']*L['coast'], 120.0, 1e-9)
out['lspb_example'] = dict(q0=0, qf=120, T=2, a=300, amin=L['amin'], tB=L['tB'], v=L['v'], coast=L['coast'])

# equality and below the minimum
Le = lspb(0, 120, 2, 120)
close('a = a_min = 120: tB = tF/2 = 1 s, v = 120 deg/s, no coast phase', [Le['tB'], Le['v'], Le['coast']], [1, 120, 0])
check('a = 100 < 120: no real root', lspb(0, 120, 2, 100)['ok'] is False)
check('a = 100: discriminant = 100^2*4 - 4*100*120 = -8000 < 0', 100**2*4 - 4*100*120 == -8000)
# P5.6 variants
check('P5.6: 90 deg in 1.5 s, min a = 4*90/2.25 = 160 deg/s^2', np.isclose(4*90/1.5**2, 160))
Lq = lspb(0, 90, 1.5, 200)
print(f"   90 deg in 1.5 s with a = 200: tB = {Lq['tB']:.6f}, v = {Lq['v']:.4f}")
Lq2 = lspb(0, 90, 1.5, 240)
print(f"   90 deg in 1.5 s with a = 240: tB = {Lq2['tB']:.6f}, v = {Lq2['v']:.4f}")
# effect of larger a: shorter blend, higher coast velocity, never above the limit 2 dq/tF
checks_mono = [lspb(0, 120, 2, a)['v'] for a in (120, 150, 300, 1000, 1e6)]
check('larger a: coast velocity falls from 2 dq/tF = 120 toward dq/tF = 60 (never below)', all(np.diff(checks_mono) < 0) and checks_mono[-1] > 60 and np.isclose(checks_mono[0], 120))

# t0 != 0: the slide's general-t0 equation is not right (valid only for t0 = 0)
t0v = 1.0; tFv = 3.0   # same motion as p.37 but shifted: starts at 1 s, ends at 3 s
Lsh = lspb(0, 120, tFv, 300, t0=t0v)
close('t0 != 0: shifting the clock leaves tB and v unchanged (tB = 0.2254, v = 67.62)', [Lsh['tB'], Lsh['v']], [L['tB'], L['v']], 1e-12)
# slide equation: a tB^2 - a (tF - t0) tB + (qF - q0 - a t0^2/2) = 0   with tF = 3, t0 = 1
cA_ = 300; cB_ = -300*(tFv - t0v); cC_ = 120 - 300*t0v**2/2
disc_slide = cB_**2 - 4*cA_*cC_
print(f'   slide general-t0 quadratic with t0 = 1, tF = 3: discriminant = {disc_slide}')
roots_slide = np.roots([cA_, cB_, cC_]) if disc_slide >= 0 else None
print('   slide roots:', roots_slide, ' correct tB:', L['tB'])
check('t0 != 0: the slide quadratic does NOT give the right blend time (so use tF - t0 as the duration)', disc_slide < 0 or not np.isclose(np.min(np.real(roots_slide)), L['tB']))
check('correct general form: a tB^2 - a (tF - t0) tB + (qF - q0) = 0 reproduces tB for the shifted motion', np.isclose(np.min(np.roots([300, -300*(tFv - t0v), 120])), L['tB']))
# also: if tB is read as an absolute time, the slide's q_B2 uses (tF - 2tB) instead of (tF - 2tB + t0)
check('slide q_B2 term v(tF - 2tB) matches the coast length only when t0 = 0', np.isclose(2 - 2*L['tB'], L['coast']) and not np.isclose((tFv - 2*L['tB']), (tFv - t0v) - 2*L['tB']))
out['lspb_shift'] = dict(t0=t0v, tF=tFv, tB=Lsh['tB'], disc_slide=float(disc_slide))

out['lspb_min'] = dict(q0=0, qf=120, T=2, a=120, tB=1.0, v=120.0)
out['lspb_below'] = dict(q0=0, qf=120, T=2, a=100, amin=120.0, ok=False)

# ------------------------------------------------------------------ 6. multi-segment LSPB with via points (Craig sec 7.3; slides give only symbols)
def craig_blend(th, td, acc):
    """th: n points, td: n-1 segment durations (via point to via point), acc: n blend acceleration magnitudes.
    Returns blend times t[0..n-1], linear velocities v[0..n-2], linear durations tl[0..n-2], accelerations a (signed)."""
    n = len(th); tb = np.zeros(n); v = np.zeros(n - 1); a = np.zeros(n)
    a[0] = np.sign(th[1] - th[0])*acc[0]
    tb[0] = td[0] - np.sqrt(td[0]**2 - 2*(th[1] - th[0])/a[0])
    v[0] = (th[1] - th[0])/(td[0] - 0.5*tb[0])
    a[-1] = np.sign(th[-2] - th[-1])*acc[-1]
    tb[-1] = td[-1] - np.sqrt(td[-1]**2 + 2*(th[-1] - th[-2])/a[-1])
    v[-1] = (th[-1] - th[-2])/(td[-1] - 0.5*tb[-1])
    for j in range(1, n - 2):
        v[j] = (th[j+1] - th[j])/td[j]
    for j in range(1, n - 1):
        a[j] = np.sign(v[j] - v[j-1])*acc[j]
        tb[j] = (v[j] - v[j-1])/a[j]
    tl = np.array([td[j] - 0.5*tb[j] - 0.5*tb[j+1] for j in range(n - 1)])
    tl[0] = td[0] - tb[0] - 0.5*tb[1]
    tl[-1] = td[-1] - 0.5*tb[-2] - tb[-1]
    return tb, v, tl, a

def craig_profile(th, td, tb, v, tl, a, n=200001):
    # piecewise acceleration; times of the via points are cumulative sums of td
    n_ = len(th); acc_t = []  # list of (t_start, t_end, acc)
    tvia = np.concatenate([[0], np.cumsum(td)])
    tcur = 0.0
    events = []
    events.append((0.0, tb[0], a[0]))
    for j in range(1, n_ - 1):
        events.append((tvia[j] - tb[j]/2, tvia[j] + tb[j]/2, a[j]))
    events.append((tvia[-1] - tb[-1], tvia[-1], a[-1]))
    tt = np.linspace(0, tvia[-1], n)
    acc = np.zeros_like(tt)
    for s_, e_, ac in events: acc[(tt >= s_) & (tt < e_)] = ac
    vel = np.concatenate([[0], np.cumsum((acc[1:] + acc[:-1])/2*np.diff(tt))])
    pos = th[0] + np.concatenate([[0], np.cumsum((vel[1:] + vel[:-1])/2*np.diff(tt))])
    return tt, pos, vel, acc, tvia

th = np.array([0., 40., 10.]); td = np.array([2., 2.]); accm = np.array([50., 50., 50.])
tb, vlin, tl, acs = craig_blend(th, td, accm)
print('   multi-segment: blend times', np.round(tb, 5).tolist(), 'linear velocities', np.round(vlin, 4).tolist(), 'linear durations', np.round(tl, 5).tolist(), 'signed a', acs.tolist())
tt, pos, vel, acc, tvia = craig_profile(th, td, tb, vlin, tl, acs)
check('multi-segment: all linear durations positive (feasible)', np.all(tl > 0))
check('multi-segment: ends at theta_F = 10 deg with zero velocity', np.isclose(pos[-1], 10, atol=2e-2) and np.isclose(vel[-1], 0, atol=1e-2))
check('multi-segment: velocity on segment 1 equals v12 after blend 1 (22.54 deg/s)', np.isclose(vel[np.searchsorted(tt, tb[0] + 0.01)], vlin[0], atol=0.02))
i_via = np.searchsorted(tt, tvia[1])
dev = th[1] - pos[i_via]
close('multi-segment: path passes NEAR the via point: miss distance a2 t2^2/8 = |dv| t2/8', abs(dev), abs(acs[1])*tb[1]**2/8, 5e-3)
print(f'   miss distance at the via point = {abs(dev):.4f} deg (theta_2 = 40, actual {pos[i_via]:.4f})')
check('multi-segment: the path does not pass exactly through the via point', abs(dev) > 1)
out['multi_lspb'] = dict(th=th.tolist(), td=td.tolist(), acc=accm.tolist(), tb=tb.tolist(), v=vlin.tolist(), tl=tl.tolist(), miss=float(abs(dev)))

# ------------------------------------------------------------------ 7. planar RR straight line (pp. 49-64)
L1_, L2_ = 4.0, 3.0
start = np.array([0, 4.75]); end = np.array([4.75, 0])
slope, icpt = np.linalg.solve(np.array([[start[0], 1], [end[0], 1]]), np.array([start[1], end[1]]))
close('p.51: y = -x + 4.75', [slope, icpt], [-1, 4.75])
plen = np.linalg.norm(end - start); ptime = plen/3
close('p.51: length 6.7175 m, time 2.2392 s (slide: 2.24 s)', [plen, ptime], [6.7175, 2.2392], 1e-4)
check('start (0, 4.75): L1 + 0.25 L2 = 4.75', np.isclose(L1_ + 0.25*L2_, 4.75))
def ik_rr(x, y, branch=+1):
    c2 = (x*x + y*y - L1_**2 - L2_**2)/(2*L1_*L2_)
    s2 = branch*np.sqrt(1 - c2*c2)
    q2 = np.arctan2(s2, c2)
    q1 = np.arctan2(y, x) - np.arctan2(L2_*s2, L1_ + L2_*c2)
    return q1, q2
def fk_rr(q1, q2):
    return L1_*np.cos(q1) + L2_*np.cos(q1 + q2), L1_*np.sin(q1) + L2_*np.sin(q1 + q2)
ts = np.linspace(0, ptime, 11)
xs = 3*ts/np.sqrt(1 + slope**2); ys = icpt - 3*ts/np.sqrt(2)
check('p.53: sampling x(t) = 3t/sqrt2, y = 4.75 - 3t/sqrt2 ends at (4.75, 0)', np.isclose(xs[-1], 4.75) and np.isclose(ys[-1], 0, atol=1e-12))
check('p.53: nominal sample times t_i = 0.224 i (rounded from 0.22392 i)', np.allclose(ts, 0.22392*np.arange(11), atol=1e-4) and abs(ts[1] - 0.224) < 1e-3)
q1s, q2s = ik_rr(xs, ys, +1)
q1s_o, q2s_o = ik_rr(xs, ys, -1)
x_chk, y_chk = fk_rr(q1s, q2s); x_chk2, y_chk2 = fk_rr(q1s_o, q2s_o)
check('IK then FK returns the sampled points (both branches)', np.allclose(x_chk, xs) and np.allclose(y_chk, ys) and np.allclose(x_chk2, xs) and np.allclose(y_chk2, ys))
print(f'   RR start (rad): q1 = {q1s[0]:.4f}, q2 = {q2s[0]:.4f}; end q1 = {q1s[-1]:.4f}, q2 = {q2s[-1]:.4f}')
print(f'   RR start (deg): q1 = {q1s[0]/deg:.3f}, q2 = {q2s[0]/deg:.3f}; end q1 = {q1s[-1]/deg:.3f}, q2 = {q2s[-1]/deg:.3f}')
close('RR start q = (0.8914, 1.6725) rad; end q = (-0.6794, 1.6725) rad', [q1s[0], q2s[0], q1s[-1], q2s[-1]], [0.8914, 1.6725, -0.6794, 1.6725], 2e-4)
check('q2 is symmetric about the midpoint (q2(0) = q2(tF)) so its cubic term is ~0', np.isclose(q2s[0], q2s[-1]))
pf1 = np.polyfit(ts, q1s, 3); pf2 = np.polyfit(ts, q2s, 3)
print('   polyfit q1:', np.round(pf1, 4).tolist(), ' q2:', np.round(pf2, 4).tolist())
close('p.59 q1 cubic (0.1347, -0.3376, -0.6181, 0.8858) vs recomputed (0.1346, -0.3375, -0.6182, 0.8858): agree to 1e-3', pf1, [0.1347, -0.3376, -0.6181, 0.8858], 1.1e-3)
close('recomputed q2 cubic (0, -0.4032, 0.9029, 1.6674)', pf2, [0.0, -0.4032, 0.9029, 1.6674], 2e-4)
check('p.59 q2 slide (0.0004, -0.4043, 0.9036, 1.6674) differs from the recomputation by up to 0.0011 in a2 and 0.0007 in a1', abs(pf2[1] + 0.4043) > 5e-4 and abs(pf2[1] + 0.4043) < 1.5e-3 and abs(pf2[2] - 0.9036) < 1e-3)
# can any sampling reproduce the slide q2? (search tF and sample count)
best = None
for TT in np.linspace(2.2, 2.3, 101):
    for N in (11,):
        tq = np.linspace(0, TT, N); xq = 3*tq/np.sqrt(2); yq = 4.75 - xq
        q1q, q2q = ik_rr(xq, yq); f2 = np.polyfit(tq, q2q, 3)
        err = np.abs(f2 - [0.0004, -0.4043, 0.9036, 1.6674]).max()
        if best is None or err < best[0]: best = (err, TT, f2)
print(f'   closest sampling to slide q2: tF = {best[1]:.4f}, err = {best[0]:.5f}, coeffs {np.round(best[2], 4).tolist()}')
fit_err = max(np.abs(np.polyval(pf1, ts) - q1s).max(), np.abs(np.polyval(pf2, ts) - q2s).max())
print(f'   max cubic-fit error at the samples = {fit_err:.5f} rad')
check('one cubic fits the 11 samples to within 0.01 rad (0.46 deg): good but not exact (slide p.60 says a single cubic may not suffice)', 5e-3 < fit_err < 1e-2)
pf5a = np.polyfit(ts, q1s, 5); pf5b = np.polyfit(ts, q2s, 5)
fit5 = max(np.abs(np.polyval(pf5a, ts) - q1s).max(), np.abs(np.polyval(pf5b, ts) - q2s).max())
print(f'   quintic fit error at the samples = {fit5:.6f} rad')
check('a quintic fits the same 11 samples more than 5x better (p.60)', fit5 < fit_err/5)
# resample to 50 points and FK: how straight is the end-effector path?
tr = np.linspace(0, ptime, 50)
q1r = np.polyval(pf1, tr); q2r = np.polyval(pf2, tr)
xe, ye = fk_rr(q1r, q2r)
dist_line = np.abs(xe + ye - 4.75)/np.sqrt(2)
print(f'   fitted path max distance from the straight line = {dist_line.max()*1000:.3f} mm')
check('fitted cubic path wanders up to ~37 mm from the straight line (6.7 m long)', 0.03 < dist_line.max() < 0.04)
out['rr'] = dict(L=[4, 3], start=[0, 4.75], end=[4.75, 0], speed=3, length=plen, time=ptime, q_start_rad=[q1s[0], q2s[0]], q_end_rad=[q1s[-1], q2s[-1]],
                 polyfit_q1=pf1.tolist(), polyfit_q2=pf2.tolist(), fit_error_rad=fit_err, line_dev_mm=float(dist_line.max()*1000))

# workspace and singularity checks for the RR line
rmin = np.min(np.hypot(xs, ys)); rmax = np.max(np.hypot(xs, ys))
check('RR line stays inside the annulus |L1-L2| = 1 < r < L1+L2 = 7 (no workspace/singularity problem)', rmin > 1 and rmax < 7)
close('closest approach of the line to the base: 4.75/sqrt2 = 3.359 m; max r = 4.75 m', [rmin, rmax], [4.75/np.sqrt(2), 4.75], 0.02)

# Q4: other IK branch
close('Q4: other branch at the start: q2 = -1.6725, q1 = ? ', [q2s_o[0]], [-1.6725], 2e-4)
print(f'   other branch start: q1 = {q1s_o[0]:.4f} rad ({q1s_o[0]/deg:.2f} deg), q2 = {q2s_o[0]:.4f}; end: q1 = {q1s_o[-1]:.4f}, q2 = {q2s_o[-1]:.4f}')
check('Q4: the other branch has q2 -> -q2 at every sample', np.allclose(q2s_o, -q2s))
check('Q4: other branch q1 = atan2(y,x) + mu (mirror about the line to the target)', np.allclose(q1s_o + q1s, 2*np.arctan2(ys, xs)))
close('Q4: the end-effector path is identical for both branches', np.c_[x_chk2, y_chk2], np.c_[x_chk, y_chk], 1e-9, show=False)
mixed = np.r_[q1s[:5], q1s_o[5:]]
check('Q4: switching branch mid-path makes q1 jump (a discontinuity, impossible to follow)', abs(q1s[4] - q1s_o[5]) > 0.3)
print(f'   switching at sample 5 would jump q1 from {q1s[4]:.4f} to {q1s_o[5]:.4f} rad and q2 from {q2s[4]:.4f} to {q2s_o[5]:.4f}')
out['rr_other_branch'] = dict(q_start=[q1s_o[0], q2s_o[0]], q_end=[q1s_o[-1], q2s_o[-1]])

# joint-space linear interpolation vs Cartesian straight line (p.46) for the same RR
qa = np.array([q1s[0], q2s[0]]); qb = np.array([q1s[-1], q2s[-1]])
s_ = np.linspace(0, 1, 1001)
qj = qa[None, :] + s_[:, None]*(qb - qa)[None, :]
xj, yj = fk_rr(qj[:, 0], qj[:, 1])
dj = np.abs(xj + yj - 4.75)/np.sqrt(2)
qm = (qa + qb)/2; xm, ym = fk_rr(*qm)
print(f'   joint-linear midpoint: q = ({qm[0]:.4f}, {qm[1]:.4f}) rad -> tool ({xm:.4f}, {ym:.4f}) m; straight-line midpoint (2.375, 2.375)')
dmid = abs(xm + ym - 4.75)/np.sqrt(2)
print(f'   deviation of the joint-linear midpoint from the line = {dmid:.4f} m; max over path = {dj.max():.4f} m')
close('joint-linear midpoint tool position is (3.3588, 3.3588), not the straight-line midpoint (2.375, 2.375)', [xm, ym], [3.3588, 3.3588], 1e-3)
check('joint-linear path is an arc of radius 4.75 m about the base (q2 constant)', np.allclose(np.hypot(xj, yj), 4.75, atol=1e-6))
check('joint-linear path bows away from the straight line by > 0.5 m at the midpoint', dmid > 0.5)
out['joint_linear'] = dict(mid_q=qm.tolist(), mid_xy=[xm, ym], dev_mid=dmid, dev_max=float(dj.max()))

# singularity: joint rates blow up near a stretched arm. Use L1 = L2 = 1, tool moves in +x at 1 m/s through (x, 0)
def jac(q1, q2, l1=1.0, l2=1.0):
    return np.array([[-l1*np.sin(q1) - l2*np.sin(q1 + q2), -l2*np.sin(q1 + q2)], [l1*np.cos(q1) + l2*np.cos(q1 + q2), l2*np.cos(q1 + q2)]])
def rates_at(x, y=0.0, vx=1.0, vy=0.0):
    c2 = (x*x + y*y - 2)/2; q2 = np.arccos(c2); q1 = np.arctan2(y, x) - np.arctan2(np.sin(q2), 1 + np.cos(q2))
    J = jac(q1, q2); qd = np.linalg.solve(J, [vx, vy])
    return q2, np.linalg.det(J), qd
rows = []
for x in (1.0, 1.9, 1.99, 1.999):
    q2, dt, qd = rates_at(x); rows.append((x, q2/deg, dt, np.abs(qd).max())); print(f'   x = {x}: q2 = {q2/deg:.3f} deg, det J = {dt:.4f}, max |qdot| = {np.abs(qd).max():.3f} rad/s')
check('singularity: det J = L1 L2 sin q2 (= sin q2 for unit links)', all(np.isclose(abs(r_[2]), np.sin(np.radians(r_[1]))) for r_ in rows))
check('singularity: max joint rate grows as the tool nears full stretch (x -> 2)', rows[0][3] < rows[1][3] < rows[2][3] < rows[3][3] and rows[3][3] > 20)
out['singularity'] = [dict(x=r_[0], q2_deg=r_[1], detJ=r_[2], qdot_max=r_[3]) for r_ in rows]

# ------------------------------------------------------------------ 8. misc numbers used in text
check('mean/peak speed of a rest-to-rest cubic = 2/3 = 66.7 %, trapezoid with a coast reaches 100 % of its own peak in the cruise', np.isclose(1/1.5, 2/3))
close('P5.5: P5.1 peak joint acceleration |q_dd|max = 6 d/T^2 = 40 deg/s^2', 6*60/9, 40.0)
close('rest-to-rest cubic peak acceleration is 6 d / T^2 (Example 1: 6*40/4 = 60)', 6*40/4, 60.0)


# ------------------------------------------------------------------ 8b. extra numbers used in the guided problems
# P5.2 three joints, same tF = 2 s, rest to rest
dq3 = np.array([40., 90., -20.]); Tj = 2.0
close('P5.2 a2 = 3 d/T^2 for the three joints: 30, 67.5, -15', 3*dq3/Tj**2, [30, 67.5, -15])
close('P5.2 peak speeds 1.5 d/T: 30, 67.5, -15 deg/s', 1.5*dq3/Tj, [30, 67.5, -15])
# notebook cell 29: mean/peak of the quintic with qd0 = 10, qdf = 5 (0 -> 15 in 1 s, 50 samples) -- computed here; the notebook saved no output
cqn = quintic_coeffs(0, 15, 10, 5, 1); tq50 = np.linspace(0, 1, 50); vq50 = poly(cqn, tq50, 1)
pct = vq50.mean()/vq50.max()*100
print(f'   notebook cell 29 (quintic, qd0=10, qdf=5): mean/peak = {pct:.3f} %   (peak {vq50.max():.4f} deg/s)')
cqz = quintic_coeffs(0, 15, 0, 0, 1); vz = poly(cqz, tq50, 1)
print(f'   same for the default zero end velocities: mean/peak = {vz.mean()/vz.max()*100:.3f} %')
out['cell29'] = dict(pct=pct, peak=float(vq50.max()))
# toolbox: trapezoidal(0, 120, linspace(0, 2, 50), V=70)  ->  coast velocity V given, a follows
Vv = 70.0; D_ = 120.0; T_ = 2.0
tBv = T_ - D_/Vv; aV = Vv/tBv
close('cell 39 (V = 70): tB = T - D/V = 0.2857 s, a = V/tB = 245 deg/s^2', [tBv, aV], [0.2857142857, 245.0], 1e-9)
check('cell 39: that a satisfies a >= a_min = 120 and reproduces D', aV >= 120 and np.isclose(Vv*(T_ - tBv), D_))
check('valid coast velocities: D/T = 60 < V <= 2D/T = 120', 60 < Vv <= 120)
check('a = v^2/(vT - D) form gives 245', np.isclose(Vv**2/(Vv*T_ - D_), 245.0))
# P5.7: shifted LSPB (t0 = 1, tF = 3) -> same tB
# slide problem 1: line y = 0.2 for the 1, 0.7 arm
check('problem 1: line y = 0.2 passes 0.2 m from the base, below |L1 - L2| = 0.3 (unreachable point (0, 0.2)); ends (+-1, 0.2) are reachable (r = 1.0198 in [0.3, 1.7])', 0.2 < 0.3 and 0.3 < np.hypot(1, 0.2) < 1.7)


# P5.8: quintic for Example 1 (10 -> 50 deg in 2 s)
cq1 = quintic_coeffs(10, 50, 0, 0, 2); tq = np.linspace(0, 2, 200001)
close('P5.8 quintic Example 1: peak speed 1.875*40/2 = 37.5 deg/s, peak accel 10/sqrt3*40/4 = 57.735 deg/s^2', [poly(cq1, tq, 1).max(), np.abs(poly(cq1, tq, 2)).max()], [37.5, 57.7350269], 1e-4)

# P5.6 / P5.8 / P5.10 numbers
close('P5.6c: coast velocity at a = 150 is 82.918, at 300 is 67.62 (120 at a = 120)', [lspb(0, 120, 2, 150)['v'], lspb(0, 120, 2, 300)['v'], lspb(0, 120, 2, 120)['v']], [82.917961, 67.620999, 120.0], 1e-6)
L8 = lspb(10, 50, 2, 100)
close('P5.8 LSPB 10 -> 50 in 2 s with a = 100: amin = 40, tB = 0.2254, v = 22.54', [L8['amin'], L8['tB'], L8['v']], [40.0, 0.225403, 22.5403], 1e-4)
qm_deg = np.degrees(qm)
xm_, ym_ = L1_*np.cos(qm[0]), L1_*np.sin(qm[0])
x2_, y2_ = L2_*np.cos(qm[0] + qm[1]), L2_*np.sin(qm[0] + qm[1])
print(f'   P5.10 halfway: q = {qm_deg} deg; 4cos q1 = {xm_:.4f}, 3cos(q1+q2) = {x2_:.4f}, 4 sin q1 = {ym_:.4f}, 3 sin(q1+q2) = {y2_:.4f}')
close('P5.10 terms 3.9776 - 0.6188 = 3.3588 and 0.4233 + 2.9355 = 3.3588', [xm_, x2_, ym_, y2_], [3.9776, -0.6188, 0.4233, 2.9355], 2e-4)
close('P5.10 q1 halfway 6.07 deg, q2 95.83 deg; q1+q2 = 101.9 deg', [qm_deg[0], qm_deg[1], qm_deg.sum()], [6.074, 95.829, 101.903], 2e-3)
close('P5.9 other branch: mu = -38.93 deg at start; q1 = 128.93, q2 = -95.83', [np.degrees(q1s_o[0]), np.degrees(q2s_o[0])], [128.93, -95.83], 5e-3)
close('P5.9 end of other branch q1 = 38.93 deg', np.degrees(q1s_o[-1]), 38.93, 5e-3)
close('P5.9 cos q2 at start = (4.75^2 - 25)/24 = -0.1016', (4.75**2 - 25)/24, -0.10156, 1e-4)

# ------------------------------------------------------------------ 9. widget test vectors
VEC = []
def add(name, **kw): VEC.append(dict(name=name, **kw))
add('cubic_rest', P=dict(mode='cubic', q0=10, qf=50, T1=2, v0=0, vf=0), coeffs=[c.tolist()], vtotal=2)
add('cubic_general', P=dict(mode='cubic', q0=20, qf=80, T1=3, v0=0, vf=10), coeffs=[c_g.tolist()], vtotal=3)
add('quintic_rest', P=dict(mode='quintic', q0=0, qf=15, T1=1, v0=0, vf=0), coeffs=[cq.tolist()], vtotal=1)
add('quintic_vel', P=dict(mode='quintic', q0=0, qf=15, T1=1, v0=10, vf=5), coeffs=[cqg.tolist()], vtotal=1)
# via, heuristic: opposite slopes -> zero
cv1 = [cubic_coeffs(10, 50, 0, 0, 5).tolist(), cubic_coeffs(50, 20, 0, 0, 5).tolist()]
add('via_heur_zero', P=dict(mode='via', vmode='heur', q0=10, qv=50, qf=20, T1=5, T2=5), vv=0.0, coeffs=cv1, vtotal=10)
# via, heuristic: same sign -> average (slopes 10 and 10 -> 10; use unequal: 10 and 20/3)
s1 = (30 - 10)/2; s2 = (60 - 30)/3   # 10, 10
s1b = (30 - 10)/2; s2b = (50 - 30)/3  # 10, 6.667
vv_b = 0.5*(s1b + s2b)
cv2 = [cubic_coeffs(10, 30, 0, vv_b, 2).tolist(), cubic_coeffs(30, 50, vv_b, 0, 3).tolist()]
add('via_heur_avg', P=dict(mode='via', vmode='heur', q0=10, qv=30, qf=50, T1=2, T2=3), vv=vv_b, coeffs=cv2, vtotal=5)
close('heuristic average: slopes 10 and 6.667 -> 8.3333 deg/s', vv_b, 8.333333333, 1e-8)
cv3 = [cubic_coeffs(10, 40, 0, 7.5, 1).tolist(), cubic_coeffs(40, 20, 7.5, 0, 1).tolist()]
add('via_cont_equal', P=dict(mode='via', vmode='cont', q0=10, qv=40, qf=20, T1=1, T2=1), vv=7.5, coeffs=cv3, vtotal=2)
add('via_cont_unequal', P=dict(mode='via', vmode='cont', q0=10, qv=40, qf=20, T1=1, T2=2), vv=25.0, coeffs=[cA.tolist(), cB.tolist()], vtotal=3)
cv4 = [cubic_coeffs(10, 40, 0, 12, 1).tolist(), cubic_coeffs(40, 20, 12, 0, 1).tolist()]
add('via_user', P=dict(mode='via', vmode='user', q0=10, qv=40, qf=20, T1=1, T2=1, vv=12), vv=12.0, coeffs=cv4, vtotal=2)
add('lspb_example', P=dict(mode='lspb', q0=0, qf=120, T1=2, a=300), tB=L['tB'], v=L['v'], amin=120.0, ok=True, vtotal=2)
add('lspb_min', P=dict(mode='lspb', q0=0, qf=120, T1=2, a=120), tB=1.0, v=120.0, amin=120.0, ok=True, vtotal=2)
add('lspb_below', P=dict(mode='lspb', q0=0, qf=120, T1=2, a=100), amin=120.0, ok=False, vtotal=2)
Ld = lspb(100, 40, 3, 100)
add('lspb_down', P=dict(mode='lspb', q0=100, qf=40, T1=3, a=100), tB=Ld['tB'], v=Ld['v'], amin=4*60/9, ok=True, vtotal=3)
print(f"   LSPB downward move: 100 -> 40 in 3 s, a = 100: amin = {4*60/9:.4f}, tB = {Ld['tB']:.6f}, v = {Ld['v']:.4f}")
# sample points of the profile for the plot-level check (position at several times)
def eval_vec(v):
    P = v['P']
    if 'coeffs' in v:
        T_list = [P['T1']] if len(v['coeffs']) == 1 else [P['T1'], P['T2']]
        res = []
        for tq in (0.0, 0.31*v['vtotal'], 0.77*v['vtotal'], v['vtotal']):
            acc_t = 0.0
            for ci, Tk in zip(v['coeffs'], T_list):
                if tq <= acc_t + Tk + 1e-12 or ci is v['coeffs'][-1]:
                    tau = min(tq - acc_t, Tk); res.append([tq, float(poly(ci, tau)), float(poly(ci, tau, 1)), float(poly(ci, tau, 2))]); break
                acc_t += Tk
        v['samples'] = res
    elif v.get('ok'):
        tt_, pos_, vel_, acc_ = lspb_profile(P['q0'], P['qf'], P['T1'], P['a'], n=400001)
        res = []
        for tq in (0.0, 0.31*P['T1'], 0.77*P['T1'], P['T1']):
            i = int(round(tq/P['T1']*(len(tt_) - 1))); res.append([tq, float(pos_[i]), float(vel_[i]), float(acc_[i])])
        v['samples'] = res
for v in VEC: eval_vec(v)
for v in VEC:
    if 'samples' in v and 'coeffs' in v:
        pass
json.dump(dict(vectors=VEC, extra=out), open(os.path.join(HERE, 'w05_vectors.json'), 'w'), indent=1)
print(f'   wrote w05_vectors.json with {len(VEC)} vectors')

# ------------------------------------------------------------------ 10. run the widget JS core in node against the vectors
frag = os.path.join(HERE, '..', 'frag', '05-w05.html')
if os.path.exists(frag) and shutil.which('node'):
    s = open(frag, encoding='utf-8').read()
    m = re.search(r'/\*BEGIN-TRAJ-CORE\*/(.*?)/\*END-TRAJ-CORE\*/', s, re.S)
    check('widget core markers present in the fragment', bool(m))
    if m:
        js = m.group(1) + '''
const V = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8')).vectors; const bad = [];
const close = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
for (const v of V) {
  const R = trajCore(v.P);
  if (v.coeffs) { v.coeffs.forEach((c, i) => c.forEach((x, k) => { if (!close(R.segs[i].c[k], x)) bad.push(v.name + ' coeff ' + i + ',' + k + ': ' + R.segs[i].c[k] + ' vs ' + x); })); if (R.segs.length !== v.coeffs.length) bad.push(v.name + ' seg count'); }
  if (v.vv !== undefined && !close(R.vv, v.vv)) bad.push(v.name + ' vv ' + R.vv + ' vs ' + v.vv);
  if (v.ok !== undefined && R.ok !== v.ok) bad.push(v.name + ' ok');
  if (v.amin !== undefined && !close(R.amin, v.amin)) bad.push(v.name + ' amin ' + R.amin);
  if (v.tB !== undefined && !close(R.tB, v.tB)) bad.push(v.name + ' tB ' + R.tB + ' vs ' + v.tB);
  if (v.v !== undefined && !close(R.v, v.v)) bad.push(v.name + ' v ' + R.v + ' vs ' + v.v);
  if (v.samples) v.samples.forEach(sm => { const e = R.ev(sm[0]); [1, 2, 3].forEach(k => { if (!close(e[k - 1], sm[k], 1e-3)) bad.push(v.name + ' sample t=' + sm[0] + ' k=' + k + ': ' + e[k - 1] + ' vs ' + sm[k]); }); });
}
console.log(JSON.stringify({ n: V.length, bad }));
'''
        tmp = os.path.join(HERE, '_w05_node_test.js')
        open(tmp, 'w').write(js)
        try:
            r_ = subprocess.run(['node', tmp, os.path.join(HERE, 'w05_vectors.json')], capture_output=True, text=True, timeout=60)
            res = json.loads(r_.stdout.strip().splitlines()[-1]) if r_.stdout.strip() else {'bad': ['node failed: ' + r_.stderr[:300]]}
        finally:
            os.remove(tmp)
        for b in res['bad'][:10]: print('   JS mismatch:', b)
        check(f'widget JS core reproduces all {len(VEC)} test vectors (coefficients, via velocity, blend time, samples)', not res['bad'])

report()
