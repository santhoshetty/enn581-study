"""v2 booklet W8 (Control 1): every NEW number in booklet/w08.tex. Run: python3 v2_w08.py"""
import numpy as np, sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
d = np.degrees; r_ = np.radians
fails = []
def chk(name, cond, extra=''):
    print(("PASS " if cond else "FAIL ") + name + ((' | ' + str(extra)) if extra != '' else ''))
    if not cond: fails.append(name)
def close(name, got, want, tol=1e-6):
    ok = np.allclose(got, want, atol=tol, rtol=0)
    chk(name, ok, f'got {np.round(got,6)} want {want}')

def sim(m, b, k, x0, v0, T, n=20001, f=lambda t, x, v: 0.0):
    t = np.linspace(0, T, n)
    s = solve_ivp(lambda t, y: [y[1], (f(t, y[0], y[1]) - b*y[1] - k*y[0])/m], [0, T], [x0, v0], t_eval=t, rtol=1e-12, atol=1e-13)
    return t, s.y[0]
def settle(t, x, tol=0.02, ref=1.0):
    idx = np.where(np.abs(x) > tol*ref)[0]
    return t[idx[-1]] if len(idx) else 0.0

print("=== A. omega_n, zeta forms of the roots (symbolic)")
m, b, k, wn, z = sp.symbols('m b k w_n zeta', positive=True)
s1 = (-b + sp.sqrt(b**2 - 4*m*k))/(2*m)
s1z = (-z*wn + wn*sp.sqrt(z**2 - 1)).subs({wn: sp.sqrt(k/m), z: b/(2*sp.sqrt(m*k))})
chk("s1 = -zeta wn + wn sqrt(zeta^2-1) equals the quadratic formula", sp.simplify(s1 - s1z) == 0)
chk("m s^2+b s+k = m (s^2 + 2 zeta wn s + wn^2)", sp.simplify(m*(sp.Symbol('s')**2 + 2*(b/(2*sp.sqrt(m*k)))*sp.sqrt(k/m)*sp.Symbol('s') + k/m) - (m*sp.Symbol('s')**2 + b*sp.Symbol('s') + k)) == 0)
# slide examples zeta and wn
for (mm, bb, kk, nm) in [(1, 3, 2, 'p19'), (1, 1, 1, 'p22'), (1, 4, 4, 'p25')]:
    zz = bb/(2*np.sqrt(mm*kk)); ww = np.sqrt(kk/mm)
    print(f"   {nm}: zeta={zz:.4f}, wn={ww:.4f}")
close("p19 zeta = 3/(2 sqrt2) = 1.0607", 3/(2*np.sqrt(2)), 1.0607, 5e-5)
close("p19 wn = sqrt2 = 1.4142", np.sqrt(2), 1.4142, 5e-5)

print("=== B. deck examples (re-derived) x(0)=-1, xd(0)=0")
close("p19 roots", np.sort(np.roots([1, 3, 2]).real), [-2, -1])
close("p19 A1,A2", np.linalg.solve([[1, 1], [-1, -2]], [-1, 0]), [-2, 1])
close("p19 x(1)", -2*np.exp(-1) + np.exp(-2), -0.6004, 5e-5)
close("p22 A2 = -1/sqrt3", -1/np.sqrt(3), -0.5774, 5e-5)
t, x2 = sim(1, 1, 1, -1, 0, 14)
chk("p22 first peak +0.1630 at 3.628 s", abs(x2.max() - 0.16303) < 1e-4 and abs(t[np.argmax(x2)] - 3.628) < 1e-2, (x2.max(), t[np.argmax(x2)]))
close("p25 A1,A2", np.linalg.solve([[1, 0], [-2, 1]], [-1, 0]), [-1, -2])
close("p25 x(1)=-3e^-2", -3*np.exp(-2), -0.4060, 5e-5)
t3, x3 = sim(1, 4, 4, -1, 0, 12); t1, x1 = sim(1, 3, 2, -1, 0, 12)
close("2% settling over,under,crit = 4.60, 8.08, 2.92", [round(settle(t1, x1), 2), round(settle(t, x2), 2), round(settle(t3, x3), 2)], [4.60, 8.08, 2.92], 1e-9)

print("=== C. general initial velocity (Q8.2): m=1,b=2,k=5,x0=1,v0=3")
A1 = 1.0; lam, mu = -1.0, 2.0; A2 = (3 - lam*A1)/mu
close("A2 = (v0 - lam A1)/mu = 2", A2, 2.0)
xq = lambda tt: np.exp(-tt)*(np.cos(2*tt) + 2*np.sin(2*tt))
tq, xs = sim(1, 2, 5, 1, 3, 3)
close("closed form = ODE", xq(tq), xs, 1e-8)
close("x(0.5) = 1.3485", xq(0.5), 1.3485, 5e-5)
close("zeta = 0.4472", 2/(2*np.sqrt(5)), 0.4472, 5e-5)

print("=== D. PD on the plant and scenario A (critical damping at a given stiffness)")
mm, bb, kk = 0.5, 1.0, 18.0; wnd = 12.0
kprime = mm*wnd**2; bprime = 2*mm*wnd
kp_raw = kprime - kk; kv_raw = bprime - bb
close("raw PD: k' = 72, b' = 12", [kprime, bprime], [72, 12]); close("raw PD gains kp=54, kv=11", [kp_raw, kv_raw], [54, 11])
close("raw PD closed loop double root -12", np.roots([mm, bb + kv_raw, kk + kp_raw]).real, [-12, -12], 1e-6)
kp_p, kv_p = wnd**2, 2*wnd
close("partitioned gains kp=144, kv=24", [kp_p, kv_p], [144, 24])
# partitioned law on the same plant: closed loop is s^2+24s+144
close("partitioned closed loop double root -12", np.roots([1, kv_p, kp_p]).real, [-12, -12], 1e-6)
# wrong: use the partitioned gains on the raw plant (no model cancellation)
rw = np.roots([mm, bb + kv_p, kk + kp_p]); print('   raw plant with kp=144,kv=24 roots', rw)
close("wrong shortcut roots -42.349, -7.651 (over-damped)", np.sort(rw.real), [-42.3494, -7.6506], 1e-3)
chk("raw plant is under-damped: b^2-4mk = 1-36 < 0", bb**2 - 4*mm*kk < 0)
# raw plant with kp=kp_raw only and kv = 0
chk("raw PD without kv: b'=1: zeta' = 1/(2*sqrt(0.5*72)) = 0.0833 (very oscillatory)", abs(1/(2*np.sqrt(0.5*72)) - 0.0833) < 1e-4)

print("=== E. steady-state error scenario B")
m_, fd, kp_ = 1.5, 6.0, 64.0
close("E_ss = 6/(64*1.5) = 0.0625 m", fd/(kp_*m_), 0.0625)
kp_need = fd/(m_*0.005); close("5 mm: kp >= 800", kp_need, 800.0); close("kv = 2 sqrt(800) = 56.57", 2*np.sqrt(800), 56.5685, 1e-3)
# simulate the unit mass error dynamics with disturbance
def ess_sim(kp, kv, fdm, T=10):
    t = np.linspace(0, T, 40001)
    s = solve_ivp(lambda t, y: [y[1], fdm - kv*y[1] - kp*y[0]], [0, T], [0, 0], t_eval=t, rtol=1e-11, atol=1e-13)
    return s.y[0][-1]
close("simulation E(10) = 0.0625", ess_sim(64, 16, 6/1.5), 0.0625, 1e-6)
close("m=4, kp=75, fdist=12: 0.04", 12/(75*4), 0.04)
close("Q8.5 kp >= 1500, kv=77.46", [12/(4*0.002), 2*np.sqrt(1500)], [1500, 77.4597], 1e-3)
close("mock Q7: kp=64 -> E_ss = 5/(64*2) = 0.0390625", 5/(64*2), 0.0390625)
# higher kv does not change it
close("E_ss independent of kv (kv=5 vs 50)", [ess_sim(64, 5, 4), ess_sim(64, 50, 4)], [4/64, 4/64], 1e-5) if False else None
# the slower (kv=5) needs long T, check kv=16 vs 40
close("E_ss same for kv=16 and kv=40", [ess_sim(64, 16, 4), ess_sim(64, 40, 4, T=20)], [4/64, 4/64], 1e-6)

print("=== F. PID stability, scenario C")
def roots3(kp, kv, ki): return np.roots([1, kv, kp, ki])
def stable3(kp, kv, ki): return np.all(roots3(kp, kv, ki).real < -1e-9)
for kp, kv, ki in [(64, 16, 900), (64, 16, 1100), (64, 16, 1024), (30, 5, 200), (30, 5, 100), (49, 14, 100), (49, 14, 686), (49, 14, 700), (75, 17.3205, 1500)]:
    rr = roots3(kp, kv, ki); print(f'   kp={kp} kv={kv} ki={ki}: roots {np.round(rr,4)}  kv*kp={kv*kp:.2f} stable={stable3(kp,kv,ki)}')
chk("(64,16,900) stable, (64,16,1100) unstable", stable3(64, 16, 900) and not stable3(64, 16, 1100))
close("(64,16,1024): (s+16)(s^2+64): roots -16, +-8j", np.sort_complex(roots3(64, 16, 1024)), np.sort_complex(np.array([-16, 8j, -8j])), 1e-6)
chk("(30,5,200): kv kp = 150 < 200 unstable", (not stable3(30, 5, 200)) and 5*30 < 200)
chk("(30,5,100): 150 > 100 stable", stable3(30, 5, 100))
close("(49,14,100) roots -10.140, -1.930 +- 2.477j", np.sort(roots3(49, 14, 100).real), [-10.1401, -1.9300, -1.9300], 1e-3)
close("(49,14,100) imag 2.477", np.max(roots3(49, 14, 100).imag), 2.4770, 1e-3)
close("(49,14): ki limit = 686", 14*49, 686)
# Routh symbolic
a2, a1, a0 = sp.symbols('a2 a1 a0', positive=True)
b1 = (a2*a1 - a0)/a2
chk("Routh first column 1, a2, (a2 a1 - a0)/a2, a0 for s^3+a2 s^2+a1 s+a0", sp.simplify(b1 - (a2*a1 - a0)/a2) == 0)
# unit mass PID: E steady-state zero and integral value
def pid_sim(kp, kv, ki, fdm, T=80, n=80001):
    t = np.linspace(0, T, n)
    s = solve_ivp(lambda t, y: [y[1], y[2] , 0], [0, 0], [0,0,0]) if False else None
    # states: E, Edot, I = int E
    s = solve_ivp(lambda t, y: [y[1], fdm - kv*y[1] - kp*y[0] - ki*y[2], y[0]], [0, T], [0, 0, 0], t_eval=t, rtol=1e-11, atol=1e-13)
    return t, s.y
tt, yy = pid_sim(64, 16, 900, 4.0)
chk("PID kp=64,kv=16,ki=900, fdist/m=4: E -> 0", abs(yy[0][-1]) < 1e-6, yy[0][-1])
close("integral term settles at (fdist/m)/ki = 4/900", yy[2][-1], 4/900, 1e-7)

print("=== G. poles -> behaviour (scenario D), x(0) normalised to 1")
def step_resp(kp, kv, T=20, n=40001):
    t = np.linspace(0, T, n)
    s = solve_ivp(lambda t, y: [y[1], -kv*y[1] - kp*y[0]], [0, T], [1, 0], t_eval=t, rtol=1e-12, atol=1e-14)
    return t, s.y[0]
# poles -3+-4j: kp=25 kv=6
tA, xA = step_resp(25, 6)
close("poles -3+-4j: zeta 0.6, wn 5, overshoot 9.478%", [6/(2*5), 5, 100*np.exp(-np.pi*0.6/0.8)], [0.6, 5, 9.478], 1e-3)
print('   -3+-4j: peak (undershoot of x from 1 to 0) at t=%.4f, depth %.4f' % (np.pi/4, np.exp(-3*np.pi/4)))
close("undershoot -0.0948 at t=pi/4=0.7854 (first trough)", [xA.min(), tA[np.argmin(xA)]], [-np.exp(-3*np.pi/4), np.pi/4], 2e-3)
ts = settle(tA, xA); print('   -3+-4j true 2% settling', ts, ' envelope e^{-3t}=0.02 at', -np.log(0.02)/3, '  rule 4/3=', 4/3)
close("true 2% settling 1.19 s (rule 4/(zeta wn) = 1.33 s)", ts, 1.19, 0.01)
# poles -2,-8 : kp=16 kv=10
tB, xB = step_resp(16, 10)
print('   poles -2,-8: true 2% settling', settle(tB, xB), ' slow pole rule 4/2 =', 4/2, '  time const 1/2 =', 0.5)
close("poles -2,-8: x = (4 e^-2t - e^-8t)/3 ; 2% settling ~ 1.8", [settle(tB, xB), ((8*np.exp(-2*0) - 2)/6)], [1.8, 1.0], 0.15) if False else None
close("poles -2,-8 closed form matches ODE", (8*np.exp(-2*tB) - 2*np.exp(-8*tB))/6, xB, 1e-8)
ts_B = settle(tB, xB); chk("2% settling for poles -2,-8 is 2.10 s (slow-pole rule 4/2 = 2.0)", abs(ts_B - 2.10) < 0.01, ts_B)
# critical -5,-5
tC, xC = step_resp(25, 10)
ts_C = settle(tC, xC); print('   crit -5,-5 settling', ts_C)
close("critical -5,-5: 2% settling = 5.83/5 = 1.166", ts_C, 1.166, 5e-3)
close("(1+pt)e^-pt = 0.02 at pt=5.8335", brentq(lambda u: (1+u)*np.exp(-u) - 0.02, 1, 20), 5.8335, 1e-3)
# undamped
close("kv=0, kp=9: poles +-3j, oscillates for ever, period 2pi/3", 2*np.pi/3, 2.0944, 1e-4)
chk("right-half-plane poles grow: s^2 - s + 4 (kv=-1) diverges", np.all(np.roots([1, -1, 4]).real > 0))
# poles vs gains
close("poles -2,-8: kp=16, kv=10", [(-2)*(-8), -(-2-8)], [16, 10])
close("poles -3+-4j: kp = 25, kv = 6", [(-3)**2 + 16, 6], [25, 6])
print('   2% settling rule 4/(zeta wn): crit wn=5 gives', 4/5, '(true 1.166: rule not valid at zeta=1)')

print("=== H. open loop vs closed loop (unit mass), error drift")
close("open loop E(5) = 0.1 + 0.02*5 = 0.2", 0.1 + 0.02*5, 0.2)
# closed loop kp=100,kv=20 from same ICs
tc = 5.0; E0, Ed0 = 0.1, 0.02
# E = (A+Bt)e^{-10t}; A=E0; B = Ed0+10 E0
Ect = (E0 + (Ed0 + 10*E0)*tc)*np.exp(-10*tc); print('   closed loop (kp=100,kv=20) E(5) =', Ect)
close("closed loop E(1) = (0.1+1.02)e^-10 = 5.1e-5", (E0 + (Ed0 + 10*E0)*1.0)*np.exp(-10), 5.0848e-5, 1e-7)

print("=== I. partitioning: one instant (mock Q7) and Q8.3")
E, Ed = 0.15 - 0.10, 0 - (-0.20)
fp = 16*Ed + 64*E; f = 2*fp + 0.5*(-0.2) + 4*0.10
close("mock Q7(d): f' = 6.4, f = 13.1", [fp, f], [6.4, 13.1])
close("plant xdd = (f - b xd - k x)/m = f'", (f - 0.5*(-0.2) - 4*0.1)/2, fp)
close("mock Q7(b): kp = 64, kv = 16", [8**2, 2*8], [64, 16])
fp3 = 0.8 + 14*0.3 + 49*0.2; f3 = 1.5*fp3 + 4*(-0.2) + 10*0.3
close("Q8.3: f' = 14.8, f = 24.4", [fp3, f3], [14.8, 24.4], 1e-9)

print("=== J. gearbox (Q8.4)")
eta, Im, cm, I, c, k4 = 40.0, 3e-5, 2e-4, 0.05, 0.2, 0.05
Ie = I + eta**2*Im; ce = c + eta**2*cm; Q = Ie*8 + ce*3; Qm = Q/eta
close("Ieff = 0.098, ceff = 0.52", [Ie, ce], [0.098, 0.52])
close("Q = 2.344, Qm = 0.0586, ia = 1.172", [Q, Qm, Qm/k4], [2.344, 0.0586, 1.172], 1e-9)
close("direct motor-side balance gives the same Qm", Im*eta*8 + cm*eta*3 + (I*8 + c*3)/eta, Qm, 1e-12)
close("wrong eta instead of eta^2: ia = 0.517", ((I + eta*Im)*8 + (c + eta*cm)*3)/eta/k4, 0.5173, 1e-3)

print("=== K. practical sheet Q1 (single link, q = b t + c t^2)")
bs, cs, ts_ = sp.symbols('b c t')
qq = bs*ts_ + cs*ts_**2
w = sp.Matrix([0, 0, sp.diff(qq, ts_)]); wd = sp.Matrix([0, 0, sp.diff(qq, ts_, 2)])
Pc = sp.Matrix([3, 0, 0]); vdc = wd.cross(Pc) + w.cross(w.cross(Pc))
chk("alpha = [0,0,2c]", sp.simplify(wd - sp.Matrix([0, 0, 2*cs])) == sp.zeros(3, 1))
chk("a_C = [-3(b+2ct)^2, 6c, 0]", sp.simplify(vdc - sp.Matrix([-3*(bs + 2*cs*ts_)**2, 6*cs, 0])) == sp.zeros(3, 1))
Icm = sp.diag(1, 2, 3); F = vdc; N = Icm*wd + w.cross(Icm*w)
n1 = N + Pc.cross(F)
chk("N = [0,0,6c]", sp.simplify(N - sp.Matrix([0, 0, 6*cs])) == sp.zeros(3, 1))
chk("F = m a_C with m=1; Pc x F = [0,0,18c]", sp.simplify(Pc.cross(F) - sp.Matrix([0, 0, 18*cs])) == sp.zeros(3, 1))
chk("n1 = [0, 0, 24c] (joint torque 24c)", sp.simplify(n1 - sp.Matrix([0, 0, 24*cs])) == sp.zeros(3, 1))
vals = {bs: 0, cs: sp.Rational(1, 2), ts_: 2}
print('   b=0,c=0.5,t=2: alpha', list(wd.subs(vals)), ' aC', list(vdc.subs(vals)), ' tau', n1.subs(vals)[2])
chk("numbers: alpha_z=1, aC=(-12,3,0), tau=12", list(wd.subs(vals)) == [0, 0, 1] and list(vdc.subs(vals)) == [-12, 3, 0] and n1.subs(vals)[2] == 12)
# force components at joint: f1 = F (no child) ; reaction x,y
print('   force F at t=2:', list(F.subs(vals)))
# gravity for a vertical-plane joint (g along -y in frame 1 at q=0?) skip
chk("parallel axis: Izz + m*3^2 = 12; 12*2c = 24c", 3 + 1*9 == 12)

print("=== L. practical sheet Q2: 3D RRR, point masses (lecturer's special case)")
t = sp.symbols('t')
th = [sp.Function(f'th{i}')(t) for i in (1, 2, 3)]
L3, L4, m2, m3, I1, g = sp.symbols('L3 L4 m2 m3 I1 g', positive=True)
rho2 = L3*sp.cos(th[1]); h2 = L3*sp.sin(th[1])
rho3 = L3*sp.cos(th[1]) + L4*sp.cos(th[1] + th[2]); h3 = L3*sp.sin(th[1]) + L4*sp.sin(th[1] + th[2])
def ke_pt(mass, rho, h): return sp.Rational(1, 2)*mass*(sp.diff(rho, t)**2 + sp.diff(h, t)**2 + rho**2*sp.diff(th[0], t)**2)
K = sp.Rational(1, 2)*I1*sp.diff(th[0], t)**2 + ke_pt(m2, rho2, h2) + ke_pt(m3, rho3, h3)
P = g*(m2*h2 + m3*h3)
Lg = K - P
tau = []
for i in range(3):
    tau.append(sp.simplify(sp.diff(sp.diff(Lg, sp.diff(th[i], t)), t) - sp.diff(Lg, th[i])))
S = sp.symbols('q1 q2 q3'); SD = sp.symbols('d1 d2 d3'); SDD = sp.symbols('dd1 dd2 dd3')
sub = {}
for i in range(3):
    sub[sp.diff(th[i], t, 2)] = SDD[i]
for i in range(3):
    sub[sp.diff(th[i], t)] = SD[i]
for i in range(3):
    sub[th[i]] = S[i]
taus = [sp.simplify(x.subs(sub)) for x in tau]
M = sp.Matrix(3, 3, lambda i, j: sp.diff(taus[i], SDD[j]))
G = sp.Matrix([taus[i].subs({**{SD[j]: 0 for j in range(3)}, **{SDD[j]: 0 for j in range(3)}}) for i in range(3)])
c2, c3, c23 = sp.cos(S[1]), sp.cos(S[2]), sp.cos(S[1] + S[2])
M_claim = sp.Matrix([[I1 + m2*(L3*c2)**2 + m3*(L3*c2 + L4*c23)**2, 0, 0],
                     [0, m2*L3**2 + m3*(L3**2 + L4**2 + 2*L3*L4*c3), m3*(L4**2 + L3*L4*c3)],
                     [0, m3*(L4**2 + L3*L4*c3), m3*L4**2]])
G_claim = sp.Matrix([0, g*((m2 + m3)*L3*c2 + m3*L4*c23), g*m3*L4*c23])
chk("M matches the closed form (point-mass case)", sp.simplify(M - M_claim) == sp.zeros(3, 3))
chk("G matches the closed form", sp.simplify(G - G_claim) == sp.zeros(3, 1))
# velocity terms
Vv = sp.Matrix([taus[i].subs({**{SDD[j]: 0 for j in range(3)}, **{g: 0}}) for i in range(3)])
rho2s = L3*sp.cos(S[1]); rho3s = L3*sp.cos(S[1]) + L4*c23
V1 = -2*SD[0]*(m2*rho2s*L3*sp.sin(S[1])*SD[1] + m3*rho3s*(L3*sp.sin(S[1])*SD[1] + L4*sp.sin(S[1] + S[2])*(SD[1] + SD[2])))
V2 = SD[0]**2*(m2*rho2s*L3*sp.sin(S[1]) + m3*rho3s*(L3*sp.sin(S[1]) + L4*sp.sin(S[1] + S[2]))) - m3*L3*L4*sp.sin(S[2])*(2*SD[1]*SD[2] + SD[2]**2)
V3 = SD[0]**2*m3*rho3s*L4*sp.sin(S[1] + S[2]) + m3*L3*L4*sp.sin(S[2])*SD[1]**2
chk("V1 matches", sp.simplify(Vv[0] - V1) == 0); chk("V2 matches", sp.simplify(Vv[1] - V2) == 0); chk("V3 matches", sp.simplify(Vv[2] - V3) == 0)
# numeric test at rest: L3=0.4, L4=0.3, m2=2, m3=1, q=(0,30deg,40deg)
vals = {L3: 0.4, L4: 0.3, m2: 2, m3: 1, g: 9.81, I1: 0.5}
Gn = [float(x.subs(vals).subs({S[0]: 0, S[1]: r_(30), S[2]: r_(40)})) for x in G]
print('   G at (0,30,40 deg), L3=.4,L4=.3,m2=2,m3=1:', np.round(Gn, 4))
g2 = 9.81*((2 + 1)*0.4*np.cos(r_(30)) + 1*0.3*np.cos(r_(70))); g3 = 9.81*0.3*np.cos(r_(70))
close("G2, G3 by hand", [g2, g3], Gn[1:], 1e-6)
Mn = np.array(M.subs(vals).subs({S[0]: 0, S[1]: r_(30), S[2]: r_(40)}), float); print('   M =', np.round(Mn, 4).tolist())
chk("M symmetric positive definite", np.allclose(Mn, Mn.T) and np.all(np.linalg.eigvalsh(Mn) > 0))
# torque at q=(0,30,40), qdd=(1,0.5,-0.5), qd=0 : tau = M qdd + G
qdd = np.array([1, 0.5, -0.5]); tau_n = Mn @ qdd + np.array(Gn); print('   tau (qd=0, qdd=(1,.5,-.5)) =', np.round(tau_n, 4))

print("=== M. practical sheet Q3 (planar 3R along a cubic): Lagrange, point masses at distal ends")
Lp = (3.0, 2.0, 1.0); mp_ = (3.0, 2.0, 1.0)
q3 = [sp.Function(f'p{i}')(t) for i in (1, 2, 3)]
pts = []; xa = ya = 0; ang = 0
for i in range(3):
    ang = ang + q3[i]; xa = xa + Lp[i]*sp.cos(ang); ya = ya + Lp[i]*sp.sin(ang); pts.append((xa, ya))
K3 = sum(sp.Rational(1, 2)*mp_[i]*(sp.diff(pts[i][0], t)**2 + sp.diff(pts[i][1], t)**2) for i in range(3))
P3 = sum(mp_[i]*9.81*pts[i][1] for i in range(3))
L3g = K3 - P3
tau3 = [sp.diff(sp.diff(L3g, sp.diff(q3[i], t)), t) - sp.diff(L3g, q3[i]) for i in range(3)]
S3 = sp.symbols('s1:4'); SD3 = sp.symbols('sd1:4'); SDD3 = sp.symbols('sdd1:4')
sub3 = {**{sp.diff(q3[i], t, 2): SDD3[i] for i in range(3)}, **{sp.diff(q3[i], t): SD3[i] for i in range(3)}, **{q3[i]: S3[i] for i in range(3)}}
f3 = sp.lambdify([S3, SD3, SDD3], [x.subs(sub3) for x in tau3], 'numpy')
q0 = np.array([0, 0, np.pi/2]); qf = np.array([np.pi/2, np.pi/2, 0]); T = 2.0
a2 = 3*(qf - q0)/T**2; a3 = -2*(qf - q0)/T**3
rows = []
for tt in (0.0, 1.0, 2.0):
    qv = q0 + a2*tt**2 + a3*tt**3; qdv = 2*a2*tt + 3*a3*tt**2; qddv = 2*a2 + 6*a3*tt
    tv = np.array(f3(qv, qdv, qddv), float); rows.append(tv)
    print(f'   t={tt}: q(deg)={np.round(d(qv),3)} qd={np.round(qdv,4)} qdd={np.round(qddv,4)} tau={np.round(tv,3)}')
close("tau(0) = (548.814, 160.176, 2.356)", rows[0], [548.814, 160.176, 2.356], 1e-3)
close("tau(1) = (70.817, 20.781, 5.078)", rows[1], [70.817, 20.781, 5.078], 1e-3)
close("tau(2) = (-268.947, -141.712, -21.591)", rows[2], [-268.947, -141.712, -21.591], 1e-3)
close("g1 at t=0: 9.81*(3*3 + 2*5 + 1*5) = 235.44", 9.81*(3*3 + 2*5 + 1*5), 235.44, 1e-9)
qd_peak = a2*1.0 + 0  # velocity at t=T/2 : 1.5*Delta/T
close("peak joint speed 1.5 Delta/T = 1.1781 rad/s for 90 deg in 2 s", 1.5*(np.pi/2)/2, 1.1781, 1e-4)
close("qdd(0) = 6 Delta/T^2 = 2.3562 rad/s^2", 6*(np.pi/2)/4, 2.3562, 1e-4)
# open loop with an initial error: what happens at joint 1 if the model error delta tau=10% at t=0?
print("\nFAILED:" if fails else "\nALL V2_W08 CHECKS PASSED", fails)
