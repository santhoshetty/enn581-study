"""Mock exams 1 and 2 — computes and verifies every number in the model answers. Run: python3 site/src/checks/mock.py"""
import json, os, sys, heapq
sys.path.insert(0, os.path.dirname(__file__))
from common import *
out = {'m1': {}, 'm2': {}}

# ======================= MOCK 1 =======================
# Q1 SCARA (RRP): tz(H) Rz(t1) tx(a1) Rz(t2) tx(a2) tz(-d3) Rx(180)
H, a1, a2 = 0.40, 0.35, 0.25
def scara(t1, t2, d3): return nchain(tz(H), Rz(t1), tx(a1), Rz(t2), tx(a2), tz(-d3), Rx(np.pi))
T = scara(30*deg, -60*deg, 0.10)
print('M1Q1 T =\n', np.round(T, 4))
check('M1Q1: R proper, tool z points down', is_rotation(T[:3, :3]) and np.isclose(T[2, 2], -1))
# proximal DH of the same arm: rows (alpha, a, theta, d): (0,0,t1,H) (0,a1,t2,0) (180,a2,0,d3) -> check equals ETS up to the tool
Tdh = nchain(DH_prox(0, 0, 30*deg, H), DH_prox(0, a1, -60*deg, 0), DH_prox(np.pi, a2, 0, 0.10))
close('M1Q1: proximal DH (0,0,θ1,H)(0,a1,θ2,0)(180°,a2,0,d3) == ETS', Tdh, T, show=False)
out['m1']['q1'] = dict(T=np.round(T, 4).tolist())
# Q2 IK of the SCARA for p = (0.45, 0.20, 0.25)
px, py, pz = 0.45, 0.20, 0.25
d3 = H - pz
c2 = (px**2 + py**2 - a1**2 - a2**2)/(2*a1*a2)
sols = []
for s in (1, -1):
    t2 = np.arctan2(s*np.sqrt(1 - c2**2), c2); t1 = np.arctan2(py, px) - np.arctan2(a2*np.sin(t2), a1 + a2*np.cos(t2))
    sols.append((t1, t2, d3))
check('M1Q2: both IK solutions reproduce p', all(np.allclose(scara(*s_)[:3, 3], [px, py, pz]) for s_ in sols))
print('M1Q2: c2 =', round(c2, 4), ' sols (deg, deg, m):', [(round(s_[0]/deg, 3), round(s_[1]/deg, 3), round(s_[2], 3)) for s_ in sols])
out['m1']['q2'] = dict(c2=r(c2), sols=[[r(s_[0]/deg, 3), r(s_[1]/deg, 3), r(s_[2], 3)] for s_ in sols], rin=r(abs(a1 - a2)), rout=r(a1 + a2))
# Q3 mobility
check('M1Q3a: planar five-bar 3(5-5-1)+5 = 2', 3*(5 - 5 - 1) + 5 == 2)
check('M1Q3b: 3-RPS 6(8-9-1)+(3+3+9) = 3', 6*(8 - 9 - 1) + (3*1 + 3*1 + 3*3) == 3)
check('M1Q3c: 3-SPS (S at both ends) 6(8-9-1)+21 = 9, R_M = 3 -> 6', 6*(8 - 9 - 1) + (3*3 + 3*1 + 3*3) == 9 and 9 - 3 == 6)
# Q4 two cubics, continuous acceleration, equal durations (W13 Q8 closed form), theta0=0, thv=40, thg=30 deg, tf=1 s
th0, thv, thg, tf = 0.0, 40.0, 30.0, 1.0
c = dict(a10=th0, a11=0, a12=(12*thv - 3*thg - 9*th0)/(4*tf**2), a13=(-8*thv + 3*thg + 5*th0)/(4*tf**3),
         a20=thv, a21=(3*thg - 3*th0)/(4*tf), a22=(-12*thv + 6*thg + 6*th0)/(4*tf**2), a23=(8*thv - 5*thg - 3*th0)/(4*tf**3))
p1 = lambda t: c['a10'] + c['a11']*t + c['a12']*t**2 + c['a13']*t**3
p2 = lambda t: c['a20'] + c['a21']*t + c['a22']*t**2 + c['a23']*t**3
v1 = lambda t: c['a11'] + 2*c['a12']*t + 3*c['a13']*t**2; v2 = lambda t: c['a21'] + 2*c['a22']*t + 3*c['a23']*t**2
A1 = lambda t: 2*c['a12'] + 6*c['a13']*t; A2 = lambda t: 2*c['a22'] + 6*c['a23']*t
check('M1Q4: spline conditions (pos, rest, via continuity of vel and acc)', np.allclose([p1(0), v1(0), p1(tf), p2(0), p2(tf), v2(tf), v1(tf) - v2(0), A1(tf) - A2(0)], [th0, 0, thv, thv, thg, 0, 0, 0]))
print('M1Q4 coefficients:', {k: round(v, 4) for k, v in c.items()})
tt = np.linspace(0, tf, 100001); vmax1 = max(v1(tt))
out['m1']['q4'] = {k: r(v, 4) for k, v in c.items()}; out['m1']['q4']['vmax1'] = r(vmax1, 3)
print('   peak speed in segment 1 =', round(vmax1, 3), 'deg/s at t =', round(tt[np.argmax(v1(tt))], 4))
# Q5 2R Jacobian, singularity, statics
L1, L2 = 0.5, 0.4; q = np.array([30, 60])*deg
J = np.array([[-L1*np.sin(q[0]) - L2*np.sin(q.sum()), -L2*np.sin(q.sum())], [L1*np.cos(q[0]) + L2*np.cos(q.sum()), L2*np.cos(q.sum())]])
F = np.array([0, -10.0]); tau = J.T @ F
detJ = np.linalg.det(J)
check('M1Q5: det J = L1 L2 sin q2', np.isclose(detJ, L1*L2*np.sin(q[1])))
xd = np.array([0.1, 0.0]); qd = np.linalg.solve(J, xd)
print('M1Q5: J =', np.round(J, 4).tolist(), ' det =', round(detJ, 4), ' tau =', np.round(tau, 4).tolist(), ' qdot for xdot=(0.1,0):', np.round(qd, 4).tolist())
out['m1']['q5'] = dict(J=np.round(J, 4).tolist(), det=r(detJ), tau=np.round(tau, 4).tolist(), qd=np.round(qd, 4).tolist())
# Q6 RP arm (point mass m at slider distance d; massless link; vertical plane, theta from horizontal)
m, g_ = 2.0, 9.81; th, d, thd, dd, thdd, ddd = 30*deg, 0.6, 0.5, 0.2, 1.0, -0.3
tau1 = m*d**2*thdd + 2*m*d*dd*thd + m*g_*d*np.cos(th)
f2 = m*ddd - m*d*thd**2 + m*g_*np.sin(th)
print(f'M1Q6: tau1 = {tau1:.4f} N·m, f2 = {f2:.4f} N')
# verify against generic Lagrange
t_ = sp.symbols('t'); TH, D_ = sp.Function('TH')(t_), sp.Function('D')(t_)
Ke = sp.Rational(1, 2)*m*(sp.diff(D_*sp.cos(TH), t_)**2 + sp.diff(D_*sp.sin(TH), t_)**2); Pe = m*g_*D_*sp.sin(TH); Lg = Ke - Pe
eqs = [sp.diff(sp.diff(Lg, sp.diff(x_, t_)), t_) - sp.diff(Lg, x_) for x_ in (TH, D_)]
sub = {sp.diff(TH, t_, 2): thdd, sp.diff(D_, t_, 2): ddd, sp.diff(TH, t_): thd, sp.diff(D_, t_): dd}
vals = [float(e.subs(sub).subs({TH: th, D_: d})) for e in eqs]
close('M1Q6: hand formulas == Lagrange', vals, [tau1, f2])
out['m1']['q6'] = dict(tau1=r(tau1), f2=r(f2))
# Q7 partitioned control
mm, bb, kk = 2.0, 0.5, 4.0; wn = 8.0; kp = wn**2; kv = 2*wn; fdist = 5.0
ess = fdist/(kp*mm)
x, xdot, xd_, xdd_d = 0.10, -0.20, 0.15, 0.0
E, Ed = xd_ - x, 0.0 - xdot
fp = xdd_d + kv*Ed + kp*E; f = mm*fp + bb*xdot + kk*x
print(f'M1Q7: kp = {kp}, kv = {kv}, Ess = {ess:.5f} m, at state: f\' = {fp:.3f}, f = {f:.3f} N')
check('M1Q7: plant acceleration equals f\'', np.isclose((f - bb*xdot - kk*x)/mm, fp))
out['m1']['q7'] = dict(kp=kp, kv=kv, ess=r(ess, 6), fp=r(fp, 4), f=r(f, 4))
# Q8 differential drive
r_, a_ = 0.08, 0.2; ph1, ph2 = 5.0, 7.5
u = r_*(ph1 + ph2)/2; w = r_*(ph2 - ph1)/(2*a_); yicr = u/w
print(f'M1Q8: u = {u:.4f}, thd = {w:.4f}, ICR y = {yicr:.4f} m')
check('M1Q8: ICR y = a(φ1+φ2)/(φ2-φ1)', np.isclose(yicr, a_*(ph1 + ph2)/(ph2 - ph1)))
# wheel speeds for a desired motion: u = 0.3, thd = -0.5
ud, wd = 0.3, -0.5
w1, w2 = (ud - a_*wd)/r_, (ud + a_*wd)/r_
print(f'   inverse: u=0.3, thd=-0.5 -> phid = ({w1:.4f}, {w2:.4f})')
out['m1']['q8'] = dict(u=r(u), thd=r(w), icr=r(yicr), inv=[r(w1), r(w2)])
# Q9 A* on a small graph
G = {'S': {'A': 2, 'B': 5}, 'A': {'S': 2, 'C': 4, 'D': 7}, 'B': {'S': 5, 'D': 3}, 'C': {'A': 4, 'G': 6}, 'D': {'A': 7, 'B': 3, 'G': 2}, 'G': {'C': 6, 'D': 2}}
h = {'S': 7, 'A': 6, 'B': 4, 'C': 5, 'D': 2, 'G': 0}
def dijkstra_all(goal):
    dist = {goal: 0}; pq = [(0, goal)]
    while pq:
        dd_, n = heapq.heappop(pq)
        if dd_ > dist[n]: continue
        for mm_, w_ in G[n].items():
            if dd_ + w_ < dist.get(mm_, 1e9): dist[mm_] = dd_ + w_; heapq.heappush(pq, (dd_ + w_, mm_))
    return dist
true = dijkstra_all('G')
check('M1Q9: h admissible', all(h[n] <= true[n] for n in G), (h, true))
check('M1Q9: h consistent', all(h[n] <= G[n][mm_] + h[mm_] for n in G for mm_ in G[n]))
def astar(s, goal):
    openl = [(h[s], 0, s)]; g = {s: 0}; par = {s: None}; closed = []; order = []
    while openl:
        f_, g_, n = heapq.heappop(openl)
        if n in closed: continue
        closed.append(n); order.append((n, g_, h[n], f_))
        if n == goal: break
        for mm_, w_ in sorted(G[n].items()):
            if mm_ in closed: continue
            if g_ + w_ < g.get(mm_, 1e9): g[mm_] = g_ + w_; par[mm_] = n; heapq.heappush(openl, (g[mm_] + h[mm_], g[mm_], mm_))
    path = [goal]
    while par[path[-1]]: path.append(par[path[-1]])
    return order, path[::-1], g[goal]
order, path, cost = astar('S', 'G')
print('M1Q9: A* order', order, 'path', path, 'cost', cost, 'true costs', true)
check('M1Q9: A* cost equals the true shortest cost', cost == true['S'])
out['m1']['q9'] = dict(order=order, path=path, cost=cost, true=true)

# ======================= MOCK 2 =======================
# Q1 DH of an RRP "Stanford-like" arm: joint 1 about z0 (vertical), joint 2 about horizontal axis through the same point, prismatic along the arm
# proximal DH: (alpha0=0, a0=0, th1, d1=H) (alpha1=-90, a1=0, th2, 0) (alpha2=+90, a2=0, 0, d3)
H2 = 0.5
def stan(t1, t2, d3):
    return nchain(DH_prox(0, 0, t1, H2), DH_prox(-np.pi/2, 0, t2, 0), DH_prox(np.pi/2, 0, 0, d3))
Ts = stan(40*deg, 30*deg, 0.6)
pos_formula = np.array([np.cos(40*deg)*0.6*np.sin(30*deg), np.sin(40*deg)*0.6*np.sin(30*deg), H2 + 0.6*np.cos(30*deg)])
close('M2Q1: tip = (c1 d3 s2, s1 d3 s2, H + d3 c2) (theta2 measured from vertical)', Ts[:3, 3], pos_formula)
print('M2Q1 T =\n', np.round(Ts, 4))
out['m2']['q1'] = dict(T=np.round(Ts, 4).tolist())
# Q2 IK of that arm for p = (0.3, 0.4, 0.9)
P2 = np.array([0.3, 0.4, 0.9]); rr = np.hypot(P2[0], P2[1]); zz = P2[2] - H2
d3s = np.hypot(rr, zz); t2s = np.arctan2(rr, zz); t1s = np.arctan2(P2[1], P2[0])
check('M2Q2: (t1, t2, d3) reproduces p', np.allclose(stan(t1s, t2s, d3s)[:3, 3], P2))
check('M2Q2: second solution (t1+pi, -t2, d3) also reproduces p', np.allclose(stan(t1s + np.pi, -t2s, d3s)[:3, 3], P2))
print(f'M2Q2: t1 = {t1s/deg:.3f}°, t2 = {t2s/deg:.3f}°, d3 = {d3s:.4f} m')
out['m2']['q2'] = dict(t1=r(t1s/deg, 3), t2=r(t2s/deg, 3), d3=r(d3s))
# Q3 four-bar FK numbers: L0=0.8, L1=0.3, L2=0.7, L3=0.5; theta1 = 60 deg; both modes; Grashof
L0b, L1b, L2b, L3b = 0.8, 0.3, 0.7, 0.5
A = L1b*np.array([np.cos(60*deg), np.sin(60*deg)]); Dp = np.array([L0b, 0])
dAD = np.linalg.norm(Dp - A); xx = (L2b**2 - L3b**2 + dAD**2)/(2*dAD); hh = np.sqrt(L2b**2 - xx**2)
uu = (Dp - A)/dAD; nn = np.array([-uu[1], uu[0]])
Bs = [A + xx*uu + s*hh*nn for s in (1, -1)]
phis = [np.degrees(np.arctan2(*(B_ - A)[::-1])) for B_ in Bs]; phi1s = [np.degrees(np.arctan2(B_[1], B_[0] - L0b)) for B_ in Bs]
check('M2Q3: both B satisfy |B-D| = L3', all(np.isclose(np.linalg.norm(B_ - Dp), L3b) for B_ in Bs))
srt = sorted([L0b, L1b, L2b, L3b]); grashof = srt[0] + srt[3] <= srt[1] + srt[2]
check('M2Q3: Grashof (0.3+0.8 <= 0.5+0.7): the shortest link (crank, L1) turns fully', grashof)
print('M2Q3: |AD| =', round(dAD, 4), ' B =', [np.round(B_, 4).tolist() for B_ in Bs], ' phi =', np.round(phis, 3), ' phi1 =', np.round(phi1s, 3))
out['m2']['q3'] = dict(AD=r(dAD), B=[np.round(B_, 4).tolist() for B_ in Bs], phi=[r(x_, 3) for x_ in phis], phi1=[r(x_, 3) for x_ in phi1s])
# Q4 LSPB: theta 10 -> 70 deg in 3 s, blend acceleration 30 deg/s^2
th0b, thfb, tfb, acc = 10.0, 70.0, 3.0, 30.0
amin = 4*(thfb - th0b)/tfb**2
tb = tfb/2 - np.sqrt(acc**2*tfb**2 - 4*acc*(thfb - th0b))/(2*acc); vl = acc*tb
thb = th0b + 0.5*acc*tb**2
end = thb + vl*(tfb - 2*tb) + vl*tb - 0.5*acc*tb**2
check('M2Q4: LSPB ends at theta_f', np.isclose(end, thfb))
print(f'M2Q4: min acc = {amin:.4f} deg/s², tb = {tb:.4f} s, linear speed = {vl:.4f} deg/s, theta at end of blend = {thb:.4f} deg')
out['m2']['q4'] = dict(amin=r(amin), tb=r(tb), v=r(vl), thb=r(thb))
# Q5 Jacobian of the W13 RRR arm (position part), singularities
L = dict(L12=1.1, L3=0.4, L4=0.3)
def fkp(t):
    rr_ = L['L3']*np.cos(t[1]) + L['L4']*np.cos(t[1] + t[2])
    return np.array([np.cos(t[0])*rr_, np.sin(t[0])*rr_, L['L12'] + L['L3']*np.sin(t[1]) + L['L4']*np.sin(t[1] + t[2])])
def Jan(t):
    c1, s1 = np.cos(t[0]), np.sin(t[0]); s2, c2 = np.sin(t[1]), np.cos(t[1]); s23, c23 = np.sin(t[1] + t[2]), np.cos(t[1] + t[2])
    rr_ = L['L3']*c2 + L['L4']*c23
    return np.array([[-s1*rr_, -c1*(L['L3']*s2 + L['L4']*s23), -c1*L['L4']*s23],
                     [c1*rr_, -s1*(L['L3']*s2 + L['L4']*s23), -s1*L['L4']*s23],
                     [0, L['L3']*c2 + L['L4']*c23, L['L4']*c23]])
tq = np.array([30, 20, 40])*deg
Jn = np.array([(fkp(tq + e) - fkp(tq - e))/(2e-7) for e in np.eye(3)*1e-7]).T
close('M2Q5: analytic J == numerical derivative of FK', Jan(tq), Jn, tol=1e-6, show=False)
rgn = np.random.default_rng(4); okdet = True
for _ in range(200):
    tt_ = rgn.uniform(-np.pi, np.pi, 3)
    okdet &= np.isclose(np.linalg.det(Jan(tt_)), -L['L3']*L['L4']*np.sin(tt_[2])*(L['L3']*np.cos(tt_[1]) + L['L4']*np.cos(tt_[1] + tt_[2])))
check('M2Q5: det J = -L3 L4 sin(t3)(L3 c2 + L4 c23) at 200 random poses', okdet)
check('M2Q5: singular at t3 = 0 (stretched) and when the wrist is over the base axis (r = 0)', abs(np.linalg.det(Jan(np.array([0.3, 0.5, 0.0])))) < 1e-12 and abs(np.linalg.det(Jan(np.array([0.3, __import__('scipy.optimize', fromlist=['brentq']).brentq(lambda b_: 0.4*np.cos(b_) + 0.3*np.cos(b_ + np.pi/3), 0.5, 2.5), np.pi/3])))) < 1e-12)
print('M2Q5: J at (30,20,40)° =', np.round(Jan(tq), 4).tolist(), ' det =', round(np.linalg.det(Jan(tq)), 5))
out['m2']['q5'] = dict(J=np.round(Jan(tq), 4).tolist(), det=r(np.linalg.det(Jan(tq)), 5))
# Q6 2R with uniform rods (mass m_i, length L_i, COM at L_i/2, I_c = m L^2/12), vertical plane, gravity -y, at rest (qd = 0)
m1, m2, l1, l2 = 3.0, 2.0, 0.6, 0.4
q_ = np.array([45, -30])*deg; qdd = np.array([1.0, 2.0])
I1, I2 = m1*l1**2/12, m2*l2**2/12
c2_ = np.cos(q_[1])
M11 = I1 + m1*(l1/2)**2 + I2 + m2*(l1**2 + (l2/2)**2 + 2*l1*(l2/2)*c2_)
M12 = I2 + m2*((l2/2)**2 + l1*(l2/2)*c2_); M22 = I2 + m2*(l2/2)**2
G1 = (m1*l1/2 + m2*l1)*g_*np.cos(q_[0]) + m2*(l2/2)*g_*np.cos(q_.sum()); G2 = m2*(l2/2)*g_*np.cos(q_.sum())
tau6 = np.array([[M11, M12], [M12, M22]]) @ qdd + np.array([G1, G2])
# verify with sympy Lagrange
q1s, q2s = sp.Function('q1')(t_), sp.Function('q2')(t_)
xc1, yc1 = l1/2*sp.cos(q1s), l1/2*sp.sin(q1s)
xc2, yc2 = l1*sp.cos(q1s) + l2/2*sp.cos(q1s + q2s), l1*sp.sin(q1s) + l2/2*sp.sin(q1s + q2s)
Kk = sp.Rational(1, 2)*m1*(sp.diff(xc1, t_)**2 + sp.diff(yc1, t_)**2) + sp.Rational(1, 2)*I1*sp.diff(q1s, t_)**2 + sp.Rational(1, 2)*m2*(sp.diff(xc2, t_)**2 + sp.diff(yc2, t_)**2) + sp.Rational(1, 2)*I2*sp.diff(q1s + q2s, t_)**2
Pp = m1*g_*yc1 + m2*g_*yc2; Lg2 = Kk - Pp
eq2 = [sp.diff(sp.diff(Lg2, sp.diff(x_, t_)), t_) - sp.diff(Lg2, x_) for x_ in (q1s, q2s)]
sub2 = {sp.diff(q1s, t_, 2): qdd[0], sp.diff(q2s, t_, 2): qdd[1], sp.diff(q1s, t_): 0, sp.diff(q2s, t_): 0}
vals2 = [float(e.subs(sub2).subs({q1s: q_[0], q2s: q_[1]})) for e in eq2]
close('M2Q6: hand M, g with rod inertias == Lagrange', vals2, tau6)
print(f'M2Q6: M = [[{M11:.4f}, {M12:.4f}], [{M12:.4f}, {M22:.4f}]], g = ({G1:.4f}, {G2:.4f}), tau = {np.round(tau6, 4).tolist()}')
out['m2']['q6'] = dict(M=[[r(M11), r(M12)], [r(M12), r(M22)]], g=[r(G1), r(G2)], tau=np.round(tau6, 4).tolist())
# Q7 PID: kp = 49, kv = 14 (critical), choose ki; stability limit; steady state
kp2, kv2 = 49.0, 14.0
check('M2Q7: kv² = 4 kp (critical)', np.isclose(kv2**2, 4*kp2))
ki_lim = kv2*kp2
roots = np.roots([1, kv2, kp2, 100.0])
check('M2Q7: ki = 100 < kv kp = 686 -> all roots in the left half plane', np.all(roots.real < 0))
print('M2Q7: stability limit ki <', ki_lim, ' roots at ki=100:', np.round(roots, 4))
out['m2']['q7'] = dict(lim=ki_lim, roots=[str(np.round(x_, 4)) for x_ in roots])
# Q8 four mecanum wheels (gamma = ±45°), standard layout, r, half-length lx, half-width ly
rr8, lx, ly = 0.05, 0.2, 0.15
wheels = [(lx, ly, 0, -45), (lx, -ly, 0, 45), (-lx, ly, 0, 45), (-lx, -ly, 0, -45)]   # (x, y, beta, gamma deg)
rows = []
for (xw, yw, be, ga) in wheels:
    gr = np.radians(ga); Rt = np.array([[np.cos(be), np.sin(be)], [-np.sin(be), np.cos(be)]])
    rows.append(np.array([1/rr8, np.tan(gr)/rr8]) @ Rt @ np.array([[1, 0, -yw], [0, 1, xw]]))
Wm = np.array(rows)
zeta = np.array([0.0, 0.2, 0.0])   # pure sideways
phid8 = Wm @ zeta
print('M2Q8: rows (rad/s per m/s):', np.round(Wm, 3).tolist(), ' sideways 0.2 m/s ->', np.round(phid8, 3).tolist())
check('M2Q8: rank 3 -> holonomic', np.linalg.matrix_rank(Wm) == 3)
out['m2']['q8'] = dict(W=np.round(Wm, 3).tolist(), side=np.round(phid8, 3).tolist())
json.dump(out, open(os.path.join(os.path.dirname(__file__), 'mock_vectors.json'), 'w'), indent=1, default=str)
report()
