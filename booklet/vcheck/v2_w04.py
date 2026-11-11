"""Verify every number in booklet/w04.tex (Week 4: parallel manipulators). Prints labelled values; exits 1 on any FAIL."""
import numpy as np, sympy as sp
deg = np.pi/180
def ok(name, cond, extra=''):
    print(('PASS ' if cond else 'FAIL ') + name + (('  -- ' + extra) if extra else ''))
    if not cond: raise SystemExit(1)
def GK(D, N, J, SF, Rc=0, Rm=0): return D*(N-J-1) + SF + Rc - Rm

# ---------------- 1. mobility counts ----------------
print('== mobility ==')
ok('3-PRR: 3(8-9-1)+9 = 3', GK(3,8,9,9) == 3)
ok('simplest delta: 6(8-9-1)+15 = 3', GK(6,8,9,3*(1+2+2)) == 3)
ok('simplest delta with the +1 slip: 6(8-9+1)+15 = 15', 6*(8-9+1)+15 == 15)
ok('common delta plain: 6(11-15-1)+39 = 9', GK(6,11,15,3*(1+4*3)) == 9)
ok('common delta extended: 9+0-6 = 3', GK(6,11,15,39,0,6) == 3)
# 4RRR
loop1 = {'Ty','Tz','Rx'}; loop2 = {'Tx','Tz','Ry'}
union = loop1 | loop2
ok('4RRR: |union| = 5', len(union) == 5)
Rc = 12 - 5
ok('4RRR: plain -6, Rc=7, M=1', GK(6,10,12,12) == -6 and GK(6,10,12,12,Rc,0) == 1)
ok('four-bar M=1', GK(3,4,4,4) == 1)
ok('slider-crank planar M=1; with D=6 it would give -2', GK(3,4,4,4) == 1 and GK(6,4,4,4) == -2)
ok('planar five-bar M=2', GK(3,5,5,5) == 2)
ok('Gough-Stewart 6-UPS M=6', GK(6,14,18,6*(2+1+3)) == 6)
ok('3-RPS plain = 3 (R_M=0)', GK(6,8,9,3*(1+1+3)) == 3)
ok('3-SPS plain = 9, extended with R_M=3 -> 6', GK(6,8,9,3*(3+1+3)) == 9 and GK(6,8,9,21,0,3) == 6)
ok('Q10 mock: three equal parallel links N=5,J=6 -> 0', GK(3,5,6,6) == 0)
ok('2 planar bodies unjoined 6 DoF, with an R joint: 4  (3(3-1-1)+1=... ) check D(N-J-1)+F', GK(3,3,1,1) == 4)
ok('spatial slider-crank-like N=6,J=7 (5R,1U,1S): sumF=10 -> -2', GK(6,6,7,5+2+3) == -2)
# ground included: two bodies + ground, one R joint (pendulum): N=2, J=1
ok('pendulum N=2 J=1 R : 3(2-1-1)+1 = 1', GK(3,2,1,1) == 1)
ok('rigid triangle truss of 3 links incl ground, 3 R: 3(3-3-1)+3 = 0 (structure)', GK(3,3,3,3) == 0)

# ---------------- 2. four-bar FK/IK ----------------
print('== four-bar (L0=1.0, L1=0.5, L3=0.6, a=0.2, b=0.25, L2=0.45) ==')
L0,L1,L3,a,b = 1.0,0.5,0.6,0.2,0.25; L2 = a+b
def fb(th1, mode, L0=L0,L1=L1,L2=L2,L3=L3,a=a):
    A = L1*np.array([np.cos(th1), np.sin(th1)]); D = np.array([L0,0.])
    d = np.linalg.norm(D-A)
    if d > L2+L3+1e-12 or d < abs(L2-L3)-1e-12: return None
    xa = (L2**2-L3**2+d**2)/(2*d); h = np.sqrt(max(L2**2-xa**2,0))
    u = (D-A)/d; n = np.array([-u[1],u[0]])
    B = A+xa*u+mode*h*n
    phi = np.arctan2(*(B-A)[::-1]); K = A + a*np.array([np.cos(phi),np.sin(phi)])
    phi1 = np.arctan2(B[1], B[0]-L0)
    return dict(A=A,B=B,K=K,phi=phi,phi1=phi1,AD=d,xa=xa,h=h)
th1 = 70*deg
for mode in (+1,-1):
    r = fb(th1,mode)
    th_ = th1; phi2 = r['phi']-th_; phi3 = np.mod(r['phi']+np.pi-r['phi1'], 2*np.pi)
    print(f"mode {mode:+d}: A={np.round(r['A'],4)} |AD|={r['AD']:.4f} xa={r['xa']:.4f} h={r['h']:.4f} B={np.round(r['B'],4)} K={np.round(r['K'],4)}")
    print(f"   phi={r['phi']/deg:.3f} deg  phi2={np.mod(phi2,2*np.pi)/deg:.3f} (={phi2/deg:.3f}) phi1={r['phi1']/deg:.3f} phi3={phi3/deg:.3f}  phi1+phi3={np.mod(r['phi1']+phi3,2*np.pi)/deg:.3f}")
    # three loop-closure equations (slide p.54)
    lhs = [L1*np.cos(th_)+a*np.cos(th_+phi2), L1*np.sin(th_)+a*np.sin(th_+phi2)]
    rhs = [L0+L3*np.cos(r['phi1'])+b*np.cos(r['phi1']+phi3), L3*np.sin(r['phi1'])+b*np.sin(r['phi1']+phi3)]
    ok(f'  mode {mode:+d}: loop eq. 1,2 hold', np.allclose(lhs,rhs,atol=1e-12) and np.allclose(lhs,r['K'],atol=1e-12), f'resid {np.abs(np.array(lhs)-np.array(rhs)).max():.1e}')
    ok(f'  mode {mode:+d}: theta1+phi2 = phi1+phi3+pi (mod 2pi)', abs(np.angle(np.exp(1j*(th_+phi2-r['phi1']-phi3-np.pi)))) < 1e-12)
    eta = (L1*np.cos(th_)-L0-L3*np.cos(r['phi1']))**2+(L1*np.sin(th_)-L3*np.sin(r['phi1']))**2-L2**2
    ok(f'  mode {mode:+d}: Freudenstein eta1=0', abs(eta) < 1e-12)
    # analytic phi1 from Freudenstein: u cos + v sin = k
    u_ = L1*np.cos(th_)-L0; v_ = L1*np.sin(th_); k = (u_**2+v_**2+L3**2-L2**2)/(2*L3); rho = np.hypot(u_,v_)
    cands = [np.arctan2(v_,u_)+s*np.arccos(k/rho) for s in (+1,-1)]
    ok(f'  mode {mode:+d}: phi1 is one of atan2(v,u) +- acos(k/rho)', any(abs(np.angle(np.exp(1j*(c-r['phi1'])))) < 1e-9 for c in cands))
    if mode == 1:
        print(f"   Freudenstein: u={u_:.4f} v={v_:.4f} k={k:.4f} rho={rho:.4f} acos={np.arccos(k/rho)/deg:.3f} atan2={np.arctan2(v_,u_)/deg:.3f} candidates {[round(c/deg,3) for c in cands]}")
    # IK round trip (Q5)
    x,y = r['K']; phi = r['phi']
    t1 = np.arctan2(y-a*np.sin(phi), x-a*np.cos(phi)); p2 = phi-t1
    p1 = np.arctan2(y+b*np.sin(phi), x+b*np.cos(phi)-L0); p3 = phi+np.pi-p1
    print(f"   IK: theta1={t1/deg:.4f} phi2={np.mod(p2,2*np.pi)/deg:.3f} phi1={np.mod(p1,2*np.pi)/deg:.3f} phi3={np.mod(p3,2*np.pi)/deg:.3f}")
    ok(f'  mode {mode:+d}: IK round trip theta1=70', abs(t1-th1) < 1e-9)
    ok(f'  mode {mode:+d}: |B-D| = L3 and |A-B| = L2', abs(np.linalg.norm(r['B']-[L0,0])-L3) < 1e-12 and abs(np.linalg.norm(r['B']-r['A'])-L2) < 1e-12)

# W13 Q5 statement values: K=(0.3710,0.4678) phi=-0.600deg
r = fb(th1,+1); print('mode+1 K exact', r['K'], 'phi', r['phi']/deg)
Kq = np.array([0.3710,0.4678]); phq = -0.600*deg
t1 = np.arctan2(Kq[1]-a*np.sin(phq), Kq[0]-a*np.cos(phq)); p1 = np.arctan2(Kq[1]+b*np.sin(phq), Kq[0]+b*np.cos(phq)-L0)
print(f'Q5 with rounded data: theta1={t1/deg:.3f} phi2={np.mod(phq-t1,2*np.pi)/deg:.3f} phi1={np.mod(p1,2*np.pi)/deg:.3f} phi3={np.mod(phq+np.pi-p1,2*np.pi)/deg:.3f}')
print('   with W13 p.11 printed relation phi3 = phi1 - phi + pi:', np.mod(p1-phq+np.pi,2*np.pi)/deg, ' (true needs phi1+phi3 = phi+pi =', (phq+np.pi)/deg,')')
bad_phi3 = p1-phq+np.pi
Kbad = np.array([L0+L3*np.cos(p1)+b*np.cos(p1+bad_phi3), L3*np.sin(p1)+b*np.sin(p1+bad_phi3)])
print('   K reproduced with printed relation:', np.round(Kbad,4), ' vs given', Kq)
ok('printed W13 p.11 relation misses K', np.linalg.norm(Kbad-Kq) > 0.05)
good_phi3 = phq+np.pi-p1
Kgood = np.array([L0+L3*np.cos(p1)+b*np.cos(p1+good_phi3), L3*np.sin(p1)+b*np.sin(p1+good_phi3)])
ok('phi3 = phi+pi-phi1 hits K', np.linalg.norm(Kgood-Kq) < 1e-3, f'{Kgood}')

# ---- slide FK route (A_i, B_i, C_i) at mode +1 ----
print('== slide route pp.62-66, mode +1 ==')
r = fb(th1,+1); x,y = r['K']; delta = r['phi']
A1,B1,C1 = x-L0, y, ((x-L0)**2+y**2+b**2-L3**2)/(2*b)
A2,B2,C2 = x, y, (L1**2-a**2-x**2-y**2)/(2*a)
print(f'A1={A1:.4f} B1={B1:.4f} C1={C1:.4f}  A2={A2:.4f} B2={B2:.4f} C2={C2:.4f}')
ok('A1 cos d + B1 sin d + C1 = 0', abs(A1*np.cos(delta)+B1*np.sin(delta)+C1) < 1e-12)
ok('A2 cos d + B2 sin d + C2 = 0', abs(A2*np.cos(delta)+B2*np.sin(delta)+C2) < 1e-12)
Dd = A1*B2-A2*B1
cd = (B1*C2-B2*C1)/Dd; sd = (A2*C1-A1*C2)/Dd
print(f'Delta={Dd:.4f} cos d={cd:.4f} sin d={sd:.5f} delta={np.arctan2(sd,cd)/deg:.3f}')
ok('Cramer gives cos,sin of delta', abs(cd-np.cos(delta)) < 1e-12 and abs(sd-np.sin(delta)) < 1e-12)
ok('consistency Delta^2 = (A1C2-A2C1)^2+(B1C2-B2C1)^2', abs(Dd**2-((A1*C2-A2*C1)**2+(B1*C2-B2*C1)**2)) < 1e-12)
t_ok = (A2*C1-A1*C2)/((B1*C2-B2*C1)+Dd)
print(f'half-angle t = {t_ok:.5f}  corrected delta = {2*np.arctan(t_ok)/deg:.4f} deg;  slide form: {np.arctan((A1*C2-A2*C1)/((B1*C2-B2*C1)+Dd))/deg:.4f} deg')
ok('corrected half-angle form gives delta', abs(2*np.arctan(t_ok)-delta) < 1e-12)
ok('slide p.65 form does not', abs(np.arctan((A1*C2-A2*C1)/((B1*C2-B2*C1)+Dd))-delta) > 1e-3)
# sympy: derive t from the two quadratics
A1s,B1s,C1s,A2s,B2s,C2s,t = sp.symbols('A1 B1 C1 A2 B2 C2 t')
q1 = (C1s-A1s)*t**2 + 2*B1s*t + (A1s+C1s); q2 = (C2s-A2s)*t**2 + 2*B2s*t + (A2s+C2s)
lin = sp.expand((C2s-A2s)*q1 - (C1s-A1s)*q2)
tsol = sp.solve(lin, t)[0]
target = (A2s*C1s-A1s*C2s)/((B1s*C2s-B2s*C1s)+(A1s*B2s-A2s*B1s))
ok('sympy: eliminating t^2 gives t = (A2C1-A1C2)/((B1C2-B2C1)+(A1B2-A2B1))', sp.simplify(tsol-target) == 0)
# check the quadratic form of the half-angle substitution
dd = sp.symbols('d')
expr = sp.simplify((A1s*sp.cos(dd)+B1s*sp.sin(dd)+C1s).subs({sp.cos(dd):(1-t**2)/(1+t**2), sp.sin(dd):2*t/(1+t**2)})*(1+t**2) - q1)
ok('sympy: half-angle substitution gives (C-A)t^2+2Bt+(A+C)=0', expr == 0)
# coupler-curve identity (slide p.65 'simplifies to') at many theta1
okc = True
for t1_ in np.linspace(-80,80,41)*deg:
    for m in (1,-1):
        rr = fb(t1_,m)
        if rr is None: continue
        xx,yy = rr['K']
        lhs = 4*a**2*b**2*L0**2*yy**2
        P = b*(xx-L0)*(L1**2-a**2-xx**2-yy**2)-a*xx*((xx-L0)**2+yy**2+b**2-L3**2)
        Q = b*(L1**2-a**2-xx**2-yy**2)-a*((xx-L0)**2+yy**2+b**2-L3**2)
        okc &= abs(lhs-(P**2+yy**2*Q**2)) < 1e-9
ok('slide p.65 quartic holds for every K on the coupler curve (it is the coupler-curve equation)', okc)

# ---------------- 3. assembly range, Grashof ----------------
print('== assembly range ==')
lim = np.degrees(np.arccos((L0**2+L1**2-(L2+L3)**2)/(2*L0*L1)))
print(f'theta1 limit (|AD| = L2+L3): {lim:.4f} deg')
angs = np.arange(-180,180,0.01)
good = [t_ for t_ in angs if fb(t_*deg,1) is not None]
ok(f'closes only for |theta1| <= {lim:.2f}', abs(max(good)-lim) < 0.02 and abs(min(good)+lim) < 0.02, f'{min(good):.2f}..{max(good):.2f}')
print('|AD| at theta1 = 81.52:', np.linalg.norm(fb(81.5*deg,1)['A']-[L0,0]))
print('Grashof: s+l =', min(L0,L1,L2,L3)+max(L0,L1,L2,L3), ' p+q =', sorted([L0,L1,L2,L3])[1]+sorted([L0,L1,L2,L3])[2])
# mock Q3: L0=.8, L1=.3, L2=.7, L3=.5, theta=60
print('-- mock Q3 four-bar: L0=.8 L1=.3 L2=.7 L3=.5, theta1=60')
M = dict(L0=.8,L1=.3,L2=.7,L3=.5)
for mode in (+1,-1):
    rr = fb(60*deg,mode,a=0.0,**M)
    print(f"   mode {mode:+d}: A={np.round(rr['A'],4)} |AD|={rr['AD']:.4f} B={np.round(rr['B'],4)} phi={rr['phi']/deg:.3f} phi1={rr['phi1']/deg:.3f}")
print('   Grashof s+l', 0.3+0.8, ' p+q', 0.5+0.7)
# practice Q: L0=.8 L1=.25 L2=.6 L3=.5, theta=60; and L1=.45
print('-- practice Q: L0=.8 L1=.25 L2=.6 L3=.5 theta1=60')
P = dict(L0=.8,L1=.25,L2=.6,L3=.5)
for mode in (+1,-1):
    rr = fb(60*deg,mode,a=0.0,**P)
    print(f"   mode {mode:+d}: A={np.round(rr['A'],4)} |AD|={rr['AD']:.4f} xa={rr['xa']:.4f} h={rr['h']:.4f} B={np.round(rr['B'],4)} phi1={rr['phi1']/deg:.3f}")
print('   L1=.45: |AD| max =', .8+.45, ' > L2+L3 =', 1.1, ' limit theta1 =', np.degrees(np.arccos((0.8**2+0.45**2-1.1**2)/(2*0.8*0.45))))
print('   L1=.45 Grashof: s+l=', .45+.8, 'p+q=', .5+.6)

# ---------------- 4. slider-crank ----------------
print('== slider-crank: crank L1=0.1, rod L2=0.3 ==')
c1,c2 = 0.1,0.3
for tt in (0,60,90,180):
    th = tt*deg; x_ = c1*np.cos(th)+np.sqrt(c2**2-c1**2*np.sin(th)**2)
    print(f'   theta={tt}: x={x_:.5f}')
th = 60*deg; xs = c1*np.cos(th)+np.sqrt(c2**2-(c1*np.sin(th))**2)
Bp = np.array([c1*np.cos(th), c1*np.sin(th)])
ok('rod length preserved at 60 deg', abs(np.hypot(xs-Bp[0], 0-Bp[1])-c2) < 1e-12)
print('   rod angle at 60: ', np.degrees(np.arcsin(-c1*np.sin(th)/c2)))

# ---------------- 5. five-bar IK ----------------
print('== five-bar: bases (-0.1,0),(0.1,0), proximal 0.3, distal 0.4, tip P=(0.05,0.45) ==')
O1 = np.array([-0.1,0.]); O2 = np.array([0.1,0.]); Lp = 0.3; Ld = 0.4; P5 = np.array([0.05,0.45])
def ik2(O,P,s):
    v = P-O; r_ = np.linalg.norm(v)
    c = (r_**2+Lp**2-Ld**2)/(2*r_*Lp)
    return np.arctan2(v[1],v[0]) + s*np.arccos(c), r_, c
res = {}
for s1 in (+1,-1):
    for s2 in (+1,-1):
        t1_,r1_,c1_ = ik2(O1,P5,s1); t2_,r2_,c2_ = ik2(O2,P5,s2)
        E1 = O1+Lp*np.array([np.cos(t1_),np.sin(t1_)]); E2 = O2+Lp*np.array([np.cos(t2_),np.sin(t2_)])
        ok(f'five-bar branch ({s1:+d},{s2:+d}) closes', abs(np.linalg.norm(P5-E1)-Ld) < 1e-12 and abs(np.linalg.norm(P5-E2)-Ld) < 1e-12)
        print(f'   branch ({s1:+d},{s2:+d}): theta1={np.degrees(t1_):.3f} theta2={np.degrees(t2_):.3f}  E1={np.round(E1,4)} E2={np.round(E2,4)}')
t1_,r1_,c1_ = ik2(O1,P5,+1); t2_,r2_,c2_ = ik2(O2,P5,-1)
print(f'   r1={r1_:.4f} cos={c1_:.4f} r2={r2_:.4f} cos={c2_:.4f}; atan2 base1={np.degrees(np.arctan2(*(P5-O1)[::-1])):.3f} base2={np.degrees(np.arctan2(*(P5-O2)[::-1])):.3f}; acos1={np.degrees(np.arccos(c1_)):.3f} acos2={np.degrees(np.arccos(c2_)):.3f}')
# FK for five-bar: intersection of two circles of radius Ld about E1,E2
E1 = O1+Lp*np.array([np.cos(t1_),np.sin(t1_)]); E2 = O2+Lp*np.array([np.cos(t2_),np.sin(t2_)])
dE = np.linalg.norm(E2-E1); xe = dE/2; he = np.sqrt(Ld**2-xe**2); uu = (E2-E1)/dE; nn = np.array([-uu[1],uu[0]])
pf = [E1+xe*uu+s*he*nn for s in (1,-1)]
print('   FK circle intersections:', [np.round(p,4) for p in pf], '|E1E2|=', round(dE,4))
ok('five-bar FK recovers P (one of two intersections)', min(np.linalg.norm(p-P5) for p in pf) < 1e-12)
# reach limit
print('   max reach from base 1 (Lp+Ld) =', Lp+Ld, '; min |Lp-Ld| =', abs(Lp-Ld))

# ---------------- 6. delta IK ----------------
print('== delta IK (added): Rb=0.20 Rp=0.05 La=0.20 Lr=0.45 ==')
Rb,Rp,La,Lr = 0.20,0.05,0.20,0.45
gam = np.array([0,120,240])*deg
def dik(p, branch=-1):
    out = []
    for g in gam:
        u = np.array([np.cos(g),np.sin(g),0.]); d = p+(Rp-Rb)*u
        du, dz = d@u, d[2]
        K = (d@d+La**2-Lr**2)/(2*La); A, B = du, -dz; rho = np.hypot(A,B)
        if abs(K) > rho: return None
        out.append(np.arctan2(B,A)+branch*np.arccos(K/rho))
    return np.array(out)
def elbow(th,g): return Rb*np.array([np.cos(g),np.sin(g),0])+La*np.array([np.cos(th)*np.cos(g),np.cos(th)*np.sin(g),-np.sin(th)])
p = np.array([0.03,-0.02,-0.38])
for br in (-1,+1):
    th = dik(p,br)
    rods = [np.linalg.norm(elbow(t_,g)-(p+Rp*np.array([np.cos(g),np.sin(g),0]))) for t_,g in zip(th,gam)]
    print(f'   branch {br:+d}: theta={np.round(th/deg,3)} rods={np.round(rods,6)} elbow radius/z leg1 = {np.hypot(*elbow(th[0],gam[0])[:2]):.4f}, {elbow(th[0],gam[0])[2]:.4f}')
th = dik(p,-1)
ok('delta branch -1: rods = Lr', np.allclose([np.linalg.norm(elbow(t_,g)-(p+Rp*np.array([np.cos(g),np.sin(g),0]))) for t_,g in zip(th,gam)], Lr))
ok('delta branch -1: elbow outside base circle and below it', all(np.hypot(*elbow(t_,g)[:2]) > Rb and elbow(t_,g)[2] < 0 for t_,g in zip(th,gam)))
# leg-1 worked numbers: u=(1,0,0)
g = gam[0]; u = np.array([1,0,0.]); d = p+(Rp-Rb)*u
print(f'   leg 1: d={np.round(d,4)} |d|^2={d@d:.5f}  K={(d@d+La**2-Lr**2)/(2*La):.5f}  A={d[0]:.4f} B={-d[2]:.4f} rho={np.hypot(d[0],-d[2]):.5f}  K/rho={((d@d+La**2-Lr**2)/(2*La))/np.hypot(d[0],-d[2]):.5f}  atan2(B,A)={np.degrees(np.arctan2(-d[2],d[0])):.3f}  acos={np.degrees(np.arccos(((d@d+La**2-Lr**2)/(2*La))/np.hypot(d[0],-d[2]))):.3f}')
print('   home p=(0,0,-0.4): theta=', np.round(dik(np.array([0,0,-0.4]))/deg,3))
rg = np.random.default_rng(1); good = True
for _ in range(300):
    pp = np.array([rg.uniform(-.1,.1),rg.uniform(-.1,.1),rg.uniform(-.5,-.3)]); tt = dik(pp)
    if tt is None: continue
    good &= np.allclose([np.linalg.norm(elbow(t_,g)-(pp+Rp*np.array([np.cos(g),np.sin(g),0]))) for t_,g in zip(tt,gam)], Lr)
ok('delta IK closes at 300 random points', good)
# platform parallel to base: parallelogram keeps orientation (rod end offsets equal) -- statement check
# ---------------- 7. 3-RPR IK and Stewart ----------------
print('== 3-RPR IK (practice) ==')
bs = np.array([[0,0],[1.0,0],[0.5,0.9]]); aa = 0.2*np.array([[np.cos(g),np.sin(g)] for g in np.array([0,120,240])*deg])
def rpr(x,y,ph):
    R = np.array([[np.cos(ph),-np.sin(ph)],[np.sin(ph),np.cos(ph)]])
    return [np.linalg.norm(np.array([x,y])+R@aa[i]-bs[i]) for i in range(3)]
print('   (0.5,0.35,10deg):', np.round(rpr(0.5,0.35,10*deg),4), ' unrotated:', np.round(rpr(0.5,0.35,0),4))
print('== Stewart leg (added) ==')
bang = np.array([-10,10,110,130,230,250])*deg; pang = np.array([-50,50,70,170,190,290])*deg
Bp = np.array([[np.cos(t_),np.sin(t_),0] for t_ in bang]); Pp = 0.5*np.array([[np.cos(t_),np.sin(t_),0] for t_ in pang])
l1 = np.array([0,0,1.0])+Pp[0]-Bp[0]
print('   home leg1: l=', np.round(l1,4), ' d=', np.linalg.norm(l1), ' squares', np.round(l1**2,4), ' azimuth', np.degrees(np.arctan2(l1[1],l1[0])), ' elevation', np.degrees(np.arctan2(l1[2],np.hypot(l1[0],l1[1]))))
print('   Bp1,Pp1:', np.round(Bp[0],4), np.round(Pp[0],4))
Rz = lambda t_: np.array([[np.cos(t_),-np.sin(t_),0],[np.sin(t_),np.cos(t_),0],[0,0,1]])
Ry = lambda t_: np.array([[np.cos(t_),0,np.sin(t_)],[0,1,0],[-np.sin(t_),0,np.cos(t_)]])
Rm = Rz(5*deg)@Ry(3*deg); tt = np.array([0.05,-0.02,1.0])
print('   pose t=(.05,-.02,1) Rz5Ry3 legs:', np.round([np.linalg.norm(tt+Rm@Pp[i]-Bp[i]) for i in range(6)],4))
# practice Q: legs 2 and 5 at t=(0,0,0.9), Rz(10)
print('   practice legs 2,5:', np.round([np.linalg.norm(np.array([0,0,.9])+Rz(10*deg)@Pp[i]-Bp[i]) for i in (1,4)],4))
# one leg full: leg direction in P frame
n_hat = (tt+Rm@Pp[0]-Bp[0]); d1 = np.linalg.norm(n_hat); n_hat /= d1
print('   leg1 posed: l=', np.round(tt+Rm@Pp[0]-Bp[0],4), 'd=', round(d1,4), ' base az/el:', np.degrees(np.arctan2(n_hat[1],n_hat[0])), np.degrees(np.arcsin(n_hat[2])), ' n in P:', np.round(Rm.T@n_hat,4))

# ---------------- 8. workspace of two-leg planar robot ----------------
print('== workspace: bases (0,0),(0.6,0), legs 0.3..0.7 ==')
b1 = np.array([0,0.]); b2 = np.array([0.6,0.]); rmin,rmax = 0.3,0.7
def inws(P): 
    r1_=np.linalg.norm(P-b1); r2_=np.linalg.norm(P-b2); return rmin<=r1_<=rmax and rmin<=r2_<=rmax
print('   apex on axis (0.3, sqrt(0.7^2-0.3^2)) =', np.sqrt(0.7**2-0.3**2), inws(np.array([0.3,np.sqrt(0.7**2-0.09)-1e-9])))
print('   (0.3,0.4): r1,r2 =', np.hypot(.3,.4), np.hypot(.3,.4), inws(np.array([.3,.4])))
print('   (0.3,0.1) in ws?', inws(np.array([.3,.1])), ' r =', np.hypot(.3,.1))
# lowest point on axis: r >= 0.3
print('   lowest on axis: y=0 r=0.3 ->', inws(np.array([.3,0.0])))
# area by grid
xs_ = np.linspace(-0.8,1.4,2201); ys_ = np.linspace(-0.8,0.8,1601)
X,Y = np.meshgrid(xs_,ys_)
r1g = np.hypot(X,Y); r2g = np.hypot(X-0.6,Y)
m = (r1g>=rmin)&(r1g<=rmax)&(r2g>=rmin)&(r2g<=rmax)
print('   upper-half area (y>=0) =', m[Y>=0].sum()*(xs_[1]-xs_[0])*(ys_[1]-ys_[0]))
mu = m&(Y>=0)
print('   upper half x range', X[mu].min(), X[mu].max(), ' y range', Y[mu].min(), Y[mu].max())
# corner points: intersection of outer circle of leg1 with inner circle of leg2 etc
def circ_int(c1,r1,c2,r2):
    d = np.linalg.norm(c2-c1); xa = (r1**2-r2**2+d**2)/(2*d); h = np.sqrt(r1**2-xa**2); u = (c2-c1)/d; n = np.array([-u[1],u[0]])
    return c1+xa*u+h*n, c1+xa*u-h*n
print('   outer1 x inner2:', circ_int(b1,0.7,b2,0.3), '\n   inner1 x outer2:', circ_int(b1,0.3,b2,0.7), '\n   outer1 x outer2:', circ_int(b1,0.7,b2,0.7), '\n   inner x inner:', circ_int(b1,0.3,b2,0.3))
print('== extra numbers for the text ==')
r = fb(70*deg,1); u = (np.array([L0,0])-r['A'])/r['AD']; n = np.array([-u[1],u[0]])
print('   u =', np.round(u,4), ' n =', np.round(n,4), ' A=', r['A'])
print('   |AD|^2 =', r['AD']**2, ' L2^2-L3^2 =', L2**2-L3**2, ' xa num =', L2**2-L3**2+r['AD']**2, ' den =', 2*r['AD'], ' h^2 =', L2**2-r['xa']**2)
print('   mode -1: phi2 mod 360 =', np.mod(fb(70*deg,-1)['phi']-70*deg,2*np.pi)/deg)
th_d = dik(np.array([0.03,-0.02,-0.38]),-1)
d1 = np.array([0.03,-0.02,-0.38])+(Rp-Rb)*np.array([1,0,0.])
Kd = (d1@d1+La**2-Lr**2)/(2*La); rho_d = np.hypot(d1[0],-d1[2])
print('   delta leg1 precise: atan2 %.4f acos %.4f theta %.4f ; other root %.4f' % (np.degrees(np.arctan2(-d1[2],d1[0])), np.degrees(np.arccos(Kd/rho_d)), th_d[0]/deg, np.degrees(np.arctan2(-d1[2],d1[0])+np.arccos(Kd/rho_d))))
for i,g in enumerate(gam):
    u_ = np.array([np.cos(g),np.sin(g),0]); d_ = np.array([0.03,-0.02,-0.38])+(Rp-Rb)*u_
    print('   leg',i+1,'d=',np.round(d_,4),'|d|^2=%.5f'%(d_@d_),'K=%.5f A=%.4f B=%.4f rho=%.5f  atan2=%.3f acos=%.3f theta=%.3f'%((d_@d_+La**2-Lr**2)/(2*La), d_@u_, -d_[2], np.hypot(d_@u_,d_[2]), np.degrees(np.arctan2(-d_[2],d_@u_)), np.degrees(np.arccos(((d_@d_+La**2-Lr**2)/(2*La))/np.hypot(d_@u_,d_[2]))), th_d[i]/deg))
nP = Rm.T@n_hat
print('   stewart leg1 P-frame az/el of n:', np.degrees(np.arctan2(nP[1],nP[0])), np.degrees(np.arcsin(nP[2])))
print('   stewart leg1 n_hat world:', np.round(n_hat,4))
# three-RPR: show p_i and vectors
R10 = np.array([[np.cos(10*deg),-np.sin(10*deg)],[np.sin(10*deg),np.cos(10*deg)]])
for i in range(3):
    pi_ = np.array([0.5,0.35])+R10@aa[i]; print('   RPR leg',i+1,'p=',np.round(pi_,4),'v=',np.round(pi_-bs[i],4),'len=%.4f'%np.linalg.norm(pi_-bs[i]))
print('   five-bar: P=(.05,.45) E1,E2 as above; mobility 2')
print('   four-bar coupler B-A length check L2:', np.linalg.norm(fb(70*deg,1)['B']-fb(70*deg,1)['A']))
print('ALL W4 v2 CHECKS PASSED')
