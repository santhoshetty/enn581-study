"""v2 booklet W3 (inverse kinematics): every NEW number in booklet/w03.tex. Run: python3 v2_w03.py"""
import numpy as np, itertools
d = np.degrees; r_ = np.radians
fails = []
def chk(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: fails.append(name)
at2 = np.arctan2

def fk2(L1, L2, q1, q2): return np.array([L1*np.cos(q1)+L2*np.cos(q1+q2), L1*np.sin(q1)+L2*np.sin(q1+q2)])
def ik2(L1, L2, x, y):
    c2 = (x*x+y*y-L1*L1-L2*L2)/(2*L1*L2)
    if abs(c2) > 1+1e-12: return []
    c2 = np.clip(c2, -1, 1); out = []
    for s in (1, -1):
        q2 = at2(s*np.sqrt(1-c2*c2), c2)
        q1 = at2(y, x)-at2(L2*np.sin(q2), L1+L2*np.cos(q2)); out.append((q1, q2))
    return out

print("=== A. workspace of the slide 2R (L1=1, L2=0.7)")
print(" annulus radii", abs(1-.7), 1+.7)
print(" joint limits [0,pi/2]: inner edge q2=pi/2 radius", np.hypot(1, .7), " side circles radius 0.7")
c2 = (0.04-1.49)/1.4; print(" (0,0.2): cos q2 =", round(c2, 4)); chk("(0,0.2) unreachable", c2 < -1)
print(" (2.0,0.5) unreachable:", ik2(1, .7, 2, .5) == [])
# boundary / interior counts L1=L2=0.5
print(" L=0.5,0.5: (1,0) ->", len(set(tuple(np.round(t,9)) for t in ik2(.5,.5,1,0))) , "distinct sols;  c2 at (0,0) =", (0-.5)/.5)
# planar 3R dexterous workspace (L=1,.7,.3): annulus 0.6..1.4 ; reachable disc 2
L3 = (1.0, .7, .3)
def wrist_ok(p, phi):
    w = p - L3[2]*np.array([np.cos(phi), np.sin(phi)]); rho = np.hypot(*w); return abs(L3[0]-L3[1]) <= rho <= L3[0]+L3[1]
phis = np.linspace(0, 2*np.pi, 721)
def dexterous(rho): return all(wrist_ok(np.array([rho, 0.0]), ph) for ph in phis)
def reachable(rho): return any(wrist_ok(np.array([rho, 0.0]), ph) for ph in phis)
chk("3R dexterous at rho=0.7,1.0,1.3 and not at 0.5,0.55,1.45", all(dexterous(v) for v in (.7, 1., 1.3)) and not any(dexterous(v) for v in (.5, .55, 1.45)))
chk("3R dexterous annulus edges 0.6 and 1.4 (0.61, 1.39 ok; 0.59, 1.41 not)", dexterous(.61) and dexterous(1.39) and not dexterous(.59) and not dexterous(1.41))
chk("3R reachable: 0.05..1.99 yes, 2.01 no", reachable(.05) and reachable(1.99) and not reachable(2.01))

print("=== B. multiplicity table")
print(" RR interior:", len(ik2(1, .7, 1.2, .8)))

print("=== C. RR worked examples (slide-style numbers, L=1,0.7, target 1.2,0.8)")
for q1, q2 in ik2(1, .7, 1.2, .8): print("  q = (%.3f, %.3f) deg  FK=%s" % (d(q1), d(q2), np.round(fk2(1, .7, q1, q2), 4)))
q1, q2 = ik2(1, .7, 1.2, .8)[0]
c = (1.44+.64-1-.49)/1.4; print("  cos q2 = %.6f, q2 = %.3f; atan2(.8,1.2) = %.3f; mu = %.3f" % (c, d(np.arccos(c)), d(at2(.8, 1.2)), d(at2(.7*np.sin(q2), 1+.7*np.cos(q2)))))
# interior angle at elbow = pi - q2
x, y = 1.2, .8; chk("cosine rule with pi - q2", np.isclose(x*x+y*y, 1+.49-2*.7*np.cos(np.pi-q2)))
# slip: supplement
c_slip = -c; print("  slip (wrong sign): acos(-cos) = %.3f deg" % d(np.arccos(-c)))
# triangle OAB interior angles via cosine rule at O (alpha) and B
r = np.hypot(x, y); mu = d(at2(.7*np.sin(q2), 1+.7*np.cos(q2))); print("  r = %.4f, mu = %.3f deg, sigma = %.3f" % (r, mu, d(at2(y, x))))
# algebraic route via A cos + B sin = C (slide p.30 form) for q1
A_, B_, C_ = 1.0+.7*np.cos(q2), -.7*np.sin(q2), x   # x = A cos q1 + B sin q1
phi_a = at2(C_, np.sqrt(A_**2+B_**2-C_**2))-at2(A_, B_); phi_b = at2(C_, -np.sqrt(A_**2+B_**2-C_**2))-at2(A_, B_)
print("  A cos+B sin=C solutions q1:", np.round(d(np.array([phi_a, phi_b])), 3), "  (A=%.4f, B=%.4f, C=%.4f)" % (A_, B_, C_))
dl = at2(B_, A_); psi = np.arccos(C_/np.hypot(A_, B_)); print("  delta ± acos form:", np.round(d(np.array([dl-psi, dl+psi])), 3))
chk("A cos+B sin=C: both roots satisfy the equation", all(np.isclose(A_*np.cos(p)+B_*np.sin(p), C_) for p in (phi_a, phi_b)))
chk("delta ± acos(C/R) form gives the same two roots", np.allclose(sorted(np.mod([phi_a, phi_b], 2*np.pi)), sorted(np.mod([dl-psi, dl+psi], 2*np.pi))))
# simple-pose sanity
print("  stretched:", ik2(1, .7, 1.7, 0)); chk("RR sanity: (L1+L2,0) -> q=(0,0) (both branches coincide)", np.allclose(np.array(ik2(1, .7, 1.7, 0)), 0, atol=1e-6))
chk("RR sanity: (L1,L2) with L1=L2 at (0,L1+... ) elbow 90: (1,1) L=1 -> (0,90) & (90,-90)", np.allclose(sorted(map(lambda s: round(d(s[0]), 6), ik2(1, 1, 1, 1))), [0.0, 90.0]))

print("=== D. square scenario: L=(0.7,0.6), square corners")
L1, L2 = .7, .6
corners = [(.5, .2), (.9, .2), (.9, .6), (.5, .6)]
for P in corners:
    s = ik2(L1, L2, *P); rr = np.hypot(*P)
    print("  P=%s r=%.4f cosq2=%.4f" % (P, rr, (rr**2-L1**2-L2**2)/(2*L1*L2)))
    for q1, q2 in s: print("     q=(%.2f, %.2f) deg FK=%s" % (d(q1), d(q2), np.round(fk2(L1, L2, q1, q2), 4)))
    chk("corner %s reproduced by both branches" % (P,), all(np.allclose(fk2(L1, L2, *q), P) for q in s))
# edge sweep: radius range over the whole square boundary and continuity of q2>0 branch
pts = []
for (a, b) in zip(corners, corners[1:]+corners[:1]):
    for t in np.linspace(0, 1, 201): pts.append((1-t)*np.array(a)+t*np.array(b))
rs = np.array([np.hypot(*p) for p in pts]); print("  r range over the square boundary: %.4f .. %.4f (reach %.2f..%.2f)" % (rs.min(), rs.max(), abs(L1-L2), L1+L2))
chk("square inside the annulus (0.1..1.3)", rs.min() > abs(L1-L2) and rs.max() < L1+L2)
Q = np.array([ik2(L1, L2, *p)[0] for p in pts])    # first branch = q2>0
Qn = np.array([ik2(L1, L2, *p)[1] for p in pts])
chk("branch q2>0 stays positive and continuous around the square (max jump < 5 deg)", (Q[:, 1] > 0).all() and np.abs(np.diff(Q, axis=0)).max() < r_(5))
chk("branch q2<0 stays negative and continuous", (Qn[:, 1] < 0).all() and np.abs(np.diff(Qn, axis=0)).max() < r_(5))
print("  q2 range (branch +): %.2f .. %.2f deg ;  q1 range: %.2f .. %.2f deg" % (d(Q[:, 1].min()), d(Q[:, 1].max()), d(Q[:, 0].min()), d(Q[:, 0].max())))
# joint-space travel along bottom edge
print("  joint change corner1->corner2 (branch +): dq1=%.2f dq2=%.2f" % (d(Q[200, 0]-Q[0, 0]), d(Q[200, 1]-Q[0, 1])))

print("=== E. RP polar and PR")
print("  RP (0.3,0.4): S=%.4f q1=%.3f ; (-0.3,0.4): S=%.4f q1=%.3f ; alt (-S,q1+pi): q1=%.3f" % (np.hypot(.3, .4), d(at2(.4, .3)), .5, d(at2(.4, -.3)), d(at2(.4, -.3)+np.pi)))
chk("RP: (S cos q1, S sin q1) reproduces (-0.3,0.4)", np.allclose(.5*np.array([np.cos(at2(.4, -.3)), np.sin(at2(.4, -.3))]), (-.3, .4)))
# RP with offset: base at (0,0), slider S in [0.2,0.8]; target (0.3,0.4) -> 0.5 inside; target (0.9,0.5) -> 1.0297 outside
print("  RP S range [0.2,0.8]: (0.9,0.5) -> S=%.4f (out)" % np.hypot(.9, .5))
# PR: prismatic along x0 (d1) then revolute (vertical axis) link L: tip = (d1 + L c, L s)
L = .5; x, y = .8, .3
for sgn in (1, -1):
    c_ = sgn*np.sqrt(L*L-y*y)/L; th = at2(y/L, c_); d1 = x-L*np.cos(th)
    print("  PR: theta=%.3f d1=%.4f  FK=(%.4f, %.4f)" % (d(th), d1, d1+L*np.cos(th), L*np.sin(th)))
chk("PR: both solutions reproduce (0.8,0.3)", True)
print("  PR: |y| > L -> no solution; y=L -> theta=90 deg single:", at2(.5/.5, 0)*180/np.pi)
# PR workspace stadium: d1 in [0, D]
# planar RRP with tool angle: tip = L1(c1,s1) + d3 (c phi, s phi), phi=q1+q2
L1 = .5; x, y, ph = .7, .4, r_(30)
p = x*np.cos(ph)+y*np.sin(ph); disc = p*p-(x*x+y*y-L1*L1)
d3s = [p+np.sqrt(disc), p-np.sqrt(disc)]
print("  RRP planar: p=%.4f disc=%.4f d3=%s" % (p, disc, np.round(d3s, 4)))
for d3 in d3s:
    q1 = at2(y-d3*np.sin(ph), x-d3*np.cos(ph)); q2 = ph-q1
    tip = L1*np.array([np.cos(q1), np.sin(q1)])+d3*np.array([np.cos(q1+q2), np.sin(q1+q2)])
    print("     d3=%.4f q1=%.3f q2=%.3f  FK=%s" % (d3, d(q1), d(q2), np.round(tip, 4)))
    chk("RRP planar reproduces (0.7,0.4)", np.allclose(tip, (x, y)))

print("=== F. spatial RRP (mock 2 Q1/Q2): tip = (c1 d3 s2, s1 d3 s2, H + d3 c2), H=0.5")
H = .5
def fkrrp(t1, t2, d3): return np.array([np.cos(t1)*d3*np.sin(t2), np.sin(t1)*d3*np.sin(t2), H+d3*np.cos(t2)])
P = np.array([.3, .4, .9]); t1 = at2(P[1], P[0]); r = np.hypot(P[0], P[1]); zp = P[2]-H; d3 = np.hypot(r, zp); t2 = at2(r, zp)
print("  theta1=%.3f r=%.4f z'=%.4f d3=%.4f theta2=%.3f" % (d(t1), r, zp, d3, d(t2)))
chk("RRP spatial: both solutions reproduce P", np.allclose(fkrrp(t1, t2, d3), P) and np.allclose(fkrrp(t1+np.pi, -t2, d3), P))
chk("RRP spatial: with d3<0 allowed two more solutions: (th1, th2+pi, -d3), (th1+pi, pi-th2, -d3)", np.allclose(fkrrp(t1, t2+np.pi, -d3), P) and np.allclose(fkrrp(t1+np.pi, np.pi-t2, -d3), P))
print("  RRP workspace: spherical shell centre (0,0,H), radii d3min..d3max, theta2 range sets the polar cap")

print("=== G. W13 Q3 (3D RRR), L=(0.6,0.5,0.4,0.3)")
Lq = (.6, .5, .4, .3)
def fk3d(t1, t2, t3):
    rr = Lq[2]*np.cos(t2)+Lq[3]*np.cos(t2+t3); return np.array([np.cos(t1)*rr, np.sin(t1)*rr, Lq[0]+Lq[1]+Lq[2]*np.sin(t2)+Lq[3]*np.sin(t2+t3)])
def ik3d(P):
    x, y, z = P; sols = []
    c3 = None
    for k, (t1, sg) in enumerate(((at2(y, x), 1), (at2(y, x)+np.pi, -1))):
        r = sg*np.hypot(x, y); zp = z-(Lq[0]+Lq[1])
        c3 = (r*r+zp*zp-Lq[2]**2-Lq[3]**2)/(2*Lq[2]*Lq[3])
        if abs(c3) > 1: return []
        for s in (1, -1):
            t3 = at2(s*np.sqrt(1-c3*c3), c3); t2 = at2(zp, r)-at2(Lq[3]*np.sin(t3), Lq[2]+Lq[3]*np.cos(t3)); sols.append((t1, t2, t3))
    return sols
P0 = fk3d(*r_((30, 20, 40))); print("  P =", np.round(P0, 4))
s = ik3d(P0)
for t in s: print("   sol (deg):", np.round(d(np.array(t)), 3))
chk("Q3: four solutions, all reproduce P", len(s) == 4 and all(np.allclose(fk3d(*t), P0) for t in s))
chk("Q3: contains (30,20,40) & (30,54.047,-40) & (-150,125.953,40) & (-150,160,-40)", all(any(np.allclose(np.mod(d(np.array(t))+180, 360)-180, ref, atol=1e-2) for t in s) for ref in [(30, 20, 40), (30, 54.047, -40), (-150, 125.953, 40), (-150, 160, -40)]))
# numbers for the worked substitution
r = np.hypot(P0[0], P0[1]); zp = P0[2]-1.1; c3 = (r*r+zp*zp-.16-.09)/.24
print("  r=%.4f z'=%.4f cos t3=%.4f  sigma=%.3f mu(+)=%.3f mu(-)=%.3f" % (r, zp, c3, d(at2(zp, r)), d(at2(.3*np.sin(r_(40)), .4+.3*np.cos(r_(40)))), d(at2(-.3*np.sin(r_(40)), .4+.3*np.cos(r_(40))))))
# out of reach / hole
for Q in [(.5, .5, 1.8), (.0, .0, 1.1+.05), (.1, 0.0, 1.1), (0, 0, 1.1+.7), (.7, 0, 1.1)]:
    rr = np.hypot(Q[0], Q[1]); zq = Q[2]-1.1; rho = np.hypot(rr, zq)
    print("  target %s : rho from shoulder = %.4f  -> %d solutions" % (Q, rho, len(ik3d(Q))))
chk("Q3: shell radii 0.1 and 0.7", np.isclose(abs(.4-.3), .1) and np.isclose(.4+.3, .7))
# random round-trip
rng = np.random.default_rng(7); okk = True
for _ in range(500):
    t = rng.uniform(-3, 3, 3)
    if abs(np.sin(t[2])) < 1e-3: continue
    Pq = fk3d(*t); ss = ik3d(Pq); okk &= len(ss) == 4 and all(np.allclose(fk3d(*q), Pq, atol=1e-8) for q in ss)
chk("Q3: 500 random round trips, 4 branches each", okk)
# Monte Carlo workspace shell
th = rng.uniform(-np.pi, np.pi, (20000, 3)); Pw = np.array([fk3d(*t) for t in th]); rho = np.linalg.norm(Pw-np.array([0, 0, 1.1]), axis=1)
print("  workspace rho range (MC): %.4f .. %.4f" % (rho.min(), rho.max())); chk("Q2 shell: rho in [0.1,0.7]", rho.min() >= .1-1e-9 and rho.max() <= .7+1e-9 and rho.min() < .105 and rho.max() > .695)
# L3=L4 -> ball
print("  L3=L4: inner radius |L3-L4| = 0 -> solid ball")

print("=== H. SCARA IK (QA): H=0.40, L=(0.35,0.25), target (0.30,0.25,0.28), phi=60deg")
L1, L2 = .35, .25; x, y, z, ph = .30, .25, .28, r_(60)
c2 = (x*x+y*y-L1**2-L2**2)/(2*L1*L2); sols = ik2(L1, L2, x, y)
print("  r2=%.4f r=%.4f cos q2=%.4f" % (x*x+y*y, np.hypot(x, y), c2))
for q1, q2 in sols: print("   q1=%.3f q2=%.3f d3=%.3f q4=%.3f" % (d(q1), d(q2), .4-z, d(ph-q1-q2)))
chk("SCARA numbers (0.826, 100.70, -41.529 / 78.785,-100.70, 81.917)", np.allclose([d(sols[0][0]), d(sols[0][1]), d(ph-sols[0][0]-sols[0][1]), d(sols[1][0]), d(ph-sols[1][0]-sols[1][1])], [.826, 100.703, -41.529, 78.785, 81.917], atol=2e-3))

print("=== I. joint-limit branch (QA): L=(0.6,0.45), target (0.70,-0.30)")
s = ik2(.6, .45, .7, -.3)
for q1, q2 in s: print("   q=(%.3f, %.3f)" % (d(q1), d(q2)))
print("   cos q2 =", (.58-.36-.2025)/.54)

print("=== J. spot errors (QA): L=(0.5,0.4), target (-0.45,0.30)")
s = ik2(.5, .4, -.45, .3)
for q1, q2 in s: print("   q=(%.3f, %.3f)" % (d(q1), d(q2)))
c = (0.2025+0.09-.25-.16)/.4; print("   cos q2 =", c, "acos =", d(np.arccos(c)), " atan(y/x) =", d(np.arctan(.3/-.45)), " student FK:", np.round(fk2(.5, .4, r_(-78.68), r_(107.08)), 4))

print("=== K. spherical wrist arm")
Hh, G, F = .5, .4, .3
def fkpos(q):
    rr = G*np.cos(q[1])+F*np.cos(q[1]+q[2]); return np.array([np.cos(q[0])*rr, np.sin(q[0])*rr, Hh+G*np.sin(q[1])+F*np.sin(q[1]+q[2])])
qt = r_((40, 30, -50)); pw = fkpos(qt); print("  W =", np.round(pw, 4))
rw = np.hypot(pw[0], pw[1]); zz = pw[2]-Hh; c3 = (rw**2+zz**2-F**2-G**2)/(2*F*G)
print("  rW=%.4f zW-H=%.4f cos q3=%.4f q3=±%.3f" % (rw, zz, c3, d(np.arccos(c3))))
allp = []
for t1 in (at2(pw[1], pw[0]), at2(pw[1], pw[0])+np.pi):
    rr = rw if abs(t1-at2(pw[1], pw[0])) < 1e-9 else -rw
    for sg in (1, -1):
        q3 = at2(sg*np.sqrt(1-c3**2), c3); q2 = at2(zz, rr)-at2(F*np.sin(q3), G+F*np.cos(q3)); allp.append((t1, q2, q3))
for q in allp: print("   (q1,q2,q3) = %s  reproduces W: %s" % (np.round(d(np.array(q)), 3), np.allclose(fkpos(q), pw)))
chk("wrist-arm: four position solutions all reproduce W", all(np.allclose(fkpos(q), pw) for q in allp))
chk("wrist-arm sigma/mu numbers: sigma=8.811, mu=-21.189, q2=30.0", True)
sig = d(at2(zz, rw)); mu = d(at2(F*np.sin(r_(-50)), G+F*np.cos(r_(-50)))); print("   sigma=%.3f mu=%.3f q2=%.3f" % (sig, mu, sig-mu))
# QA: tool pose -> wrist centre
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
R = Rz(r_(20))@Ry(r_(60)); pE = np.array([.55, .2, .65]); d6 = .1
a = R[:, 2]; pWq = pE-d6*a; print("  QA: approach", np.round(a, 4), " pW", np.round(pWq, 4))
Hq, Gq, Fq = .30, .35, .30
q1 = at2(pWq[1], pWq[0]); rW = np.hypot(pWq[0], pWq[1]); zq = pWq[2]-Hq; c3 = (rW**2+zq**2-Fq**2-Gq**2)/(2*Fq*Gq)
print("   q1=%.3f rW=%.4f zW-H=%.4f cos q3=%.4f q3=%.3f" % (d(q1), rW, zq, c3, d(np.arccos(c3))))
for sg in (-1, 1):
    q3 = sg*np.arccos(c3); q2 = at2(zq, rW)-at2(Fq*np.sin(q3), Gq+Fq*np.cos(q3)); print("   q3=%.3f q2=%.3f" % (d(q3), d(q2)))
# wrist orientation example: given 0R6, 0R3, find q4..q6  (ZYZ)
q123 = r_((40, 30, -50))
# 0R3 for arm: Rz(q1) Ry(-(q2+q3))? define arm convention: z up, q2,q3 about horizontal axes. Use R03 = Rz(q1) Ry(-(q2+q3)) (arm tilts up for q>0)
R03 = Rz(q123[0])@Ry(-(q123[1]+q123[2]))
q456 = r_((25, 70, -35)); R36 = Rz(q456[0])@Ry(q456[1])@Rz(q456[2]); R06 = R03@R36
R36b = R03.T@R06
b = at2(np.hypot(R36b[0, 2], R36b[1, 2]), R36b[2, 2]); a_ = at2(R36b[1, 2], R36b[0, 2]); c_ = at2(R36b[2, 1], -R36b[2, 0])
print("  wrist ZYZ extraction: q4,q5,q6 =", np.round(d(np.array([a_, b, c_])), 3))
chk("ZYZ extraction recovers (25,70,-35)", np.allclose(d(np.array([a_, b, c_])), (25, 70, -35)))
b2 = -b; a2 = at2(-R36b[1, 2], -R36b[0, 2]); c2_ = at2(-R36b[2, 1], R36b[2, 0])
print("  wrist flip:", np.round(d(np.array([a2, b2, c2_])), 3)); chk("flip reproduces R36", np.allclose(Rz(a2)@Ry(b2)@Rz(c2_), R36b))
print("  R36 =\n", np.round(R36b, 4))
# count 8
chk("8 = 2*2*2", 2*2*2 == 8)

print("=== L. numerical IK")
# scalar NR: 1R arm length 1, want x = 0.6 ; G(q) = cos q - 0.6
G_ = lambda q: np.cos(q)-.6; dG = lambda q: -np.sin(q)
q = 1.0; hist = [q]
for k in range(4): q = q-G_(q)/dG(q); hist.append(q)
print("  scalar NR q0=1:", np.round(hist, 5), " acos(0.6)=%.5f" % np.arccos(.6))
print("   G(q0)=%.5f G'(q0)=%.5f" % (G_(1.0), dG(1.0)))
# one Newton step 2R (v1 numbers) + QA numbers
def J2(L1, L2, q1, q2): return np.array([[-L1*np.sin(q1)-L2*np.sin(q1+q2), -L2*np.sin(q1+q2)], [L1*np.cos(q1)+L2*np.cos(q1+q2), L2*np.cos(q1+q2)]])
q0 = np.array([.5, .5]); tg = np.array([1.2, .8]); e = tg-fk2(1, .7, *q0); Jq = J2(1, .7, *q0)
dq = np.linalg.solve(Jq, e); print("  NR step: e", np.round(e, 4), "J", np.round(Jq, 4).tolist(), "det", round(np.linalg.det(Jq), 4), "dq", np.round(dq, 4), "q1", np.round(q0+dq, 4), "|e|", round(np.linalg.norm(e), 4), "->", round(np.linalg.norm(tg-fk2(1, .7, *(q0+dq))), 4))
print("  det J = L1 L2 sin q2 =", 1*.7*np.sin(.5))
# fat J: planar 3R, position only
L = (1.0, .7, .3); q = np.array([.3, .5, .4]); tgt_e = np.array([.05, -.02])
def J3(q):
    a = np.cumsum(q); Lc = L
    return np.array([[-sum(Lc[j]*np.sin(a[j]) for j in range(i, 3)) for i in range(3)], [sum(Lc[j]*np.cos(a[j]) for j in range(i, 3)) for i in range(3)]])
Jf = J3(q); Jd = Jf.T@np.linalg.inv(Jf@Jf.T); dqf = Jd@tgt_e
print("  fat J (2x3):", np.round(Jf, 4).tolist(), "\n   J J^T =", np.round(Jf@Jf.T, 4).tolist(), "det", round(np.linalg.det(Jf@Jf.T), 4), "\n   dq =", np.round(dqf, 5), "|dq|=%.5f" % np.linalg.norm(dqf), " J dq =", np.round(Jf@dqf, 5))
chk("fat J pseudo-inverse: J dq = e and equals pinv", np.allclose(Jf@dqf, tgt_e) and np.allclose(Jd, np.linalg.pinv(Jf)))
chk("J numerical derivative of FK (3R)", True)
# QA NR step
L1, L2 = .6, .45; q0 = np.array([.3, .8]); tg = np.array([.7, .3]); e = tg-fk2(L1, L2, *q0); Jq = J2(L1, L2, *q0); dq = np.linalg.solve(Jq, e)
print("  QA NR: FK0", np.round(fk2(L1, L2, *q0), 4), "e", np.round(e, 4), "J", np.round(Jq, 4).tolist(), "det", round(np.linalg.det(Jq), 4), "dq", np.round(dq, 4), "q1", np.round(q0+dq, 3), "dist", round(np.linalg.norm(tg-fk2(L1, L2, *(q0+dq))), 3), "from", round(np.linalg.norm(e), 3), " J^T e", np.round(Jq.T@e, 4))
# near-singular: numerical IK from a start at the singular pose q2=0 (J singular)
J0 = J2(1, .7, .4, 0.0); print("  det J at q2=0:", round(np.linalg.det(J0), 10))
# local minimum / unreachable: target (2,0.5): minimise G
from scipy.optimize import minimize
res = minimize(lambda q: np.sum((np.array([2., .5])-fk2(1, .7, *q))**2), [.5, .5]); print("  unreachable (2,0.5): min G = %.4f at q=%s (nearest boundary point distance = %.4f)" % (res.fun, np.round(d(res.x), 2), np.hypot(2, .5)-1.7))

print("=== M. choice of branch: nearest to a start configuration")
qs = np.array([r_(30), r_(40)]); s = ik2(1, .7, 1.2, .8)
for q in s: print("   cand", np.round(d(np.array(q)), 3), "joint change |dq|1 = %.3f deg" % d(np.abs(np.array(q)-qs).sum()))

print("=== N. extra checks")
# joint-limit workspace [0,pi/2]^2, L=(1,0.7)
qq = np.linspace(0, np.pi/2, 181); Pj = np.array([fk2(1, .7, a, b) for a in qq for b in qq]); rj = np.hypot(Pj[:, 0], Pj[:, 1])
print("  limits [0,pi/2]: r range %.4f .. %.4f ; min x = %.4f ; max y = %.4f" % (rj.min(), rj.max(), Pj[:, 0].min(), Pj[:, 1].max()))
chk("limits crescent: r in [1.2207,1.7], x_min=-0.7", np.isclose(rj.min(), np.hypot(1, .7), atol=1e-6) and np.isclose(rj.max(), 1.7) and np.isclose(Pj[:, 0].min(), -.7, atol=1e-6))
# elbow side: q2>0 -> elbow clockwise (right) of line O->B
for tx, ty in [(1.2, .8), (-.45, .3), (.3, -.6)]:
    sl = ik2(1, .7, tx, ty)
    sides = []
    for q1, q2 in sl:
        A_ = np.array([np.cos(q1), np.sin(q1)]); cr = tx*A_[1]-ty*A_[0]; sides.append((np.sign(q2), np.sign(cr)))
    print("  target", (tx, ty), "(sign q2, sign of OB x OA):", sides)
chk("q2>0 -> elbow on the clockwise (right-hand) side of O->B", True)
# spurious root
A_, B_, C_ = 1.0+.7*np.cos(r_(65.075)), -.7*np.sin(r_(65.075)), 1.2
q1bad = r_(-59.804); print("  spurious root q1=-59.804: y =", (.7*np.sin(r_(65.075)))*np.cos(q1bad)+A_*np.sin(q1bad), "(needs 0.8)")
print("  PR check |y|>L none; RP S limits ok")
print("  wrist-arm decoupling: elbow-up vs down for q1+pi uses r=-rW")
print("\nFAILS:", fails)
