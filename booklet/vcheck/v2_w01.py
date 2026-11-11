"""v2 booklet W1 (mechanisms, DoF, G-K, workspace): every number in booklet/w01.tex. Run: python3 v2_w01.py"""
import numpy as np, sympy as sp
fails = []
def chk(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + ("  -> " + str(extra) if extra != "" else ""))
    if not cond: fails.append(name)

def gk2(D, N, F):            # form 2: M = D(N-J-1) + sum F
    J = len(F); return D*(N-J-1) + sum(F)
def gk1(D, N, F):            # form 1: M = D(N-1) - sum (D-F_i)  (= D(N-1) - sum_i (D-i) P_i)
    return D*(N-1) - sum(D-f for f in F)

def count(name, D, N, F, want):
    m2, m1 = gk2(D, N, F), gk1(D, N, F)
    chk(f"{name}: D={D} N={N} J={len(F)} sumF={sum(F)} -> M={m2} (form1 {m1})", m2 == want and m1 == want)

print("=== A. rigid-body DoF and joints")
# 3 non-collinear points in 3D: 9 coordinates - 3 fixed distances = 6 ; 2 points in 2D: 4 - 1 = 3
chk("3D body: 3x3 - 3 = 6", 3*3-3 == 6); chk("2D body: 2x2 - 1 = 3", 2*2-1 == 3)
# revolute joint between two planar bodies removes 2: pin point fixed in both -> 2 equations
x1, y1, t1, x2, y2, t2, a, b, c, d = sp.symbols('x1 y1 t1 x2 y2 t2 a b c d', real=True)
# body1 pin at (a,b) in body 1 frame; body 2 pin at (c,d) in body 2 frame
R = lambda t: sp.Matrix([[sp.cos(t), -sp.sin(t)], [sp.sin(t), sp.cos(t)]])
eq = sp.Matrix([x1, y1]) + R(t1)*sp.Matrix([a, b]) - sp.Matrix([x2, y2]) - R(t2)*sp.Matrix([c, d])
Jc = eq.jacobian([x1, y1, t1, x2, y2, t2]); vals = {t1: .4, t2: 1.1, a: .5, b: 0, c: -.5, d: 0}
rk = np.linalg.matrix_rank(np.array(Jc.subs(vals).evalf().tolist(), dtype=float))
chk("planar pin: 2 independent equations on 6 coordinates -> 6-2 = 4 DoF", rk == 2 and 6-rk == 4)
for nm, D, F in (("R", 3, 1), ("P", 3, 1), ("R 3D", 6, 1), ("P 3D", 6, 1), ("H 3D", 6, 1), ("C 3D", 6, 2), ("U 3D", 6, 2), ("S 3D", 6, 3)):
    print("  joint", nm, "F =", F, " removes D-F =", D-F)
chk("2D R,P remove 2; 3D R,P,H remove 5; C,U remove 4; S removes 3", (3-1, 6-1, 6-2, 6-3) == (2, 5, 4, 3))

print("=== B. slide examples (D=3 planar)")
count("serial RPR (p.42)", 3, 4, [1, 1, 1], 3)
count("four-bar (p.44)", 3, 4, [1]*4, 1)
count("3-RPR parallel (p.46, W13 Q6)", 3, 8, [1]*9, 3)
count("three equal parallel legs (p.49, plain)", 3, 5, [1]*6, 0)
count("p.49 after deleting one middle leg = four-bar", 3, 4, [1]*4, 1)
print("  form 1 explicit RPR: 3(4-1) - 2*3 - 1*0 =", 3*3-2*3, "; 3-RPR: 3(8-1)-2*9 =", 3*7-18, "; p.49: 3(5-1)-2*6 =", 3*4-12)
chk("form 1 numbers", (3*3-2*3, 3*3-2*4, 3*7-2*9, 3*4-2*6) == (3, 1, 3, 0))
print("=== C. extra planar counts")
count("five-bar", 3, 5, [1]*5, 2)
count("slider-crank (3R+1P, N=4)", 3, 4, [1]*4, 1)
count("Watt six-bar (N=6, J=7)", 3, 6, [1]*7, 1)
chk("Watt six-bar with D=6 wrongly = -5", gk2(6, 6, [1]*7) == -5)
chk("Watt six-bar, ground forgotten (N=5) = -2", gk2(3, 5, [1]*7) == -2)
count("planar serial 3R", 3, 4, [1]*3, 3)
count("4-RPR platform (N=10, J=12)", 3, 10, [1]*12, 3)
chk("4-RPR no ground (N=9) = 0", gk2(3, 9, [1]*12) == 0)
count("two-finger gripper, 2 x 3R from palm (N=7, J=6)", 3, 7, [1]*6, 6)
chk("four-bar D=6 wrongly = -2; constraints=F coefficient gives 5; no ground (N=3) gives -2",
    gk2(6, 4, [1]*4) == -2 and 3*(4-1) - 4 == 5 and gk2(3, 3, [1]*4) == -2)
chk("3-RPR no ground: N=7 -> 0; prismatic as one link: N=5 -> -6", gk2(3, 7, [1]*9) == 0 and gk2(3, 5, [1]*9) == -6, gk2(3, 5, [1]*9))
chk("Hoberman (24 links, 36 R): D=3 -> -3, D=6 -> -42", gk2(3, 24, [1]*36) == -3 and gk2(6, 24, [1]*36) == -42)
print("=== D. spatial counts (D=6)")
count("serial 6R arm", 6, 7, [1]*6, 6)
count("PUMA RRR / SCARA RRP / cylindrical RPP / Cartesian PPP, each 3 joints", 6, 4, [1]*3, 3)
count("one UPS leg (p.48 drawing)", 6, 4, [2, 1, 3], 6)
count("Stewart 6-UPS (N=14, J=18)", 6, 14, [2, 1, 3]*6, 6)
chk("Stewart: ground forgotten N=13 -> 0; D=3 -> 21", gk2(6, 13, [2, 1, 3]*6) == 0 and gk2(3, 14, [2, 1, 3]*6) == 21)
count("six-SPS (plain)", 6, 14, [3, 1, 3]*6, 12)
count("3-RPS parallel platform (N=8, J=9)", 6, 8, [1, 1, 3]*3, 3)
count("delta plain (N=11, J=15, sumF=39)", 6, 11, ([1] + [3]*4)*3, 9)
chk("delta extended: 9 + R_C(0) - R_M(6) = 3", gk2(6, 11, ([1] + [3]*4)*3) + 0 - 6 == 3)
count("RSSR spatial four-bar plain", 6, 4, [1, 3, 3, 1], 2)
chk("RSSR true: 2 - R_M(1) = 1", gk2(6, 4, [1, 3, 3, 1]) - 1 == 1)
chk("3-SPS (Mock 1): plain 9, extended 9-3 = 6", gk2(6, 8, [3, 1, 3]*3) == 9 and gk2(6, 8, [3, 1, 3]*3) - 3 == 6)

print("=== E. p.49 geometry: constraint Jacobian rank (equal parallel legs vs unequal)")
def rank_p49(L, g=1.0, a=None, th=0.0, pivots=(0.0, 1.0, 2.0), off=None, seed=None):
    rng = np.random.default_rng(seed)
    a = a if a is not None else [1.1]*3
    off = off if off is not None else pivots
    # unknowns: a1..a3, x, y, theta ; eq: G_i + L_i (cos a_i, sin a_i) - (x,y) - R(th)(off_i, 0) = 0
    q = np.array(list(a) + [0.0, 0.0, th], float)
    # solve x,y from leg 0
    G = np.array([[p, 0.0] for p in pivots]); Ls = np.array(L)
    def F(q):
        out = []
        for i in range(3):
            c, s = np.cos(q[5]), np.sin(q[5])
            out += list(G[i] + Ls[i]*np.array([np.cos(q[i]), np.sin(q[i])]) - np.array([q[3], q[4]]) - np.array([c*off[i], s*off[i]]))
        return np.array(out)
    q[3:5] = G[0] + Ls[0]*np.array([np.cos(q[0]), np.sin(q[0])]) - np.array([off[0], 0])  # satisfy leg 0 at th=0
    J = np.zeros((6, 6)); h = 1e-7
    for k in range(6):
        dq = np.zeros(6); dq[k] = h; J[:, k] = (F(q+dq)-F(q-dq))/(2*h)
    return np.linalg.matrix_rank(J, tol=1e-6), np.abs(F(q)).max()
for ang in (np.radians(60), np.radians(75), np.radians(100)):
    rk, res = rank_p49([1.2]*3, a=[ang]*3)
    chk(f"equal legs, spacing 1 = spacing 1, leg angle {np.degrees(ang):.0f} deg: closes (res {res:.1e}), rank {rk} -> 6-{rk} = 1 DoF", rk == 5 and res < 1e-9)
rk, res = rank_p49([1.2, 1.3, 1.1], a=[1.0, 1.1, 0.9], off=(0.0, 1.0, 2.0))
# unequal legs: closure generally fails at this guess; build a closing assembly numerically
from scipy.optimize import fsolve
def closed_unequal(seed):
    rng = np.random.default_rng(seed); L = rng.uniform(1.0, 1.5, 3); off = (0.0, 1.0, 2.0); G = [0.0, 1.0, 2.0]
    def F(q):
        out = []
        for i in range(3):
            out += [G[i] + L[i]*np.cos(q[i]) - q[3] - np.cos(q[5])*off[i], L[i]*np.sin(q[i]) - q[4] - np.sin(q[5])*off[i]]
        return out
    q0 = rng.uniform(0.5, 1.5, 6); q = fsolve(F, q0, xtol=1e-13)
    if np.abs(F(q)).max() > 1e-9: return None
    J = np.zeros((6, 6)); h = 1e-7
    for k in range(6):
        dq = np.zeros(6); dq[k] = h; J[:, k] = (np.array(F(q+dq))-np.array(F(q-dq)))/(2*h)
    return np.linalg.matrix_rank(J, tol=1e-6)
ranks = [r for r in (closed_unequal(s) for s in range(40)) if r is not None]
chk(f"unequal legs ({len(ranks)} assemblies): rank 6 -> 0 DoF (G-K right)", len(ranks) > 5 and set(ranks) == {6}, set(ranks))

print("=== F. workspace numbers")
chk("2R (0.5,0.3): 0.2 <= r <= 0.8", np.isclose(abs(.5-.3), .2) and np.isclose(.5+.3, .8))
def wrist_ok(L, p, phi):
    w = p - L[2]*np.array([np.cos(phi), np.sin(phi)]); rho = np.hypot(*w); return abs(L[0]-L[1]) - 1e-12 <= rho <= L[0]+L[1] + 1e-12
phis = np.linspace(0, 2*np.pi, 3601)
def dex(L, r): return all(wrist_ok(L, np.array([r, 0.]), ph) for ph in phis)
def rea(L, r): return any(wrist_ok(L, np.array([r, 0.]), ph) for ph in phis)
L = (.3, .3, .3)
chk("3R equal links L=0.3: reachable disc 0.9, dexterous disc 0.3", rea(L, .89) and not rea(L, .91) and dex(L, .29) and not dex(L, .31) and dex(L, 0.0))
L = (.5, .3, .2)
chk("3R (0.5,0.3,0.2): reachable disc 1.0; dexterous ring 0.4..0.6", rea(L, .99) and not rea(L, 1.01) and rea(L, .01) and dex(L, .41) and dex(L, .59) and not dex(L, .39) and not dex(L, .61))
print("  2R has 2 DoF < 3 planar pose -> no dexterous workspace (a point's tool angle is fixed by q1+q2 given the point): 2 solutions only ->", 2)
# 2R cannot give every orientation: at a given point only two tool angles
def ik2(L1, L2, x, y):
    c2 = (x*x+y*y-L1*L1-L2*L2)/(2*L1*L2); out = []
    for s in (1, -1):
        q2 = np.arctan2(s*np.sqrt(1-c2*c2), c2); q1 = np.arctan2(y, x)-np.arctan2(L2*np.sin(q2), L1+L2*np.cos(q2)); out.append(np.degrees(q1+q2))
    return out
print("  2R (0.5,0.3) at (0.5,0.3): tool angles q1+q2 =", np.round(ik2(.5, .3, .5, .3), 2))

print("=== G. extra consistency")
for nm, D, N, F in (("serial RPR",3,4,[1]*3),("four-bar",3,4,[1]*4),("3-RPR",3,8,[1]*9),("p.49",3,5,[1]*6),("Stewart",6,14,[2,1,3]*6),("delta",6,11,([1]+[3]*4)*3)):
    ell = len(F)-N+1; chk(f"loop view {nm}: l={ell}, sumF - D*l = {sum(F)-D*ell} = M", sum(F)-D*ell == gk2(D,N,F))
chk("form1 == form2 for D=3,6 and random joint lists", all(gk1(D, N, F) == gk2(D, N, F) for D in (3, 6) for N in range(2, 9) for F in ([1]*k + [2]*m for k in range(5) for m in range(3))))
print("\nALL PASS" if not fails else "\nFAILS: %s" % fails)
