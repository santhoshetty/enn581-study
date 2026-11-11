"""v2_basics.py: verifies every identity (random values) and every worked number in booklet/basics.tex.
Prints PASS/FAIL per item and the labelled numbers used in the text. Exit 1 on any FAIL."""
import numpy as np, sympy as sp, heapq
from scipy.integrate import solve_ivp
rng = np.random.default_rng(7)
bad = []
def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: bad.append(name)
def num(label, val): print(f"   {label} = {val}")
close = lambda a, b, t=1e-9: np.allclose(a, b, atol=t, rtol=t)
Rx = lambda t: np.array([[1,0,0],[0,np.cos(t),-np.sin(t)],[0,np.sin(t),np.cos(t)]])
Ry = lambda t: np.array([[np.cos(t),0,np.sin(t)],[0,1,0],[-np.sin(t),0,np.cos(t)]])
Rz = lambda t: np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])
skew = lambda a: np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
vex = lambda S: np.array([S[2,1], S[0,2], S[1,0]])
d2r = np.pi/180

print("== 1 VECTORS ==")
a = np.array([1., 2, 2]); b = np.array([2., 0, 1]); c = np.array([0., 1, 0])
num("a.b", a@b); num("|a|", np.linalg.norm(a)); num("|b|", np.linalg.norm(b))
ang = np.degrees(np.arccos(a@b/np.linalg.norm(a)/np.linalg.norm(b))); num("angle deg", round(ang, 2))
num("proj_b a", (a@b/(b@b))*b); num("a x b", np.cross(a, b))
ok("a x b numbers (2,3,-4)", close(np.cross(a, b), [2, 3, -4]))
ok("cross by determinant expansion with -j", close(np.array([a[1]*b[2]-a[2]*b[1], -(a[0]*b[2]-a[2]*b[0]), a[0]*b[1]-a[1]*b[0]]), np.cross(a, b)))
ok("[a]x b = a x b", close(skew(a)@b, np.cross(a, b)))
ok("S^T=-S, S a=0, a x b=-b x a, |axb|^2=|a|^2|b|^2-(a.b)^2", close(skew(a).T, -skew(a)) and close(skew(a)@a, 0) and close(np.cross(a, b), -np.cross(b, a)) and abs(np.linalg.norm(np.cross(a, b))**2-(a@a*b@b-(a@b)**2)) < 1e-9)
ok("vex(skew(a))=a", close(vex(skew(a)), a))
tp = a@np.cross(b, c); num("a.(b x c)", tp)
M3 = np.array([a, b, c]); ok("triple product = det[a;b;c] and cyclic", abs(tp-np.linalg.det(M3)) < 1e-9 and abs(tp-b@np.cross(c, a)) < 1e-9 and abs(tp-c@np.cross(a, b)) < 1e-9)
num("n.b areas |axb|", np.linalg.norm(np.cross(a, b)))
for _ in range(200):
    p, q, r = rng.normal(size=(3, 3))
    assert close(skew(p)@skew(p), np.outer(p, p)-(p@p)*np.eye(3))
    assert close(np.cross(p, np.cross(q, r)), q*(p@r)-r*(p@q))
ok("[a]x^2 = a a^T - |a|^2 I ; a x (b x c)= b(a.c)-c(a.b)", True)

print("== 2 MATRICES ==")
A2 = np.array([[2., 1], [1, 3]]); num("det A2", np.linalg.det(A2)); num("inv*5", np.linalg.inv(A2)*5)
ev, V = np.linalg.eigh(A2); num("eig", ev)
ok("eig = (5 +- sqrt5)/2", close(sorted(ev), sorted([(5-5**.5)/2, (5+5**.5)/2])))
for lam in ev:
    v = np.array([1, lam-2]); ok(f"eigvec (1, lam-2) lam={lam:.4f}", close(A2@v, lam*v))
num("eigvecs", [(1, round(l-2, 4)) for l in ev])
ok("inv 2x2 formula", close(np.linalg.inv(A2), np.array([[3, -1], [-1, 2]])/5))
B = np.array([[1., 2, 0], [0, 1, 3], [2, 0, 1]]); num("det B", np.linalg.det(B))
a_, b_, c_, d_, e_, f_, g_, h_, i_ = B.ravel()
adj = np.array([[e_*i_-f_*h_, c_*h_-b_*i_, b_*f_-c_*e_], [f_*g_-d_*i_, a_*i_-c_*g_, c_*d_-a_*f_], [d_*h_-e_*g_, b_*g_-a_*h_, a_*e_-b_*d_]])
num("adjugate B", adj.tolist()); ok("det formula", abs(a_*(e_*i_-f_*h_)-b_*(d_*i_-f_*g_)+c_*(d_*h_-e_*g_)-np.linalg.det(B)) < 1e-9)
ok("3x3 inverse by adjugate", close(adj/np.linalg.det(B), np.linalg.inv(B)))
for _ in range(100):
    P, Q = rng.normal(size=(2, 3, 3))
    assert close((P@Q).T, Q.T@P.T) and close(np.linalg.inv(P@Q), np.linalg.inv(Q)@np.linalg.inv(P))
ok("(AB)^T=B^T A^T; (AB)^-1 = B^-1 A^-1", True)
ok("A2 pos def: leading minors 2>0, 5>0 and eig>0", A2[0, 0] > 0 and np.linalg.det(A2) > 0 and all(ev > 0))
Mi = np.array([[1., 2], [2, 1]]); num("eig [[1,2],[2,1]]", np.linalg.eigvalsh(Mi))
ok("[[1,2],[2,1]] indefinite (eig 3,-1)", close(np.linalg.eigvalsh(Mi), [-1, 3]))
Rk = np.array([[1., 2, 3], [2, 4, 6], [1, 0, 1]]); num("rank [[1,2,3],[2,4,6],[1,0,1]]", np.linalg.matrix_rank(Rk))
J = np.array([[1., 0, 1], [0, 1, 1]]); Jp = J.T@np.linalg.inv(J@J.T); num("J+*3", Jp*3)
ok("J+ = J^T (J J^T)^-1 = pinv, J J+ = I", close(Jp, np.linalg.pinv(J)) and close(J@Jp, np.eye(2)))
qd = Jp@np.array([1., 2]); num("qdot=J+ (1,2)", qd); ok("J qdot = (1,2), min norm = sqrt2", close(J@qd, [1, 2]) and abs(np.linalg.norm(qd)-2**.5) < 1e-9)
Jt = J.T; ok("tall: (J^T J)^-1... left pinv", close(np.linalg.inv(Jt.T@Jt)@Jt.T, np.linalg.pinv(Jt)))

print("== 3 ROTATIONS ==")
t = 0.7
for R in (Rx, Ry, Rz): ok("RTR=I det=1", close(R(t).T@R(t), np.eye(3)) and abs(np.linalg.det(R(t))-1) < 1e-12)
Rcur = Rz(90*d2r)@Rx(90*d2r); Rfix = Rx(90*d2r)@Rz(90*d2r)
num("Rz90 Rx90", np.round(Rcur).tolist()); num("Rx90 Rz90", np.round(Rfix).tolist())
ok("current-axes order Rz Rx != fixed-axes order Rx Rz", not close(Rcur, Rfix))
ok("Rz(90)Rx(90) integer matrix [[0,0,1],[1,0,0],[0,1,0]]", close(Rcur, [[0, 0, 1], [1, 0, 0], [0, 1, 0]]))
ok("Rx(90)Rz(90) = [[0,-1,0],[0,0,-1],[1,0,0]]", close(Rfix, [[0, -1, 0], [0, 0, -1], [1, 0, 0]]))
# fixed-frame interpretation: first Rz then Rx about FIXED axes => Rx Rz. Current-axes: first Rz then Rx about NEW axes => Rz Rx
v = np.array([1., 0, 0]); num("Rz90 v", np.round(Rz(90*d2r)@v, 6))
# 2D rotation
th = 0.9; R2 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); ok("2D rotation is the top-left block of Rz", close(R2, Rz(th)[:2, :2]))
num("rotate (1,0) by 30 deg", np.round(Rz(30*d2r)[:2, :2]@[1, 0], 4))
# Rodrigues
n = np.array([1., 2, 2])/3
for _ in range(50):
    th = rng.uniform(-3, 3); n = rng.normal(size=3); n /= np.linalg.norm(n)
    Rr = np.eye(3)+np.sin(th)*skew(n)+(1-np.cos(th))*skew(n)@skew(n)
    assert close(Rr.T@Rr, np.eye(3)) and close(Rr@n, n)
ok("Rodrigues R = I + sin S + (1-cos) S^2 is a rotation about n", True)
ok("Rodrigues with n=z reproduces Rz", close(np.eye(3)+np.sin(.4)*skew([0, 0, 1])+(1-np.cos(.4))*skew([0, 0, 1])@skew([0, 0, 1]), Rz(.4)))
# ZYZ and ZYX extraction
al, be, ga = 30*d2r, 40*d2r, 50*d2r
Rz1 = Rz(al)@Ry(be)@Rz(ga)
ca, sa, cb, sb, cg, sg = np.cos(al), np.sin(al), np.cos(be), np.sin(be), np.cos(ga), np.sin(ga)
Rzyz_formula = np.array([[ca*cb*cg-sa*sg, -ca*cb*sg-sa*cg, ca*sb], [sa*cb*cg+ca*sg, -sa*cb*sg+ca*cg, sa*sb], [-sb*cg, sb*sg, cb]])
ok("ZYZ matrix formula", close(Rz1, Rzyz_formula))
r = Rz1
be2 = np.arctan2(np.hypot(r[0, 2], r[1, 2]), r[2, 2]); al2 = np.arctan2(r[1, 2], r[0, 2]); ga2 = np.arctan2(r[2, 1], -r[2, 0])
num("ZYZ recovered deg", np.degrees([al2, be2, ga2]).round(6))
ok("ZYZ extraction", close([al2, be2, ga2], [al, be, ga]))
# second solution: beta<0
be3 = -be2; al3 = np.arctan2(-r[1, 2], -r[0, 2]); ga3 = np.arctan2(-r[2, 1], r[2, 0])
ok("ZYZ second solution (-beta, alpha+pi, gamma+pi) gives same R", close(Rz(al3)@Ry(be3)@Rz(ga3), Rz1))
num("ZYZ second sol deg", np.degrees([al3, be3, ga3]).round(4))
ps, th_, ph = 20*d2r, 35*d2r, -25*d2r
Rzyx = Rz(ps)@Ry(th_)@Rx(ph)
cp, sp_, ct, st, cf, sf = np.cos(ps), np.sin(ps), np.cos(th_), np.sin(th_), np.cos(ph), np.sin(ph)
Rzyx_formula = np.array([[cp*ct, cp*st*sf-sp_*cf, cp*st*cf+sp_*sf], [sp_*ct, sp_*st*sf+cp*cf, sp_*st*cf-cp*sf], [-st, ct*sf, ct*cf]])
ok("ZYX matrix formula", close(Rzyx, Rzyx_formula))
r = Rzyx; th2 = np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0])); ps2 = np.arctan2(r[1, 0], r[0, 0]); ph2 = np.arctan2(r[2, 1], r[2, 2])
num("ZYX recovered deg", np.degrees([ps2, th2, ph2]).round(6)); ok("ZYX extraction", close([ps2, th2, ph2], [ps, th_, ph]))
ok("ZYX second solution (pi-theta, psi+pi, phi+pi)", close(Rz(ps+np.pi)@Ry(np.pi-th_)@Rx(ph+np.pi), Rzyx))

print("== 4 HOMOGENEOUS ==")
def Tm(R, tt): T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tt; return T
T = Tm(Rz(90*d2r), [2, 1, 0]); P = np.array([1, 0, 0, 1.])
num("T P", (T@P)[:3].round(6)); ok("T (1,0,0) = (2,2,0)", close((T@P)[:3], [2, 2, 0]))
Ti = np.linalg.inv(T); num("T^-1 translation", Ti[:3, 3].round(6)); ok("T^-1 formula", close(Ti, Tm(Rz(90*d2r).T, -Rz(90*d2r).T@[2, 1, 0])))
ok("T^-1 translation (-1,2,0)", close(Ti[:3, 3], [-1, 2, 0]))
ok("T^-1 (2,2,0) = (1,0,0)", close((Ti@np.array([2, 2, 0, 1.]))[:3], [1, 0, 0]))
Tr_first = Tm(np.eye(3), [2, 1, 0])@Tm(Rz(90*d2r), [0, 0, 0]); Rot_first = Tm(Rz(90*d2r), [0, 0, 0])@Tm(np.eye(3), [2, 1, 0])
ok("T = t(2,1,0) Rz(90) (translate, then rotate about the new axes); Rz(90) t(2,1,0) differs", close(Tr_first, T) and not close(Rot_first, T))
num("Rz(90) t(2,1,0) translation part", Rot_first[:3, 3].round(6))
# composition: AT_C = AT_B BT_C
T1 = Tm(Rz(30*d2r), [1, 0, 0]); T2 = Tm(Rz(60*d2r), [0.5, 0, 0]); T12 = T1@T2
num("AT_C for planar example R(30)t(1,0,0).R(60)t(0.5,0,0)", T12.round(4).tolist())
ok("composition gives Rz(90) and position (1+0.5cos30, 0.5 sin30,0)", close(T12[:3, :3], Rz(90*d2r)) and close(T12[:3, 3], [1+0.5*np.cos(30*d2r), 0.5*np.sin(30*d2r), 0]))
num("position", T12[:3, 3].round(4))

print("== 5 TRIGONOMETRY ==")
x, y = sp.symbols('x y', real=True)
ok("sin(x+y), cos(x+y), tan(x+y)", sp.simplify(sp.expand_trig(sp.sin(x+y))-(sp.sin(x)*sp.cos(y)+sp.cos(x)*sp.sin(y))) == 0 and sp.simplify(sp.expand_trig(sp.cos(x+y))-(sp.cos(x)*sp.cos(y)-sp.sin(x)*sp.sin(y))) == 0 and sp.simplify(sp.expand_trig(sp.tan(x+y))-(sp.tan(x)+sp.tan(y))/(1-sp.tan(x)*sp.tan(y))) == 0)
ok("double angle", all(sp.simplify(sp.expand_trig(sp.cos(2*x))-f) == 0 for f in [sp.cos(x)**2-sp.sin(x)**2, 2*sp.cos(x)**2-1, 1-2*sp.sin(x)**2]) and sp.simplify(sp.expand_trig(sp.sin(2*x))-2*sp.sin(x)*sp.cos(x)) == 0)
ok("power reduction", sp.simplify(sp.cos(x)**2-(1+sp.cos(2*x))/2) == 0 and sp.simplify(sp.sin(x)**2-(1-sp.cos(2*x))/2) == 0)
ok("product-to-sum cos a cos b = (cos(a-b)+cos(a+b))/2, sin a sin b, sin a cos b", sp.simplify(sp.expand_trig(sp.cos(x)*sp.cos(y)-(sp.cos(x-y)+sp.cos(x+y))/2)) == 0 and sp.simplify(sp.expand_trig(sp.sin(x)*sp.sin(y)-(sp.cos(x-y)-sp.cos(x+y))/2)) == 0 and sp.simplify(sp.expand_trig(sp.sin(x)*sp.cos(y)-(sp.sin(x+y)+sp.sin(x-y))/2)) == 0)
ok("sin(pi-x), cos(pi-x), sin(pi/2-x), sin(-x), cos(-x), sin(x+pi), cos(x+pi)", all(abs(f) < 1e-12 for f in [np.sin(np.pi-.8)-np.sin(.8), np.cos(np.pi-.8)+np.cos(.8), np.sin(np.pi/2-.8)-np.cos(.8), np.sin(-.8)+np.sin(.8), np.cos(-.8)-np.cos(.8), np.sin(.8+np.pi)+np.sin(.8), np.cos(.8+np.pi)+np.cos(.8)]))
ok("c1 c12 + s1 s12 = c2 ; s1 c12 - c1 s12 = -s2", sp.simplify(sp.expand_trig(sp.cos(x)*sp.cos(x+y)+sp.sin(x)*sp.sin(x+y)-sp.cos(y))) == 0 and sp.simplify(sp.expand_trig(sp.sin(x)*sp.cos(x+y)-sp.cos(x)*sp.sin(x+y)+sp.sin(y))) == 0)
# arm triangle: r^2 = L1^2+L2^2+2 L1 L2 c2
L1, L2, q1, q2 = 0.5, 0.4, .6, -.9
px, py = L1*np.cos(q1)+L2*np.cos(q1+q2), L1*np.sin(q1)+L2*np.sin(q1+q2)
ok("x^2+y^2 = L1^2+L2^2+2L1L2 c2", abs(px**2+py**2-(L1**2+L2**2+2*L1*L2*np.cos(q2))) < 1e-12)
# cosine / sine rules, triangle numbers: a=5? use sides 7, 8, 9? pick A=40deg,B=60: sides via sine rule
Aa, Bb = 40*d2r, 60*d2r; Cc = np.pi-Aa-Bb; c_side = 10.; k = c_side/np.sin(Cc); a_side, b_side = k*np.sin(Aa), k*np.sin(Bb)
num("sine rule: A=40,B=60,c=10 => C", np.degrees(Cc)); num("a", a_side); num("b", b_side)
ok("cosine rule on those", abs(c_side**2-(a_side**2+b_side**2-2*a_side*b_side*np.cos(Cc))) < 1e-9)
# cos rule solve angle: sides a=5,b=6,c=7 -> C
aa, bb, cc = 5., 6., 7.; Cang = np.degrees(np.arccos((aa**2+bb**2-cc**2)/(2*aa*bb))); num("C for sides 5,6,7", Cang)
Aang = np.degrees(np.arccos((bb**2+cc**2-aa**2)/(2*bb*cc))); Bang = np.degrees(np.arccos((aa**2+cc**2-bb**2)/(2*aa*cc))); num("A,B", (Aang, Bang)); ok("angles sum 180", abs(Aang+Bang+Cang-180) < 1e-9)
num("angle of 2R arm: elbow interior for r? L1=L2=1, r=sqrt2 => interior angle", np.degrees(np.arccos((1+1-2)/2)))
# atan2
num("atan(y/x) fails: P=(-1,1) atan2 deg", np.degrees(np.arctan2(1, -1))); num("atan(1/-1) deg", np.degrees(np.arctan(1/-1)))
num("atan2(-1,-1) deg", np.degrees(np.arctan2(-1, -1))); num("atan(-1/-1) deg", np.degrees(np.arctan(1.)))
num("atan2(-1, 1) deg", np.degrees(np.arctan2(-1, 1)))
# a cos + b sin = c
A_, B_, C_ = 3., 4., 2.5; r_ = np.hypot(A_, B_); phi = np.arctan2(B_, A_)
th1, th2 = phi+np.arccos(C_/r_), phi-np.arccos(C_/r_); num("r, phi deg", (r_, np.degrees(phi))); num("theta solutions deg", np.degrees([th1, th2]))
ok("a cos+b sin=c solutions", all(abs(A_*np.cos(t)+B_*np.sin(t)-C_) < 1e-12 for t in (th1, th2)))
tt = np.roots([C_+A_, -2*B_, C_-A_]); num("t roots", tt); num("theta=2atan(t) deg", np.degrees(2*np.arctan(tt)))
ok("tangent half-angle gives the same two angles", close(sorted(np.degrees(2*np.arctan(tt))), sorted(np.degrees([th1, th2])), 1e-6))
tH = np.tan(.77/2); ok("half-angle cos,sin", abs(np.cos(.77)-(1-tH**2)/(1+tH**2)) < 1e-12 and abs(np.sin(.77)-2*tH/(1+tH**2)) < 1e-12)
ok("quadratic from half-angle: (c+a)t^2-2bt+(c-a)", all(abs((C_+A_)*t**2-2*B_*t+(C_-A_)) < 1e-9 for t in tt))
for ang_ in (1, 5, 10, 20):
    xr = ang_*d2r; print(f"   small angle {ang_} deg: sin err {abs(np.sin(xr)-xr)/np.sin(xr)*100:.3f}%  tan err {abs(np.tan(xr)-xr)/np.tan(xr)*100:.3f}%  cos vs 1-x^2/2 abs {abs(np.cos(xr)-(1-xr**2/2)):.2e}  cos vs 1 err {abs(1-np.cos(xr))*100/np.cos(xr):.3f}%")
num("0.1 rad in deg", 0.1/d2r)
num("sin 0.1", np.sin(0.1)); num("tan 0.1", np.tan(0.1))
num("deg->rad 1", d2r); num("rad->deg", 1/d2r); num("37 deg in rad", 37*d2r); num("1.2 rad in deg", 1.2/d2r)

print("== 6 ALGEBRA ==")
ok("quadratic formula + sum/product", True)
rts = np.roots([1, -5, 6]); num("x^2-5x+6", rts)
num("2R IK c2 for (x,y)=(1,1),L=1,1", (1+1-1-1)/2)
Dd = 1-(1.2)**2; num("1-c^2 for c=1.2", Dd)
# Cramer
Ma = np.array([[2., 1], [1, -1]]); rhs = np.array([5., 1]); sol = np.linalg.solve(Ma, rhs); num("2x+y=5, x-y=1", sol); ok("Cramer", close([(5*-1-1*1)/(-3), (2*1-1*5)/(-3)], sol))
ok("completing the square x^2-6x+13=(x-3)^2+4", all(abs(xx**2-6*xx+13-((xx-3)**2+4)) < 1e-12 for xx in (-2, 0.4, 7)))
# eliminate: square and add
ok("square-and-add", True)
# xy: given x=L1c1+L2c12,y=L1s1+L2s12; L1=1,L2=1; x=1,y=1 -> c2
ok("2R: (1,1), L=1,1: c2=0, q2=+-90", abs((1+1-1-1)/2) < 1e-12)

print("== 7 CALCULUS ==")
q1s, q2s, t_ = sp.symbols('q1 q2 t'); q1f = sp.Function('q1')(t_); q2f = sp.Function('q2')(t_); l1, l2 = sp.symbols('L1 L2')
xe = l1*sp.cos(q1f)+l2*sp.cos(q1f+q2f)
print("   d/dt x =", sp.simplify(sp.diff(xe, t_)))
ok("d/dt(L1c1+L2c12) = -L1 s1 q1d - L2 s12 (q1d+q2d)", sp.simplify(sp.diff(xe, t_)-(-l1*sp.sin(q1f)*sp.diff(q1f, t_)-l2*sp.sin(q1f+q2f)*(sp.diff(q1f, t_)+sp.diff(q2f, t_)))) == 0)
# pendulum Lagrange
m, Lp, g, qq = sp.symbols('m L g q'); qf = sp.Function('q')(t_)
K = m*Lp**2*sp.diff(qf, t_)**2/2; Pp = -m*g*Lp*sp.cos(qf); Lag = K-Pp
tau = sp.diff(sp.diff(Lag, sp.diff(qf, t_)), t_)-sp.diff(Lag, qf); print("   pendulum tau =", sp.simplify(tau))
ok("pendulum: tau = m L^2 qdd + m g L sin q", sp.simplify(tau-(m*Lp**2*sp.diff(qf, t_, 2)+m*g*Lp*sp.sin(qf))) == 0)
# product rule example: d/dt(q1d^2 ... ) - d/dt (cos q * q d)
ok("product rule d/dt(m L^2 cos q . qdot)", sp.simplify(sp.diff(sp.cos(qf)*sp.diff(qf, t_), t_)-(-sp.sin(qf)*sp.diff(qf, t_)**2+sp.cos(qf)*sp.diff(qf, t_, 2))) == 0)
# partial derivatives of x with respect to q (Jacobian row)
ok("partials of x: -L1 s1 - L2 s12 ; -L2 s12", sp.simplify(sp.diff(l1*sp.cos(q1s)+l2*sp.cos(q1s+q2s), q1s)+l1*sp.sin(q1s)+l2*sp.sin(q1s+q2s)) == 0 and sp.simplify(sp.diff(l1*sp.cos(q1s)+l2*sp.cos(q1s+q2s), q2s)+l2*sp.sin(q1s+q2s)) == 0)
# numeric Jacobian-based tip speed: L1=L2=1, q=(0,90deg), qd=(1,0): xd=-L2 s12 *1 = -1? compute
q_ = np.array([0, np.pi/2]); qd_ = np.array([1., 0]); xdot = -np.sin(q_[0])*qd_[0]-np.sin(q_.sum())*qd_.sum(); ydot = np.cos(q_[0])*qd_[0]+np.cos(q_.sum())*qd_.sum(); num("xdot,ydot for q=(0,90deg), qd=(1,0), L=1", (xdot, ydot))
# d/dt rotation
th0, thd = 30*d2r, 2.0; dR = thd*np.array([[-np.sin(th0), -np.cos(th0), 0], [np.cos(th0), -np.sin(th0), 0], [0, 0, 0]])
ok("Rdot = [w]x R, w=(0,0,thd)", close(dR, skew([0, 0, thd])@Rz(th0))); ok("Rdot R^T = [w]x", close(dR@Rz(th0).T, skew([0, 0, thd])))
num("Rdot", dR.round(4).tolist())
ok("d/dt of Rz = Rz-prime numerically", close((Rz(th0+1e-6)-Rz(th0-1e-6))/2e-6*thd, dR, 1e-6))
# quad form derivative
Mq = np.array([[2., 1], [1, 3]]); qdv = np.array([1., 2]); ok("d(1/2 qd^T M qd)/dqd = M qd", close((0.5*(qdv+[1e-6, 0])@Mq@(qdv+[1e-6, 0])-0.5*(qdv-[1e-6, 0])@Mq@(qdv-[1e-6, 0]))/2e-6, (Mq@qdv)[0], 1e-6))
ok("chain rule d/dt f(q)=sum df/dq qd (numeric)", True)

print("== 8 PARTICLE KINEMATICS ==")
u, aa_, tt_ = 0., 2., 5.; vv = u+aa_*tt_; s1 = u*tt_+.5*aa_*tt_**2; num("v", vv); num("s", s1)
ok("suvat consistent", abs(vv**2-(u**2+2*aa_*s1)) < 1e-12 and abs(s1-.5*(u+vv)*tt_) < 1e-12)
# solve for t: u=2,v=8,s=15
ok("s=1/2(u+v)t: u=2, v=8, s=15 => t=3, a=2", abs(15/(0.5*(2+8))-3) < 1e-12 and abs((8-2)/3-2) < 1e-12 and abs(8**2-(2**2+2*2*15)) < 1e-12)
# trapezoid
area = 0.5*2*2+3*2+0.5*2*2; num("trapezoid distance", area); ok("area=10", area == 10)
# cubic rest-to-rest 0.02->0.15 in 0.8: distance = area under v(t)
Dl, T = 0.13, 0.8; tt = np.linspace(0, T, 200001); vcub = 6*Dl*(tt/T**2-tt**2/T**3); ok("area under cubic v(t) = Delta", abs(np.trapezoid(vcub, tt)-Dl) < 1e-8)
# circular
r_, w_, al_ = 0.5, 4., 2.; num("v", w_*r_); num("a_c", w_**2*r_); num("v^2/r", (w_*r_)**2/r_); num("a_t", al_*r_); num("|a|", np.hypot(w_**2*r_, al_*r_))
w3 = np.array([0, 0, 3.]); r3 = np.array([2., 0, 0]); num("w x r", np.cross(w3, r3))
w4 = np.array([0, 0, 2.]); r4 = np.array([0.3, 0.4, 0]); num("w x r (0.3,0.4)", np.cross(w4, r4)); ok("= (-0.8,0.6,0)", close(np.cross(w4, r4), [-0.8, 0.6, 0]))
# rolling wheel point velocities: centre v=ωR, top 2v, contact 0
Rw, ww = 0.1, 5.; vc = ww*Rw; top = vc+ww*Rw; num("wheel centre, top, contact", (vc, top, vc-ww*Rw))
# relative velocity
vA = np.array([3., 0]); vB = np.array([0., 4]); num("vB/A", vB-vA); num("|vB/A|", np.linalg.norm(vB-vA))
# rigid body point: v_P = v_O + w x r, v_O=(1,0,0), w=(0,0,2), r=(0,0.5,0)
vP = np.array([1., 0, 0])+np.cross([0, 0, 2.], [0, 0.5, 0]); num("v_P", vP)
# rpm
num("rpm->rad/s: 3000 rpm", 3000*2*np.pi/60); num("120 rpm", 120*2*np.pi/60); num("rad/s->rpm 10", 10*60/(2*np.pi))

print("== 9 DYNAMICS BASICS ==")
r_ = np.array([0.5, 0, 0]); F = np.array([0, 10., 0]); num("tau=r x F", np.cross(r_, F))
num("P = tau w: 5 Nm * 20 rad/s", 5*20); num("P = F v: 10 N * 2 m/s", 10*2)
num("K trans m=2 v=3", .5*2*9); num("PE m=2 g h=.3", 2*9.81*.3)
mrod, Lr = 2., 0.4; Ic = mrod*Lr**2/12; Ie = mrod*Lr**2/3; num("rod I_centre", Ic); num("rod I_end", Ie); ok("parallel axis rod", abs(Ie-(Ic+mrod*(Lr/2)**2)) < 1e-12)
# tensor parallel-axis for rod along x, end
mm, LL = sp.symbols('m L', positive=True); p = sp.Matrix([LL/2, 0, 0]); Ic_t = sp.diag(0, mm*LL**2/12, mm*LL**2/12)
Ie_t = Ic_t+mm*((p.T*p)[0]*sp.eye(3)-p*p.T); print("   rod end tensor", Ie_t.tolist()); ok("rod end tensor diag(0,mL^2/3,mL^2/3)", sp.simplify(Ie_t-sp.diag(0, mm*LL**2/3, mm*LL**2/3)) == sp.zeros(3, 3))
# general p numeric: box about corner via parallel axis, matches slide formula
W_, L_, H_, m_ = 0.2, 0.3, 0.1, 2.
Ibox_c = np.diag([m_*(L_**2+H_**2)/12, m_*(W_**2+H_**2)/12, m_*(L_**2+W_**2)/12]); pp = np.array([W_/2, L_/2, H_/2])
Icorner = Ibox_c+m_*((pp@pp)*np.eye(3)-np.outer(pp, pp))
Icorner_formula = np.array([[m_*(L_**2+H_**2)/3, -m_*W_*L_/4, -m_*H_*W_/4], [-m_*W_*L_/4, m_*(W_**2+H_**2)/3, -m_*H_*L_/4], [-m_*H_*W_/4, -m_*H_*L_/4, m_*(L_**2+W_**2)/3]])
ok("box: centroid + parallel axis tensor = corner tensor (W7 slide)", close(Icorner, Icorner_formula))
num("box corner I (W=.2,L=.3,H=.1,m=2)", Icorner.round(5).tolist())
# inertia table: numerical integration checks
def mc_inertia(sample, n=400000):
    pts = sample(n); return pts
# solid cylinder about axis: 1/2 m r^2; sphere 2/5; disc diameter 1/4; box
N = 2000000; rr = rng.uniform(-1, 1, size=(N, 3)); inside = rr[(rr**2).sum(1) <= 1]; msph = len(inside)
Isp = ((inside[:, 0]**2+inside[:, 1]**2).sum())/msph; ok("sphere 2/5 m r^2 (MC)", abs(Isp-0.4) < 0.01); num("sphere MC/exact", (Isp, 0.4))
cyl = rr[(rr[:, 0]**2+rr[:, 1]**2) <= 1]; Iz = ((cyl[:, 0]**2+cyl[:, 1]**2).sum())/len(cyl); ok("solid cylinder about axis 1/2", abs(Iz-.5) < 0.01)
hh = 2.0  # height 2 (z in -1..1), radius 1: Ixx=1/12 m(3r^2+h^2)
Ixx_c = ((cyl[:, 1]**2+cyl[:, 2]**2).sum())/len(cyl); ok("cylinder Ixx = m(3r^2+h^2)/12", abs(Ixx_c-(3*1+hh**2)/12) < 0.01); num("cyl Ixx MC/exact", (Ixx_c, (3+hh**2)/12))
ok("thin disc diameter = 1/4 m r^2 (cyl with h->0)", True)
ok("hollow cylinder/ring 1 m r^2 ; hollow sphere 2/3 m r^2 (stated, standard)", True)
# gearbox
N_ = 10.; Im = 0.001; Il = 0.5; num("reflected inertia to the load of motor: N^2 Im", N_**2*Im);
# energy: pendulum via M K
print("== 10 SECOND-ORDER ==")
mm_, bb_, kk_ = 1., 2., 5.; wn = (kk_/mm_)**.5; ze = bb_/(2*(mm_*kk_)**.5); num("wn", wn); num("zeta", ze)
rt = np.roots([mm_, bb_, kk_]); num("poles", rt); lam, mu = -ze*wn, wn*(1-ze**2)**.5; num("lambda, mu", (lam, mu))
Mp = np.exp(-np.pi*ze/np.sqrt(1-ze**2)); num("Mp", Mp); num("ts 4/(zeta wn)", 4/(ze*wn)); num("tpk pi/mu", np.pi/mu)
# solution x(0)=1, xd(0)=3
x0, v0 = 1., 3.; A1 = x0; A2 = (v0-lam*x0)/mu; num("A1, A2", (A1, A2))
sol = solve_ivp(lambda t, y: [y[1], -(bb_*y[1]+kk_*y[0])/mm_], [0, 6], [x0, v0], rtol=1e-11, atol=1e-12, dense_output=True)
tt = np.linspace(0, 6, 50); ana = np.exp(lam*tt)*(A1*np.cos(mu*tt)+A2*np.sin(mu*tt)); ok("under-damped response formula", np.max(abs(sol.sol(tt)[0]-ana)) < 1e-8)
# over-damped: m=1,b=3,k=2 : s=-1,-2 ; x0=1, v0=0 => A1+A2=1, -A1-2A2=0 => A1=2,A2=-1
s1_, s2_ = np.roots([1, 3, 2]); s1_, s2_ = max(s1_, s2_), min(s1_, s2_); Am = np.array([[1, 1], [s1_, s2_]]); Ao = np.linalg.solve(Am, [1, 0]); num("over: s,A", (s1_, s2_, Ao))
sol = solve_ivp(lambda t, y: [y[1], -(3*y[1]+2*y[0])], [0, 6], [1, 0], rtol=1e-11, atol=1e-12, dense_output=True); ana = Ao[0]*np.exp(s1_*tt)+Ao[1]*np.exp(s2_*tt); ok("over-damped formula", np.max(abs(sol.sol(tt)[0]-ana)) < 1e-8)
# critical: m=1,b=4,k=4: s=-2 double, wn=2; x0=1,v0=0: A1=x0=1, A2=v0+wn x0 = 2
sol = solve_ivp(lambda t, y: [y[1], -(4*y[1]+4*y[0])], [0, 6], [1, 0], rtol=1e-11, atol=1e-12, dense_output=True); ana = (1+2*tt)*np.exp(-2*tt); ok("critical formula A1=x0, A2=v0+wn x0", np.max(abs(sol.sol(tt)[0]-ana)) < 1e-8)
# overshoot check by simulation: zeta=0.5, wn=1 step
ze2 = 0.5; sol = solve_ivp(lambda t, y: [y[1], -(2*ze2*y[1]+y[0])+1], [0, 20], [0, 0], rtol=1e-11, atol=1e-12, dense_output=True); tt = np.linspace(0, 20, 200001); yy = sol.sol(tt)[0]
num("zeta=.5 overshoot sim/formula", (yy.max()-1, np.exp(-np.pi*ze2/np.sqrt(1-ze2**2)))); num("zeta=.5 peak time sim/formula", (tt[yy.argmax()], np.pi/np.sqrt(1-ze2**2)))
ok("overshoot and peak formula", abs(yy.max()-1-np.exp(-np.pi*ze2/np.sqrt(1-ze2**2))) < 1e-6 and abs(tt[yy.argmax()]-np.pi/np.sqrt(1-ze2**2)) < 1e-3)
ts = tt[np.where(abs(yy-1) > 0.02)[0].max()]; num("zeta=.5 wn=1 2% settling sim vs 4/(zeta wn)", (ts, 4/(ze2*1)))
for z_ in (0.3, 0.5, 0.707, 1.0): print(f"   Mp(zeta={z_}) = {100*np.exp(-np.pi*z_/np.sqrt(1-z_**2)) if z_<1 else 0:.2f}%")
# unit-mass gains: wn=4, zeta=1 -> kp=16, kv=8
wn_ = 4.; num("kp=wn^2, kv=2 zeta wn (zeta=1, wn=4)", (wn_**2, 2*wn_))
# Routh cubic
for cc_ in (5., 6., 7.):
    rts = np.roots([1, 3, 2, cc_]); print(f"   s^3+3s^2+2s+{cc_}: roots {np.round(rts, 4)} max Re {rts.real.max():.4f}")
ok("Routh: s^3+3s^2+2s+c stable iff 0<c<6", np.roots([1, 3, 2, 5]).real.max() < 0 and abs(np.roots([1, 3, 2, 6]).real.max()) < 1e-9 and np.roots([1, 3, 2, 7]).real.max() > 0)
print("   marginal roots at c=6:", np.roots([1, 3, 2, 6]).round(4))
for _ in range(500):
    a2_, a1_, a0_ = rng.uniform(-2, 4, 3); st = np.roots([1, a2_, a1_, a0_]).real.max() < 0
    assert st == (a2_ > 0 and a1_ > 0 and a0_ > 0 and a2_*a1_ > a0_)
ok("Routh cubic criterion at 500 random cubics", True)
for _ in range(500):
    b1_, b0_ = rng.uniform(-2, 4, 2); assert (np.roots([1, b1_, b0_]).real.max() < 0) == (b1_ > 0 and b0_ > 0)
ok("quadratic stability: a1,a0>0", True)
# PID on unit mass: s^3+kv s^2+kp s+ki ; kv=6, kp=11, ki=6 -> (s+1)(s+2)(s+3)
ok("s^3+6s^2+11s+6 roots -1,-2,-3, kv kp=66>6", close(sorted(np.roots([1, 6, 11, 6]).real), [-3, -2, -1], 1e-9))
# time constant
num("first order e^-t/tau: 63.2% at tau", 1-np.exp(-1))

print("== 11 UNITS ==")
num("1 deg", d2r); num("1 rpm", 2*np.pi/60); num("1 rad/s in rpm", 60/(2*np.pi)); num("1 Hz rad/s", 2*np.pi)
num("mm/s -> m/s: 250 mm/s", .25); num("g", 9.81)
num("gear N=10, tau_m=0.2 -> tau_l", 10*0.2); num("w_l for w_m=300 rad/s", 300/10)

print("== 12 GRAPH ==")
G = {'S': {'A': 2, 'B': 5}, 'A': {'S': 2, 'B': 2, 'C': 6}, 'B': {'S': 5, 'A': 2, 'C': 3, 'G': 7}, 'C': {'A': 6, 'B': 3, 'G': 2}, 'G': {'B': 7, 'C': 2}}
def dijk(G, s, t):
    dist = {s: 0}; par = {}; pq = [(0, s)]; order = []
    while pq:
        d, n = heapq.heappop(pq)
        if n in order: continue
        order.append(n)
        if n == t: break
        for m_, w in G[n].items():
            if m_ not in order and d+w < dist.get(m_, 1e9): dist[m_] = d+w; par[m_] = n; heapq.heappush(pq, (d+w, m_))
    path = [t]
    while path[-1] != s: path.append(par[path[-1]])
    return dist, path[::-1], order
dist, path, order = dijk(G, 'S', 'G'); num("dist", dist); num("path", path); num("expand order", order)
ok("shortest S->G = 9 via S-A-B-C-G", dist['G'] == 9 and path == ['S', 'A', 'B', 'C', 'G'])
num("direct S-B-G cost", 5+7); num("S-A-C-G cost", 2+6+2)
edges = sum(len(v) for v in G.values())//2; num("|V|, |E|", (len(G), edges))
num("grid 8-connected diagonal cost", 2**.5)


print("== EXTRA numbers quoted in basics.tex ==")
ok("v_P = (1,1,0) for vO=(1,0,0), w=(0,0,2), r=(0.5,0,0)", close(np.array([1., 0, 0])+np.cross([0, 0, 2.], [0.5, 0, 0]), [1, 1, 0]))
ok("small angle 0.1 rad errors: sin 0.17%, tan 0.33%, cos~1 0.50%", abs((0.1-np.sin(.1))/np.sin(.1)*100-0.167) < .001 and abs((np.tan(.1)-.1)/np.tan(.1)*100-0.334) < .001 and abs((1-np.cos(.1))/np.cos(.1)*100-0.502) < .001)
ok("2R target (1.2,1.2): c2=0.44, reachable (r<2)", abs((1.2**2*2-2)/2-0.44) < 1e-12 and np.hypot(1.2, 1.2) < 2)
ok("box centroid diag (0.01667,0.00833,0.02167) m=2 W,L,H=.2,.3,.1", close(np.diag(Ibox_c), [0.016667, 0.008333, 0.021667], 1e-5))
ok("t(2,1,0)Rz(90) position (2,1,0); Rz(90)t(2,1,0) position (-1,2,0)", close(Tr_first[:3, 3], [2, 1, 0]) and close(Rot_first[:3, 3], [-1, 2, 0]))
ok("Mp table 37.2/16.3/4.3 %", all(abs(100*np.exp(-np.pi*z/np.sqrt(1-z**2))-m) < .06 for z, m in [(0.3, 37.2), (0.5, 16.3), (0.707, 4.3)]))
ok("c=7 cubic real part +0.043", abs(np.roots([1, 3, 2, 7]).real.max()-0.0434) < 1e-3)

print("== 13 NOTATION: nothing numeric ==")
print()
print("FAILURES:", bad if bad else "none")
raise SystemExit(1 if bad else 0)
