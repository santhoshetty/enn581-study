#!/usr/bin/env python3
"""v2 W5 booklet numbers: every number printed here is used in w05.tex."""
import numpy as np, sympy as sp
np.set_printoptions(precision=5, suppress=True)
def P(label, *v):
    print(f"{label:46s}", *[(np.round(x,5) if isinstance(x,(float,np.floating)) else x) for x in v])

# ---------- A. cubic derivation (symbolic) ----------
t,T,q0,qF,v0,vF,t0 = sp.symbols('t T q0 qF v0 vF t0')
a0,a1,a2,a3 = sp.symbols('a0:4')
q = a0+a1*t+a2*t**2+a3*t**3
qd = sp.diff(q,t)
sol = sp.solve([q.subs(t,0)-q0, q.subs(t,T)-qF, qd.subs(t,0)-v0, qd.subs(t,T)-vF],[a0,a1,a2,a3],dict=True)[0]
D = sp.symbols('D')
P('a2 general', sp.simplify(sol[a2]-(3*(qF-q0)-(2*v0+vF)*T)/T**2))
P('a3 general', sp.simplify(sol[a3]-(-2*(qF-q0)+(v0+vF)*T)/T**3))
# elimination step: with a0=q0,a1=v0:  a2 T^2 + a3 T^3 = D - v0 T ; 2 a2 T + 3 a3 T^2 = vF - v0
A = sp.Matrix([[T**2,T**3],[2*T,3*T**2]])
b = sp.Matrix([qF-q0-v0*T, vF-v0])
s2 = sp.simplify(A.inv()*b)
P('2x2 inverse det (T^4)', sp.simplify(A.det()))
P('2x2 solve a2', sp.simplify(s2[0]-sol[a2]), 'a3', sp.simplify(s2[1]-sol[a3]))
# inverse of the 4x4 matrix (rest-to-rest rows)
M4 = sp.Matrix([[1,0,0,0],[1,T,T**2,T**3],[0,1,0,0],[0,1,2*T,3*T**2]])
print('M4^-1 =', sp.simplify(M4.inv()))
def cubic(q0,qF,v0,vF,T):
    return (q0, v0, (3*(qF-q0)-(2*v0+vF)*T)/T**2, (-2*(qF-q0)+(v0+vF)*T)/T**3)
P('ex1 10->50 2s rest', cubic(10,50,0,0,2))
c=cubic(10,50,0,0,2); P('  ex1 peak v at 1', c[1]+2*c[2]*1+3*c[3]*1, ' qdd(0),qdd(2)', 2*c[2], 2*c[2]+6*c[3]*2)
P('ex2 20->80 3s vF=10', cubic(20,80,0,10,3))
# general t0 : 10 at t0=1 -> 50 at tF=3 ; tau=t-t0
c=cubic(10,50,0,0,3-1); P('t0=1 tF=3 coefficients (tau)', c, ' q(t=2)=', c[0]+c[2]*1+c[3]*1)
# multi joint
for D_ in (40,90,-20):
    c=cubic(0,D_,0,0,2); P(f'multi joint D={D_} T=2 a2,a3,peak', c[2],c[3],1.5*D_/2)
# peak facts
P('rest-rest end qdd = 6D/T^2 (D=40,T=2)', 6*40/4)
# ---------- B. via points: five points rule ----------
def rule(qs,ts):
    s=[(qs[i+1]-qs[i])/(ts[i+1]-ts[i]) for i in range(len(qs)-1)]
    w=[0.0]
    for i in range(1,len(qs)-1):
        w.append(0.0 if s[i-1]*s[i]<=0 else 0.5*(s[i-1]+s[i]))
    w.append(0.0); return s,w
qs=[10,50,20,40,45]; ts=[0,5,10,15,20]
s,w=rule(qs,ts); P('Q2 slopes',s); P('Q2 via velocities',w)
segs=[cubic(qs[i],qs[i+1],w[i],w[i+1],ts[i+1]-ts[i]) for i in range(4)]
for i,c in enumerate(segs): P(f'Q2 seg{i} a0..a3',c)
Ts=[5]*4
for i in range(3):
    c1=segs[i];c2=segs[i+1]
    P(f'Q2 accel jump at via{i+1}', 2*c2[2]-(2*c1[2]+6*c1[3]*5))
c=segs[3]; P('Q2 last seg q(5)', c[0]+c[1]*5+c[2]*25+c[3]*125)
# QA via points 10,30,70,80,60 at 0,2,5,9,12
s,w=rule([10,30,70,80,60],[0,2,5,9,12]); P('QA slopes',s); P('QA via vel',w)
c=cubic(70,80,w[2],w[3],4); P('QA seg 5->9', c)
# ---------- C. Q8 symbolic: two cubics, continuous acceleration ----------
th0,thv,thg,T1,T2,w_ = sp.symbols('theta0 theta_v theta_g T1 T2 w')
def cub(q0,qF,v0,vF,T): return ((3*(qF-q0)-(2*v0+vF)*T)/T**2, (-2*(qF-q0)+(v0+vF)*T)/T**3)
a2_1,a3_1=cub(th0,thv,0,w_,T1); a2_2,a3_2=cub(thv,thg,w_,0,T2)
acc1=2*a2_1+6*a3_1*T1; acc2=2*a2_2
ww=sp.solve(sp.Eq(acc1,acc2),w_)[0]
P('w general check', sp.simplify(ww-3*((T1**2)*(thg-thv)+(T2**2)*(thv-th0))/(2*T1*T2*(T1+T2))))
P('acc1 form', sp.simplify(acc1-(-6*(thv-th0)+4*w_*T1)/T1**2))
tf=sp.symbols('t_f')
wE=sp.simplify(ww.subs({T1:tf,T2:tf})); P('w equal T', sp.simplify(wE-3*(thg-th0)/(4*tf)))
co={}
for name,expr in [('a12',a2_1),('a13',a3_1),('b21',w_),('b22',a2_2),('b23',a3_2)]:
    e=sp.simplify(expr.subs(w_,wE).subs({T1:tf,T2:tf})); co[name]=e; print(name,'=',sp.factor(e))
# slide statement: a12=(12thv-3thg-9th0)/(4tf^2) ...
P('a12 match', sp.simplify(co['a12']-(12*thv-3*thg-9*th0)/(4*tf**2)))
P('a13 match', sp.simplify(co['a13']-(-8*thv+3*thg+5*th0)/(4*tf**3)))
P('b22 match', sp.simplify(co['b22']-(-12*thv+6*thg+6*th0)/(4*tf**2)))
P('b23 match', sp.simplify(co['b23']-(8*thv-5*thg-3*th0)/(4*tf**3)))
accE=sp.simplify(acc2.subs(w_,wE).subs({T1:tf,T2:tf})); P('shared accel', sp.simplify(accE-3*(th0+thg-2*thv)/tf**2))
# brute force 8x8 for the numbers
def two_cubics(q0,qv,qg,T1,T2):
    a=sp.symbols('a0:4'); b=sp.symbols('b0:4'); tt=sp.symbols('tt')
    f1=sum(a[k]*tt**k for k in range(4)); f2=sum(b[k]*tt**k for k in range(4))
    eqs=[f1.subs(tt,0)-q0, sp.diff(f1,tt).subs(tt,0), f1.subs(tt,T1)-qv, f2.subs(tt,0)-qv, f2.subs(tt,T2)-qg,
         sp.diff(f2,tt).subs(tt,T2), sp.diff(f1,tt).subs(tt,T1)-sp.diff(f2,tt).subs(tt,0),
         sp.diff(f1,tt,2).subs(tt,T1)-sp.diff(f2,tt,2).subs(tt,0)]
    s=sp.solve(eqs,list(a)+list(b),dict=True)[0]
    return [sp.nsimplify(s[x]) for x in a],[sp.nsimplify(s[x]) for x in b]
for args in [(10,40,20,1,1),(10,40,20,1,2),(0,40,30,1,1),(0,40,60,2,2)]:
    A_,B_=two_cubics(*[sp.Rational(x) for x in args]); P('two cubics',args,[float(x) for x in A_],[float(x) for x in B_])
    a=[float(x) for x in A_]; b=[float(x) for x in B_]; T1_=args[3]
    P('   via velocity,acc1,acc2', a[1]+2*a[2]*T1_+3*a[3]*T1_**2, b[1], 2*a[2]+6*a[3]*T1_, 2*b[2])
# peak speed segment 1 for (0,40,30,1,1)
A_,B_=two_cubics(0,40,30,1,1); a=[float(x) for x in A_]
tp=-2*a[2]/(6*a[3]); P('mock: t of peak speed, peak speed', tp, a[1]+2*a[2]*tp+3*a[3]*tp**2)
# symbolic: via velocity examples
P('w (10,40,20,T1=1,T2=2)', float(ww.subs({th0:10,thv:40,thg:20,T1:1,T2:2})))
P('w QA (0,40,60,2,2)', float(ww.subs({th0:0,thv:40,thg:60,T1:2,T2:2})))
# ---------- D. Q9 ----------
c=cubic(0.02,0.15,0,0,0.8); P('Q9 a0..a3', c); P('Q9 peak v,t', 1.5*0.13/0.8, 0.4, ' qdd end', 6*0.13/0.64, ' d(0.4)', c[0]+c[2]*0.16+c[3]*0.064)
# ---------- E. quintic ----------
a3_,a4_,a5_=sp.symbols('a3 a4 a5'); Dq=sp.symbols('Dq')
eq=[a3_*T**3+a4_*T**4+a5_*T**5-Dq, 3*a3_*T**2+4*a4_*T**3+5*a5_*T**4, 6*a3_*T+12*a4_*T**2+20*a5_*T**3]
s5=sp.solve(eq,[a3_,a4_,a5_]); print('quintic', {k:sp.simplify(v) for k,v in s5.items()})
# eliminating: divide by T^3 etc
P('quintic 0->15 1s', 150,-225,90)
tq=sp.symbols('tq'); fq=Dq*(10*(tq/T)**3-15*(tq/T)**4+6*(tq/T)**5)
P('quintic peak speed', sp.simplify(sp.diff(fq,tq).subs(tq,T/2)))
acc=sp.diff(fq,tq,2); crit=sp.solve(sp.diff(acc,tq),tq); P('quintic accel extremum t/T', [sp.simplify(c_/T) for c_ in crit], 'val', [sp.simplify(acc.subs(tq,c_)*T**2/Dq) for c_ in crit])
M6=sp.Matrix([[1,0,0,0,0,0],[1,T,T**2,T**3,T**4,T**5],[0,1,0,0,0,0],[0,1,2*T,3*T**2,4*T**3,5*T**4],[0,0,2,0,0,0],[0,0,2,6*T,12*T**2,20*T**3]])
P('M6 det', sp.factor(M6.det()))
# time-scaling functions
tau=sp.symbols('tau'); sc=3*tau**2-2*tau**3; P('cubic s peak', sp.diff(sc,tau).subs(tau,sp.Rational(1,2)))
# ---------- F. LSPB ----------
def lspb(D,tF,a):
    disc=a*a*tF*tF-4*a*D
    if disc<0: return None
    tB=tF/2-np.sqrt(disc)/(2*a); return tB,a*tB
tB,v=lspb(120,2,300); P('LSPB 120/2/300 tB,v', tB,v)
P('   qB1, q at tF-tB, area check', 0.5*300*tB**2, 120-0.5*300*tB**2, 300*tB*(2-tB), v*(2-tB))
P('   plus root', 1-(-1)*np.sqrt(300**2*4-4*300*120)/600, 'v with plus root', 300*(1+np.sqrt(300**2*4-4*300*120)/600))
P('   a=120', lspb(120,2,120)); P('   a=100 disc', 100**2*4-4*100*120)
P('   v=70 -> tB, a', 2-120/70, 70/(2-120/70))
P('   area form tB=tF-D/v for v=67.62', 2-120/67.6177)
P('   v in terms: 0.5(a tF - sqrt)', 0.5*(300*2-np.sqrt(300**2*4-4*300*120)))
P('   a=150: tB,v', lspb(120,2,150))
P('LSPB 75/1.5/200', lspb(75,1.5,200), 'min a', 4*75/1.5**2)
tB,v=lspb(75,1.5,200); P('   coast time, check', 1.5-2*tB, 0.5*200*tB**2*2+v*(1.5-2*tB))
P('LSPB 75/1.5/130 disc', 130**2*1.5**2-4*130*75)
P('LSPB mock 60/3/30', lspb(60,3,30), 'min', 4*60/9, 'qB1', 10+0.5*30*1.0, 'coast', 3-2)
P('LSPB 45/1.5/160', lspb(45,1.5,160), 'min', 4*45/1.5**2)
P('LSPB 90/1.5 min a, tB, peak', 4*90/1.5**2, 0.75, 2*90/1.5)
# min time with vmax, amax
def tmin(D,vm,am):
    return (D/vm+vm/am, 'trap') if D>=vm*vm/am else (2*np.sqrt(D/am),'tri')
P('tmin 90 (60,120)', tmin(90,60,120), ' 20', tmin(20,60,120), 'vp20', np.sqrt(120*20), 'limit', 60*60/120)
P('cubic min time 90 vmax60 amax120', 1.5*90/60, np.sqrt(6*90/120))
# 120 deg example with vmax constraints: slope check for a=300: v>D/tF
P('v range D/tF..2D/tF for 120/2', 60,120)
# kinematics-builder example: accelerate from rest to v in tB at a: s=1/2 a tB^2, v=a tB
# ---------- G. Craig multi-segment ----------
def craig(th,td,acc):
    n=len(th); sg=np.sign
    tb=[0]*n; vv=[0]*(n-1); aa=[0]*n
    aa[0]=sg(th[1]-th[0])*abs(acc[0]); tb[0]=td[0]-np.sqrt(td[0]**2-2*(th[1]-th[0])/aa[0]); vv[0]=(th[1]-th[0])/(td[0]-0.5*tb[0])
    aa[-1]=sg(th[-2]-th[-1])*abs(acc[-1]); tb[-1]=td[-1]-np.sqrt(td[-1]**2+2*(th[-1]-th[-2])/aa[-1])
    for j in range(1,n-2): vv[j]=(th[j+1]-th[j])/td[j]
    if n>2: vv[n-2]=(th[-1]-th[-2])/(td[-1]-0.5*tb[-1])   # last segment: its end blend eats half
    for j in range(1,n-1):
        aa[j]=sg(vv[j]-vv[j-1])*abs(acc[j]); tb[j]=(vv[j]-vv[j-1])/aa[j]
    return tb,vv,aa
tb,vv,aa=craig([0,40,10],[2,2],[50,50,50]); P('Craig t_j',tb); P('Craig v',vv); P('Craig a',aa)
P('  linear parts (first: td-t1-t2/2, last: td-t2/2-t3)', 2-tb[0]-0.5*tb[1], 2-0.5*tb[1]-tb[2], ' miss at 40', abs(aa[1])*tb[1]**2/8, 40-abs(aa[1])*tb[1]**2/8)
# ---------- H. Cartesian: RR ----------
def ik(x,y,L1,L2,sgn=1):
    c2=(x*x+y*y-L1**2-L2**2)/(2*L1*L2); s2=sgn*np.sqrt(max(0,1-c2*c2))
    q2=np.arctan2(s2,c2); q1=np.arctan2(y,x)-np.arctan2(L2*np.sin(q2),L1+L2*np.cos(q2)); return q1,q2
def fk(q1,q2,L1,L2): return L1*np.cos(q1)+L2*np.cos(q1+q2), L1*np.sin(q1)+L2*np.sin(q1+q2)
L1,L2=4.0,3.0
Lp=np.hypot(4.75,4.75); tF=Lp/3; P('path length, time', Lp, tF)
print('i  t   x   y   q1deg q2deg  (q2>0)')
for i in range(11):
    ti=tF*i/10; x=3*ti/np.sqrt(2); y=4.75-x
    q1,q2=ik(x,y,L1,L2); print(i, round(ti,4), round(x,4), round(y,4), round(np.degrees(q1),2), round(np.degrees(q2),2), ' fk ok', np.allclose(fk(q1,q2,L1,L2),(x,y)))
# T/2
ti=tF/2; x=3*ti/np.sqrt(2); y=4.75-x; q1,q2=ik(x,y,L1,L2); P('T/2: t,x,y', ti,x,y); P('T/2 q1,q2 deg', np.degrees(q1),np.degrees(q2),' rad', q1,q2)
q1b,q2b=ik(x,y,L1,L2,-1); P('T/2 other branch deg', np.degrees(q1b),np.degrees(q2b))
qa=np.array(ik(0,4.75,L1,L2)); qb=np.array(ik(4.75,0,L1,L2)); P('start deg', np.degrees(qa),' end deg', np.degrees(qb), 'rad', qa, qb)
qm=(qa+qb)/2; P('joint-linear mid deg', np.degrees(qm), 'tool', fk(*qm,L1,L2), 'dist from line mid', np.hypot(*(np.array(fk(*qm,L1,L2))-2.375)))
qa2=np.array(ik(0,4.75,L1,L2,-1)); qb2=np.array(ik(4.75,0,L1,L2,-1)); P('other branch start,end deg', np.degrees(qa2),np.degrees(qb2))
# cubic fit
ti=np.array([tF*i/10 for i in range(11)]); xs=3*ti/np.sqrt(2); ys=4.75-xs
Q=np.array([ik(x,y,L1,L2) for x,y in zip(xs,ys)])
for k in (0,1):
    p3=np.polyfit(ti,Q[:,k],3); P(f'cubic fit q{k+1}', p3, 'max err deg', np.degrees(np.max(abs(np.polyval(p3,ti)-Q[:,k]))))
# wander of fitted tool path
tt=np.linspace(0,tF,200); P1=np.polyfit(ti,Q[:,0],3); P2=np.polyfit(ti,Q[:,1],3)
xf,yf=fk(np.polyval(P1,tt),np.polyval(P2,tt),L1,L2); d=np.abs(xf+yf-4.75)/np.sqrt(2); P('fitted path max dist from line (mm)', d.max()*1000)
P5=np.polyfit(ti,Q[:,0],5);P6=np.polyfit(ti,Q[:,1],5); xf,yf=fk(np.polyval(P5,tt),np.polyval(P6,tt),L1,L2); P('quintic fit max dist (mm)', (np.abs(xf+yf-4.75)/np.sqrt(2)).max()*1000)
# velocity via Jacobian at mid (3 m/s along line)
def J(q1,q2,L1,L2): return np.array([[-L1*np.sin(q1)-L2*np.sin(q1+q2), -L2*np.sin(q1+q2)],[L1*np.cos(q1)+L2*np.cos(q1+q2), L2*np.cos(q1+q2)]])
q1,q2=ik(2.375,2.375,L1,L2); vt=3*np.array([1,-1])/np.sqrt(2); P('qdot at midpoint (rad/s)', np.linalg.solve(J(q1,q2,L1,L2),vt), 'det J', np.linalg.det(J(q1,q2,L1,L2)), 'L1L2 sin q2', L1*L2*np.sin(q2))
# singularity examples (L1=L2=1, 1 m/s along x on y=0 ... from v1: max rate at x)
for x in (1.0,1.9,1.99,1.999):
    q1,q2=ik(x,0.0,1,1); Jm=J(q1,q2,1,1); qd=np.linalg.solve(Jm,[1,0]); P(f'RR L=1 y=0 x={x} max |qdot|', np.abs(qd).max())
# hole: L1=1 L2=0.7, line y=0.2
P('hole ends r', np.hypot(1,0.2), 'middle r', 0.2, 'inner radius', 0.3)
# circle scenario
Lc1,Lc2=0.6,0.4; cx,cy,r=0.5,0.3,0.2; Tc=4.0
print('circle  centre (0.5,0.3) r=0.2 T=4  L1=0.6 L2=0.4 (q2>0)')
for tc in (0,1,2,3):
    ph=2*np.pi*tc/Tc; x=cx+r*np.cos(ph); y=cy+r*np.sin(ph); q1,q2=ik(x,y,Lc1,Lc2)
    P(f' t={tc} x,y', x,y,' q1,q2 deg', np.degrees(q1),np.degrees(q2))
x0,y0=cx+r,cy; q1,q2=ik(x0,y0,Lc1,Lc2); om=2*np.pi/Tc; vt=np.array([0,om*r]); P('circle: speed', om*r, 'qdot at t=0', np.linalg.solve(J(q1,q2,Lc1,Lc2),vt), 'detJ', np.linalg.det(J(q1,q2,Lc1,Lc2)))
P('circle reach check: |centre|, r_out, r_in', np.hypot(cx,cy), Lc1+Lc2, Lc1-Lc2, 'min,max radius of path', np.hypot(cx,cy)-r, np.hypot(cx,cy)+r)
# other points for circle:  t=2 point (0.3,0.3)
P('circle t=2 FK check', fk(*ik(0.3,0.3,Lc1,Lc2),Lc1,Lc2))

# Craig check by integrating: build the path and verify it reaches 0 ... 36.22 ... 10 and speeds continuous
tb,vv,aa=craig([0,40,10],[2,2],[50,50,50])
# position at end of first blend, start of 2nd blend
q1e=0.5*aa[0]*tb[0]**2; P('first blend end q', q1e, 'speed', aa[0]*tb[0])
# linear to start of blend 2 (time 2 - t2/2 )
tlin=(2-0.5*tb[1])-tb[0]; qs2=q1e+vv[0]*tlin; P('start of blend2 q', qs2)
qmid=qs2+vv[0]*tb[1]+0.5*aa[1]*tb[1]**2; P('end of blend2 q', qmid, 'speed', vv[0]+aa[1]*tb[1])
tlin2=(4-tb[2])-(2+0.5*tb[1]); P('end q at start of last blend', qmid+vv[1]*tlin2, ' final', qmid+vv[1]*tlin2+vv[1]*tb[2]+0.5*aa[2]*tb[2]**2)

# ---------- I. extras used in the text ----------
# natural-ish spline with zero end velocities (clamped): tridiagonal for interior velocities, Q2 data
qs=np.array([10,50,20,40,45.]); ts=np.array([0,5,10,15,20.]); h=np.diff(ts); sl=np.diff(qs)/h
n=len(qs)-2; Am=np.zeros((n,n)); bm=np.zeros(n)
for i in range(1,n+1):
    Am[i-1,i-1]=2*(1/h[i-1]+1/h[i]); bm[i-1]=3*(sl[i-1]/h[i-1]+sl[i]/h[i])
    if i>1: Am[i-1,i-2]=1/h[i-1]
    if i<n: Am[i-1,i]=1/h[i]
P('spline interior velocities Q2 data', np.linalg.solve(Am,bm))
# two-segment check of the spline formula equals w
P('spline eq for 2 segments (10,40,20; T=1)', 3*(30+(-20))/4)
# Jacobian at circle start
q1,q2=ik(0.7,0.3,Lc1,Lc2); Jc=J(q1,q2,Lc1,Lc2); P('circle J(t=0)', Jc.round(4).tolist(), 'q12 deg', np.degrees(q1+q2))
c2=(0.3**2+0.3**2-Lc1**2-Lc2**2)/(2*Lc1*Lc2); P('circle t=2: cos q2', c2, 'sin q2', np.sqrt(1-c2*c2), '0.4 sin q2', Lc2*np.sqrt(1-c2*c2), '0.6+0.4 cos', Lc1+Lc2*c2,
  'atan2 deg', np.degrees(np.arctan2(Lc2*np.sqrt(1-c2*c2), Lc1+Lc2*c2)))
c2=(2.375**2*2-25)/24; s2=np.sqrt(1-c2*c2); P('RR mid: cos q2, sin q2, 3sin, 4+3cos, atan2 deg', c2,s2,3*s2,4+3*c2,np.degrees(np.arctan2(3*s2,4+3*c2)))
# LSPB normalised with a = 2*min (a=8, D=1,T=1)
tB,v=lspb(1,1,8); P('LSPB norm a=8: tB, v, mean/peak', tB, v, 1/v)
P('linear mean/peak 1, cubic', 1/1.5, 'quintic', 1/1.875)
# LSPB first-line check 90 deg, 1.5 s
P('mock Q4 qdot,q at t=2', 30, 25+30*1)
# peak accel quintic / cubic normalised
P('quintic peak accel 10/sqrt3', 10/np.sqrt(3))
P('hole example L1=1 L2=0.7: outer, inner', 1.7, 0.3)
# FK check of the T/2 point (text uses these terms)
q1=np.radians(-2.12840);q2=np.radians(124.86289)
P('T/2 FK terms 4c1,3c12,4s1,3s12', 4*np.cos(q1),3*np.cos(q1+q2),4*np.sin(q1),3*np.sin(q1+q2), ' sum', 4*np.cos(q1)+3*np.cos(q1+q2), 4*np.sin(q1)+3*np.sin(q1+q2))
P('Craig t3 = 2-sqrt(2.8), t1 = 2-sqrt(2.4)', 2-np.sqrt(2.8), 2-np.sqrt(2.4))
P('circle J^-1 v', np.linalg.solve(np.array([[-0.3,-0.3857],[0.7,0.1061]]),[0,0.31416]))

# ---------- J. small numbers quoted in the text ----------
P('rest-to-rest 20->80 in 3 s: a3', -2*60/27, ' wrong (T^2):', -2*60/9)
s_,w_=rule([10,30,70,80,60],[0,2,5,9,12]); c=cubic(30,70,w_[1],w_[2],3)
P('QA seg 2->5 a2,a3', c[2], c[3], ' accel at end', 2*c[2]+6*c[3]*3, ' start of next', 2*cubic(70,80,w_[2],w_[3],4)[2], ' jump', 2*cubic(70,80,w_[2],w_[3],4)[2]-(2*c[2]+6*c[3]*3))
c=cubic(70,80,w_[2],w_[3],4); P('QA seg 5->9 q(4), qd(4)', c[0]+c[1]*4+c[2]*16+c[3]*64, c[1]+2*c[2]*4+3*c[3]*16)
P('Q9 qdot coefficients', 2*0.609375, 3*0.5078125, ' qd(0.8)', 2*0.609375*0.8-3*0.5078125*0.64)
P('Q8 num checks q1(1), qd1(1), qdd1(1), q2(1), qd2(1)', 10+82.5-52.5, 165-157.5, 165-315, 40+7.5-75+47.5, 7.5-150+142.5)
P('E-spot: seg1 (0,22.5,T=2,D=40)', cubic(0,40,0,22.5,2), ' seg2', cubic(40,60,22.5,0,2), 'q1(2)', 0+18.75*4-4.375*8+0, ' q2(2)', 40+45-30+5)
P('E-spot qd1(2)', 2*18.75*2+3*(-4.375)*4, ' qdd1(2)', 2*18.75+6*(-4.375)*2)
P('Q8 variant (0,40,30) seg1 q,qd at 1', 97.5-57.5, 195-172.5)
P('T/2 time', (4.75*np.sqrt(2)/3)/2)
P('circle t=1: cos q2', (0.5**2+0.5**2-0.52)/0.48, ' q2 deg', np.degrees(np.arccos((0.5**2+0.5**2-0.52)/0.48)))
P('90 deg in 1.5 s: v at a_min', 2*90/1.5)
P('slide-form t0=1 quadratic roots', np.roots([300,-600,120-0.5*300*1]))
