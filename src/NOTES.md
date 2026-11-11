# Slides vs textbook / errata log (shown on the site in "Slides vs textbook" boxes)

Status: OPEN = flagged by a digest agent, not yet verified by Opus; VERIFIED = checked against the slide image and/or a sympy script; REJECTED = not an error.

## W2
- VERIFIED (nb cell 35): `E5 = rtb.ET.tz(-30)` should be `tz(-0.30)` (link lengths are in metres). The printed result (-18.17, 10.95, -20.71) is therefore wrong. checks/w02.py gives the correct pose.
- VERIFIED (nb cell 46, filled-in answer to Q3 "two frames"): `ty(5) tx(3) Ry(90°) Rx(90°)` is missing `tz(2)` and the final rotation must be `Rx(-90°)`. checks/w02.py.
- Furuta DH table (p. 34) ends at frame {3}; {3} ≠ {E}, so ³T_E = Rz(-90°) is needed. checks/w02.py.

## W13
- p. 11: the four-bar relation is printed as φ1 − φ = φ3 − π. That form is inconsistent with the position equations on the same slide, which need φ3 = φ − φ1 + π. To verify against the rendered slide.

## W1 (digest agent flags; OPEN)
- "Redundant" is defined as controllable DoF > total DoF, but a 7-DoF arm is cited as the example.
- Stewart platform: no count on the slides; the standard result is M = 6 (to be checked).
- p. 49 G–K failure example: 1 DoF only because the parallel links are equal.

## W3 (OPEN)
- p. 36: cos(π − q3) = (r² + (z−H)² − F² − G²)/(2FG). By the cosine rule the RHS is cos q3, so the sign looks wrong.
- p. 54 "unique solution" contradicts the multiple solutions on pp. 17 and 20.

## W4 (OPEN)
- p. 65: the tangent-half-angle result for δ appears wrong. The agent gives δ = 2·atan[(A₂C₁−A₁C₂)/((B₁C₂−B₂C₁)+(A₁B₂−A₂B₁))].
- p. 29: handwritten D(N−J+1) contradicts the printed D(N−J−1).
- 4RRR example: N=10, J=12, ΣF=12, Σu=5, R_C=7, R_M=0 → M=1 (to be checked).

## W5 (OPEN)
- pp. 33–34 / nb cell 35: the blend equations look valid only for t0 = 0.
- p. 37 blend example: no final answer on the slide; the agent gets t_b ≈ 0.2254 s, v ≈ 67.62 °/s (to be checked).
- p. 59: q2 cubic fit is −0.4043t² + 0.9036t on the slide; the agent's polyfit gives −0.4032t² + 0.9029t.
