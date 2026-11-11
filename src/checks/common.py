"""Shared helpers for the ENN581 check scripts: elementary transforms (sympy + numpy), proximal DH, reporting."""
import sympy as sp
import numpy as np

# ---------- sympy elementary transforms (column-vector convention, 4x4 homogeneous) ----------
def sRx(a):
    c, s = sp.cos(a), sp.sin(a)
    return sp.Matrix([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

def sRy(a):
    c, s = sp.cos(a), sp.sin(a)
    return sp.Matrix([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

def sRz(a):
    c, s = sp.cos(a), sp.sin(a)
    return sp.Matrix([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def sT(x=0, y=0, z=0):
    return sp.Matrix([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

def stx(d): return sT(x=d)
def sty(d): return sT(y=d)
def stz(d): return sT(z=d)

def sDH_prox(alpha_im1, a_im1, theta_i, d_i):
    """Craig proximal (modified) DH: Rx(alpha_{i-1}) tx(a_{i-1}) Rz(theta_i) tz(d_i)."""
    return sRx(alpha_im1) * stx(a_im1) * sRz(theta_i) * stz(d_i)

def sDH_dist(theta_i, d_i, a_i, alpha_i):
    """Standard (distal) DH: Rz(theta_i) tz(d_i) tx(a_i) Rx(alpha_i)."""
    return sRz(theta_i) * stz(d_i) * stx(a_i) * sRx(alpha_i)

def chain(*Ts):
    out = sp.eye(4)
    for T in Ts:
        out = out * T
    return out

# ---------- numpy versions ----------
def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])

def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])

def Tr(x=0, y=0, z=0):
    T = np.eye(4); T[:3, 3] = [x, y, z]; return T

def tx(d): return Tr(x=d)
def ty(d): return Tr(y=d)
def tz(d): return Tr(z=d)

def DH_prox(alpha_im1, a_im1, theta_i, d_i):
    return Rx(alpha_im1) @ tx(a_im1) @ Rz(theta_i) @ tz(d_i)

def nchain(*Ts):
    out = np.eye(4)
    for T in Ts:
        out = out @ T
    return out

deg = np.pi / 180

# ---------- reporting ----------
_fails = []
def check(name, ok, detail=''):
    tag = 'PASS' if ok else 'FAIL'
    print(f'[{tag}] {name}' + (f'  — {detail}' if detail else ''))
    if not ok:
        _fails.append(name)

def close(name, a, b, tol=1e-9, show=True):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ok = a.shape == b.shape and np.allclose(a, b, atol=tol)
    check(name, ok, (f'got {np.round(a, 6).tolist()} want {np.round(b, 6).tolist()}' if (show or not ok) else ''))
    return ok

def is_rotation(R, tol=1e-9):
    R = np.asarray(R, dtype=float)
    return np.allclose(R.T @ R, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1) < tol

def report():
    print('-' * 60)
    if _fails:
        print(f'{len(_fails)} FAILED: {_fails}')
        raise SystemExit(1)
    print('ALL CHECKS PASSED')

def r(x, n=4):
    """round for display"""
    return float(np.round(float(x), n))
