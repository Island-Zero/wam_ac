"""Dimension-averaged distances between normalized continuous action predictions."""
import numpy as np

def action_readouts(left, right, horizon=10):
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or not 0 < horizon <= len(a):
        raise ValueError('Expected matching [H,D] arrays with sufficient horizon')
    a, b = a[:horizon].ravel(), b[:horizon].ravel()
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Nonfinite action prediction')
    rms = float(np.sqrt(np.mean((a-b)**2)))
    aa, bb = float(np.sqrt(np.mean(a*a))), float(np.sqrt(np.mean(b*b)))
    cosine = (0. if aa == bb == 0 else 1.) if aa*bb < 1e-24 else float(1-np.clip(np.mean(a*b)/(aa*bb),-1,1))
    return dict(rms=rms, relative_l2=rms/max(np.sqrt((aa*aa+bb*bb)/2),1e-8), cosine=cosine)
