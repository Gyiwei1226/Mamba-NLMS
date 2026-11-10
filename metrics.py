import numpy as np


def mse_curve(y_pred, d, eps=1e-12):
    """Computes the instantaneous squared error curve e^2[n] = (d[n] - y_pred[n])^2."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    return np.maximum((d - y_pred) ** 2, eps)


def mse_curve_db(y_pred, d, eps=1e-12):
    """
    [Addresses Reviewer 1, Comment 9]:
    Computes the MSE learning curve in logarithmic decibel (dB) scale:
        MSE_dB[n] = 10 * log10((d[n] - y_pred[n])^2 + eps)
    """
    return 10.0 * np.log10(mse_curve(y_pred, d, eps=eps))


def steady_state_mse(mse_arr, last_n=500, tail_ratio=0.2):
    """
    Computes the linear steady-state MSE over the final `last_n` samples (default: 500),
    strictly matching Section 7.4 & Table 2 of the reference PDF ('Average MSE computed
    after convergence, e.g., final 500 samples').
    """
    mse_arr = np.asarray(mse_arr, dtype=np.float64)
    if last_n is not None and len(mse_arr) > last_n:
        return float(np.mean(mse_arr[-int(last_n):]))
    start_idx = int(len(mse_arr) * (1.0 - tail_ratio))
    return float(np.mean(mse_arr[start_idx:]))


def steady_state_mse_db(mse_arr, last_n=500, tail_ratio=0.2, eps=1e-12):
    """Computes the steady-state MSE in dB: 10 * log10(steady_state_mse)."""
    ss_lin = steady_state_mse(mse_arr, last_n=last_n, tail_ratio=tail_ratio)
    return float(10.0 * np.log10(max(ss_lin, eps)))