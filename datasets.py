import numpy as np
import math


def awgn_like(y, snr_db, rng, return_var=False):
    # add white noise to achieve target SNR at signal y (power ratio signal/noise)
    p_sig = np.mean(y ** 2) + 1e-12
    snr_lin = 10 ** (snr_db / 10)
    p_noise = p_sig / snr_lin
    noise = rng.normal(0.0, np.sqrt(p_noise), size=y.shape)
    if return_var:
        return y + noise, float(p_noise)
    return y + noise


def gen_ar2_drifting(N=5000, a1_0=0.7, a2_0=-0.2, fm=0.0005, drift_amp=0.1, snr_db=20, seed=0, delta=None):
    # 如果外部传入了 delta，就用 delta 来控制漂移频率 fm
    if delta is not None:
        fm = delta

    rng = np.random.default_rng(seed)
    y = np.zeros(N)
    v = rng.normal(0.0, 1.0, size=N) * 0.05
    a1 = a1_0 + drift_amp * np.sin(2 * np.pi * fm * np.arange(N))
    a2 = a2_0 + drift_amp * np.cos(2 * np.pi * fm * np.arange(N))
    for n in range(2, N):
        y[n] = a1[n] * y[n - 1] + a2[n] * y[n - 2] + v[n]
    y_noisy, p_noise = awgn_like(y, snr_db, rng, return_var=True)
    # regression form x[n]=[y[n-1], y[n-2]]
    X = np.stack([np.roll(y_noisy, 1), np.roll(y_noisy, 2)], axis=1)
    d = y_noisy
    X[:2, :] = 0.0
    q_var = float((2 * np.pi * fm * drift_amp) ** 2 + 1e-6)
    return X, d, y, dict(a1=a1, a2=a2, noise_var=p_noise, q_var=q_var)


def gen_piecewise_fir(N=6000, M=16, change_every=2000, snr_db=20, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=N)
    segs = math.ceil(N / change_every)
    W = []
    for s in range(segs):
        w = rng.normal(0.0, 1.0, size=M)
        w /= np.linalg.norm(w) + 1e-12
        W.append(w)
    y_clean = np.zeros(N)
    for s in range(segs):
        w = W[s]
        start = s * change_every
        end = min((s + 1) * change_every, N)
        for n in range(start, end):
            xn = x[max(0, n - M + 1):n + 1][::-1]
            if len(xn) < M:
                xn = np.pad(xn, (0, M - len(xn)))
            y_clean[n] = np.dot(w, xn)
    d, p_noise = awgn_like(y_clean, snr_db, rng, return_var=True)
    X = np.zeros((N, M))
    for n in range(N):
        xn = x[max(0, n - M + 1):n + 1][::-1]
        if len(xn) < M:
            xn = np.pad(xn, (0, M - len(xn)))
        X[n] = xn
    return X, d, y_clean, dict(W=W, change_every=change_every, noise_var=p_noise, q_var=1e-4)


def gen_drifting_sinusoid(N=5000, fs=1000.0, A0=1.0, f0=5.0, dA=0.5, df=2.0, snr_db=20, seed=2):
    rng = np.random.default_rng(seed)
    n = np.arange(N)
    A = A0 + dA * np.sin(2 * np.pi * n / (0.4 * N))
    f = f0 + df * np.sin(2 * np.pi * n / (0.3 * N))
    y = A * np.sin(2 * np.pi * f * n / fs)
    d, p_noise = awgn_like(y, snr_db, rng, return_var=True)
    freqs = np.linspace(f0 - df, f0 + df, 8)
    X = np.stack(
        [np.sin(2 * np.pi * ff * n / fs) for ff in freqs] +
        [np.cos(2 * np.pi * ff * n / fs) for ff in freqs],
        axis=1
    )
    return X, d, y, dict(freqs=freqs, A=A, f=f, noise_var=p_noise, q_var=5e-4)