import numpy as np
import argparse, os, json
import torch
import time

from datasets import gen_ar2_drifting, gen_piecewise_fir, gen_drifting_sinusoid
from algorithms import LMS, NLMS, RLS, VSSLMS_MathewsXie, KalmanFilter
from metrics import mse_curve, steady_state_mse, steady_state_mse_db
from plots import plot_mse_curves, plot_bar, plot_step_size_curves

from mamba_filter import (
    DL_Mamba_StepSizeNet,
    DL_LSTM_StepSizeNet,
    DL_MLP_StepSizeNet,
    Mamba_NLMS
)


def run_one(X, d, algos, steady_window=500):

    N, M = X.shape
    tail_ratio = float(steady_window) / float(N)
    results = {}

    for name, make in algos.items():
        print(f"Running algorithm evaluation: {name}...")
        af = make(M)
        y_pred = np.zeros(N)
        mu_traj = np.zeros(N)

        start_time = time.time()
        for n in range(N):
            y_pred[n], e = af.step(X[n], d[n])
            mu_traj[n] = getattr(af, "last_mu", 0.0)
        end_time = time.time()

        avg_ms_per_step = ((end_time - start_time) * 1000.0) / N
        mse = mse_curve(y_pred, d)
        ss_lin = steady_state_mse(mse, tail_ratio=tail_ratio)
        ss_db = steady_state_mse_db(mse, tail_ratio=tail_ratio)

        results[name] = {
            "mse": mse,
            "mu": mu_traj,
            "steady": ss_lin,
            "steady_db": ss_db,
            "time_ms": avg_ms_per_step
        }
        print(f"  -> Steady-state MSE: {ss_lin:.5f} ({ss_db:.2f} dB) | Time per step: {avg_ms_per_step:.4f} ms")
    return results


def safe_load_weights(model, filepath, device):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filepath)
    for p in (filepath, full_path):
        if os.path.exists(p):
            model.load_state_dict(torch.load(p, map_location=device), strict=False)
            print(f"  Successfully loaded pre-trained weights: {p}")
            return True
    print(f"  Notice: Checkpoint '{filepath}' not found; using default initialization.")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['ar2', 'fir', 'sin'], default='ar2')
    parser.add_argument('--snr', type=float, default=20.0)
    parser.add_argument('--outdir', type=str, default='results_mamba')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device('cpu')

    print("=" * 60)
    print(f"Launching Non-Stationary Scenario Evaluation | Case: {args.case.upper()} | SNR: {args.snr} dB")
    print("=" * 60)

    # --- 1. Synthetic Non-Stationary Data Generation ---
    if args.case == 'ar2':
        X, d, y_clean, info = gen_ar2_drifting(snr_db=args.snr)
        abrupt_points = None
        # Baseline parameters matching Table 2 of Reference PDF (AR2, 20 dB)
        lms_mu, nlms_mu = 0.05, 0.50
        vss_cfg = dict(mu0=0.08, rho=0.005, mu_min=0.015, mu_max=0.35)
        rls_cfg = dict(lam=0.995, delta=10.0)
        kf_cfg = dict(q_var=1e-5, r_var=info.get("noise_var", 0.0024), p0=1.0)
        steady_win = 1000

    elif args.case == 'fir':
        X, d, y_clean, info = gen_piecewise_fir(snr_db=args.snr)
        abrupt_points = [2000, 4000]
        # Baseline parameters matching Table 2 of Reference PDF (FIR jumps, 15 dB)
        lms_mu, nlms_mu = 0.015, 0.45
        vss_cfg = dict(mu0=0.25, rho=0.02, mu_min=0.08, mu_max=0.55)
        rls_cfg = dict(lam=0.99, delta=10.0)
        kf_cfg = dict(q_var=5e-4, r_var=info.get("noise_var", 0.032), p0=1.0)
        steady_win = 1000

    else:  # 'sin'
        X, d, y_clean, info = gen_drifting_sinusoid(snr_db=args.snr)
        abrupt_points = None
        # Baseline parameters matching Table 2 of Reference PDF (Sinusoid, 25 dB)
        lms_mu, nlms_mu = 0.002, 0.32
        vss_cfg = dict(mu0=0.32, rho=0.01, mu_min=0.31, mu_max=0.35)
        rls_cfg = dict(lam=0.9995, delta=0.1)
        kf_cfg = dict(q_var=2e-3, r_var=info.get("noise_var", 0.005), p0=1.0)
        steady_win = 1500

    # --- 2. Initialize Step-Size Modulation Networks ---
    print("\nInitializing deep learning networks for ablation studies...")
    mamba_net = DL_Mamba_StepSizeNet(input_dim=3)
    safe_load_weights(mamba_net, 'best_mamba_weights.pth', device)
    mamba_net.eval()

    lstm_net = DL_LSTM_StepSizeNet(input_dim=3, hidden_dim=16)
    safe_load_weights(lstm_net, 'best_lstm_weights.pth', device)
    lstm_net.eval()

    mlp_net = DL_MLP_StepSizeNet(input_dim=3, seq_len=32, hidden_dim=32)
    safe_load_weights(mlp_net, 'best_mlp_weights.pth', device)
    mlp_net.eval()

    # --- 3. Construct Algorithm Evaluation Dictionary ---
    ALGOS_DICT = {
        "LMS": lambda M: LMS(M, mu=lms_mu),
        "NLMS": lambda M: NLMS(M, mu=nlms_mu),
        "VSS-LMS": lambda M: VSSLMS_MathewsXie(M, **vss_cfg),
        "RLS": lambda M: RLS(M, **rls_cfg),
        "Kalman": lambda M: KalmanFilter(M, **kf_cfg),
        "MLP-NLMS": lambda M: Mamba_NLMS(M, model=mlp_net, mu_min=0.001, mu_max=1.0),
        "LSTM-NLMS": lambda M: Mamba_NLMS(M, model=lstm_net, mu_min=0.001, mu_max=1.0),
        "Mamba-NLMS": lambda M: Mamba_NLMS(M, model=mamba_net, mu_min=0.001, mu_max=1.0, seq_len=32)
    }

    # --- 4. Run Online Adaptive Filtering Loop ---
    print("\nExecuting online iterative filtering...")
    results = run_one(X, d, ALGOS_DICT, steady_window=steady_win)

    # --- 5. Performance Visualization and Data Logging ---
    curves = {k: v["mse"] for k, v in results.items()}
    fig_curve = os.path.join(args.outdir, f'{args.case}_mse_iter_mamba.png')
    plot_mse_curves(
        curves,
        title=f'MSE Learning Curves ({args.case}, SNR={args.snr}dB)',
        ylabel='MSE (dB)',
        outfile=fig_curve,
        window_size=50,
        use_db=True
    )

    steady_vals = {k: v["steady"] for k, v in results.items()}
    fig_bar = os.path.join(args.outdir, f'{args.case}_steady_mamba.png')
    plot_bar(steady_vals, title=f'Steady-State MSE ({args.case}, SNR={args.snr}dB)', outfile=fig_bar)

    mu_curves = {k: v["mu"] for k, v in results.items()}
    fig_step = os.path.join(args.outdir, f'{args.case}_step_size_mamba.png')
    plot_step_size_curves(
        mu_curves,
        title=f'Dynamic Step-Size Evolution ({args.case}, SNR={args.snr}dB)',
        abrupt_changes=abrupt_points,
        outfile=fig_step
    )

    summary = {k: {
        "steady_mse": float(v["steady"]),
        "steady_mse_db": float(v["steady_db"]),
        "avg_ms_per_step": float(v["time_ms"])
    } for k, v in results.items()}

    json_path = os.path.join(args.outdir, f'{args.case}_summary_mamba.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Evaluation pipeline executed successfully.")
    print(f"  - Statistical metrics JSON:  {json_path}")
    print(f"  - dB Learning curve figure:  {fig_curve}")
    print(f"  - Steady-state bar figure:   {fig_bar}")
    print(f"  - Step-size evolution curve: {fig_step}")
    print("=" * 60)


if __name__ == '__main__':
    main()