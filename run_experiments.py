import numpy as np
import argparse, os, json

from datasets import gen_ar2_drifting, gen_piecewise_fir, gen_drifting_sinusoid
from algorithms import LMS, NLMS, RLS, VSSLMS_MathewsXie
from metrics import mse_curve, steady_state_mse
from plots import plot_mse_curves, plot_bar

ALGOS = {
    "LMS": lambda M: LMS(M, mu=0.01),
    "NLMS": lambda M: NLMS(M, mu=0.5),
    "RLS": lambda M: RLS(M, lam=0.99, delta=100.0),
    "VSS-LMS": lambda M: VSSLMS_MathewsXie(M, mu0=0.05, rho=0.005),
}

def run_one(X, d, algos):
    N, M = X.shape
    results = {}
    for name, make in algos.items():
        af = make(M)
        y_pred = np.zeros(N)
        for n in range(N):
            y_pred[n], e = af.step(X[n], d[n])
        mse = mse_curve(y_pred, d)
        results[name] = {"mse": mse, "steady": steady_state_mse(mse)}
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['ar2','fir','sin'], default='ar2')
    parser.add_argument('--snr', type=float, default=20.0)
    parser.add_argument('--outdir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.case == 'ar2':
        X, d, y_clean, info = gen_ar2_drifting(snr_db=args.snr)
    elif args.case == 'fir':
        X, d, y_clean, info = gen_piecewise_fir(snr_db=args.snr)
    else:
        X, d, y_clean, info = gen_drifting_sinusoid(snr_db=args.snr)

    results = run_one(X, d, ALGOS)

    # Plots
    curves = {k: v["mse"] for k,v in results.items()}
    fig1 = os.path.join(args.outdir, f'{args.case}_mse_iter.png')
    plot_mse_curves(curves, title=f'MSE vs Iterations ({args.case}, SNR={args.snr} dB)', outfile=fig1)

    # Bar of steady-state
    steady = {k: v["steady"] for k,v in results.items()}
    fig2 = os.path.join(args.outdir, f'{args.case}_steady.png')
    plot_bar(steady, title=f'Steady-State MSE ({args.case}, SNR={args.snr} dB)', outfile=fig2)

    # Save JSON summary
    summary = {k: {"steady_mse": float(v["steady"])} for k,v in results.items()}
    with open(os.path.join(args.outdir, f'{args.case}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(" -", fig1)
    print(" -", fig2)
    print(" -", os.path.join(args.outdir, f'{args.case}_summary.json'))

if __name__ == '__main__':
    main()
