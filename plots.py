import numpy as np
import matplotlib.pyplot as plt


def smooth_data(data, window_size=50):

    data = np.asarray(data, dtype=np.float64).copy()
    if len(data) > 5:
        # 找到第一个非零有效误差点，回填前几个初始化零点
        valid_idx = np.where(data > 1e-8)[0]
        if len(valid_idx) > 0 and valid_idx[0] > 0:
            first_valid = valid_idx[0]
            data[:first_valid] = data[first_valid]

    if window_size <= 1:
        return data
    pad_width = window_size - 1
    padded_data = np.pad(data, (pad_width, 0), mode='edge')
    smoothed = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    return smoothed


def plot_mse_curves(curves_dict, title='MSE Learning Curves', ylabel='MSE (dB)',
                    xlabel='Iteration', outfile=None, window_size=50, use_db=True):

    plt.figure(figsize=(7.2, 4.8))

    for name, mse in curves_dict.items():
        smoothed_mse = smooth_data(mse, window_size)
        lw = 1.6 if name in ('Mamba-NLMS', 'Kalman') else 1.0
        ls = '--' if name == 'Kalman' else '-'

        if use_db:
            mse_db = 10.0 * np.log10(np.maximum(smoothed_mse, 1e-6))
            plt.plot(mse_db, label=name, linewidth=lw, linestyle=ls)
        else:
            plt.plot(smoothed_mse, label=name, linewidth=lw, linestyle=ls)

    if not use_db:
        plt.yscale('log')
        ylabel = 'MSE'

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(ncol=2, fontsize=8.5)

    if outfile:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.close()


def plot_bar(values_dict, title='Steady-State MSE', ylabel='MSE', outfile=None):

    names = list(values_dict.keys())
    vals = [float(values_dict[k]) for k in names]
    plt.figure(figsize=(6.8, 4.6))
    bars = plt.bar(names, vals, color='#1f77b4', width=0.65)

    if 'Kalman' in names:
        bars[names.index('Kalman')].set_color('#ff7f0e')
    if 'Mamba-NLMS' in names:
        bars[names.index('Mamba-NLMS')].set_color('#2ca02c')

    plt.ylim(0.0, max(vals) * 1.15)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)

    if outfile:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.close()


def plot_step_size_curves(mu_curves_dict, title='Evolution of Dynamic Step-Size $\\mu[n]$',
                          ylabel='Dynamic Step-Size $\\mu[n]$', xlabel='Iteration',
                          abrupt_changes=None, outfile=None):

    plt.figure(figsize=(7.2, 4.6))
    target_order = ['NLMS', 'VSS-LMS', 'Kalman', 'LSTM-NLMS', 'Mamba-NLMS']
    colors = {
        'NLMS': '#0284C7',
        'VSS-LMS': '#0D9488',
        'Kalman': '#D97706',
        'LSTM-NLMS': '#7C3AED',
        'Mamba-NLMS': '#16A34A'
    }

    for name in target_order:
        if name not in mu_curves_dict:
            continue
        mu_seq = np.asarray(mu_curves_dict[name], dtype=np.float64)
        lw = 2.0 if name == 'Mamba-NLMS' else 1.2
        ls = '--' if name in ('NLMS', 'Kalman') else '-'
        lbl = 'Kalman (Equiv. Gain $\\mathbf{x}^T\\mathbf{k}[n]$)' if name == 'Kalman' else name
        plt.plot(mu_seq, label=lbl, color=colors.get(name, None), linewidth=lw, linestyle=ls)

    if abrupt_changes:
        for idx, cp in enumerate(abrupt_changes):
            lbl = 'System Abrupt Jump' if idx == 0 else None
            plt.axvline(x=cp, color='#DC2626', linestyle=':', linewidth=1.3, label=lbl)

    plt.ylim(-0.02, 1.05)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right', fontsize=8.5)

    if outfile:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.close()