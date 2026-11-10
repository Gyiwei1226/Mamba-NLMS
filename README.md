# Mamba-NLMS: Selective State-Space Step-Size Modulation for Non-Stationary Adaptive Filtering and Acoustic Echo Cancellation

Official PyTorch implementation of **Mamba-NLMS**, a closed-loop neural adaptive filtering framework that embeds a lightweight selective state-space model (Mamba) as a stability-constrained step-size modulator within the normalized least mean square (NLMS) algorithm.

This repository reproduces the network architecture, training protocols (**Section 6**), and all synthetic and real-speech acoustic echo cancellation (AEC) benchmarks, figures, and tables (**Section 7**).

---

## 1. Environment & Requirements

The experiments in the paper were conducted and verified under the following environment:
- **Python**: `3.11.0` (compatible with Python $\ge 3.9$)
- **PyTorch**: `2.1.0` with **CUDA `11.8`**
- **IDE / OS**: PyCharm `2023.2.1` (Linux / Windows)
- **Scientific Libraries**: `numpy`, `scipy`, `matplotlib`

```bash
# 1) Create and activate a virtual environment (optional)
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install PyTorch 2.1.0 (CUDA 11.8) and scientific dependencies
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy matplotlib
```

---

## 2. Repository Structure

- `mamba_filter.py` – Core implementation of the lightweight Mamba step-size modulator ($L=32, d_{\text{model}}=32, d_{\text{state}}=16, d_{\text{conv}}=4$), neural baselines (`MLP-NLMS`, `LSTM-NLMS`), the closed-loop `Mamba-NLMS` update rule, offline supervised meta-training loss $\mathcal{L}(\boldsymbol{\theta})$ (Eq. 21), and ground-truth-free self-supervised error-power loss $\mathcal{L}_{\text{self}}(\boldsymbol{\theta})$ (Eq. 22).
- `datasets.py` – Non-stationary trajectory generators:
  - **Scenario 1 (`ar2`)**: AR(2) continuous smooth pole drift ($M=2, N=5000, \text{SNR}=20\text{ dB}$).
  - **Scenario 2 (`fir`)**: Piecewise-constant FIR system with abrupt weight jumps at $n=2000$ and $n=4000$ ($M=8, N=5000, \text{SNR}=15\text{ dB}$).
  - **Scenario 3 (`sin`)**: Fast continuous sinusoidal weight modulation ($M=8, N=5000, \text{SNR}=25\text{ dB}$, out-of-distribution zero-shot test).
- `algorithms.py` – Classical and state-space baselines:
  - Fixed-step-size filters: `LMS` ($\mu=0.02$) and `NLMS` ($\mu=0.10, \epsilon=10^{-6}$).
  - Heuristic variable-step-size filter: `VSS-LMS` ($\alpha=0.97, \gamma=4.8\times 10^{-4}, \mu[n]\in [10^{-4}, 0.5]$).
  - Second-order $\mathcal{O}(M^2)$ estimators: `RLS` ($\lambda=0.995, \mathbf{P}[0]=10\mathbf{I}_M$) and the random-walk state-space `Kalman Filter` ($\mathbf{Q}=10^{-4}\mathbf{I}_M, R=\sigma_v^2$).
- `metrics.py` – Evaluation metrics: logarithmic MSE learning curves ($10\log_{10}(\text{MSE})$, in dB), steady-state MSE ($\text{mean} \pm \text{std}$ over $n \in [4001, 5000]$ across $20$ Monte Carlo runs), and time-varying Echo Return Loss Enhancement ($\text{ERLE}(n)$, in dB).
- `plots.py` – Publication-quality plotting utilities for logarithmic (dB) learning curves, upright steady-state MSE bar charts, dynamic step-size $\mu[n]$ trajectories, and dual-axis ablation figures.
- `run_experiments_mamba.py` – Command-line driver for evaluating all 8 competing algorithms on individual synthetic scenarios.
- `run_all_mamba.py` – Automated batch script executing Scenarios 1–3 and exporting all figures and JSON summaries.
- `run_delta_validation.py` – Empirical verification of steady-state MSE against non-stationary drift rate $\delta \in [0, 5 \times 10^{-4}]$ across `NLMS`, `RLS`, `Kalman Filter`, and `Mamba-NLMS` (**Fig. 5**).
- `run_seq_len_ablation.py` – Dual-axis ablation study of sliding window length $L \in \{16, 32, 64, 128\}$ on steady-state MSE and per-step execution latency (**Fig. 6**).
- `real AEC.py` – High-order ($M=256$, $16\text{ kHz}$, $\text{SNR}=32\text{ dB}$) real-speech acoustic echo cancellation benchmark using the ICASSP AEC Challenge corpus with an abrupt RIR cyclic shift and sign inversion at $n=58{,}000$, comparing `NLMS`, `Kalman Filter`, and `Mamba-NLMS` (**Fig. 7**).
- `results_mamba/` – Output directory for generated figures and quantitative JSON summaries.

---

## 3. Quick Start & Reproducing Paper Results

```bash
# 1) Run all three synthetic benchmarks (Scenarios 1-3: AR(2), FIR Jump, Sinusoidal Modulation)
python run_all_mamba.py

# 2) Or run individual synthetic scenarios via command line
python run_experiments_mamba.py --case ar2 --snr 20
python run_experiments_mamba.py --case fir --snr 15
python run_experiments_mamba.py --case sin --snr 25

# 3) Run theoretical drift-rate (delta) validation and sequence length (L) ablation (Figs. 5 & 6)
python run_delta_validation.py
python run_seq_len_ablation.py

# 4) Run real-speech Acoustic Echo Cancellation (AEC) evaluation (Fig. 7, M = 256, 16 kHz)
python "real AEC.py"
```

### Generated Figures & Metrics in `results_mamba/`
- `<case>_mse_iter_mamba.png` – Logarithmic MSE learning curves ($10\log_{10}(\text{MSE})$, in dB) across 5000 iterations (**Figs. 2(a), 3(a), 4(a)**).
- `<case>_steady_mamba.png` – Quantitative steady-state MSE bar chart across all 8 algorithms (**Figs. 2(b), 3(b), 4(b)**).
- `<case>_step_size_mamba.png` – Dynamic step-size $\mu[n]$ evolution trajectory (**Figs. 2(c), 3(c), 4(c)**).
- `<case>_summary_mamba.json` – High-precision numerical metrics (`steady_mse`, `steady_mse_db`, and `avg_ms_per_step`) for **Table 1**.
- `delta_robustness_validation.png` – Steady-state MSE vs. system drift rate $\delta$ (**Fig. 5**).
- `seq_len_ablation.png` – Dual-axis window length $L \in \{16, 32, 64, 128\}$ ablation chart (**Fig. 6**).
- `real_speech_erle_mamba.png` / `real_speech_erle_mamba.pdf` – Real-speech AEC $\text{ERLE}(n)$ ($\text{dB}$) comparison and synchronous $\mu[n]$ evolution (**Fig. 7**).

---

## 4. Summary of Quantitative Benchmark Results

| Algorithm | Asymptotic Complexity | Scenario 1: AR(2) Drift ($\text{SNR}=20\text{ dB}$) | Scenario 2: FIR Jump ($\text{SNR}=15\text{ dB}$) | Scenario 3: Sinusoid ($\text{SNR}=25\text{ dB}$) | Time / Step (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LMS** | $\mathcal{O}(M)$ | $0.00342 \pm 0.00014$ ($-24.65\text{ dB}$) | $0.03350$ ($-14.75\text{ dB}$) | $0.64707$ ($-1.89\text{ dB}$) | $\sim 0.004$ |
| **NLMS** | $\mathcal{O}(M)$ | $0.00725 \pm 0.00022$ ($-21.40\text{ dB}$) | $0.04209$ ($-13.76\text{ dB}$) | $0.12091$ ($-9.18\text{ dB}$) | $\sim 0.007$ |
| **VSS-LMS** | $\mathcal{O}(M)$ | $0.00336 \pm 0.00015$ ($-24.74\text{ dB}$) | $0.05085$ ($-12.94\text{ dB}$) | $0.12205$ ($-9.13\text{ dB}$) | $\sim 0.018$ |
| **RLS** | $\mathcal{O}(M^2)$ | $0.00238 \pm 0.00008$ ($-26.24\text{ dB}$) | $0.03341$ ($-14.76\text{ dB}$) | $0.78572$ ($-1.05\text{ dB}$) | $\sim 0.020$ |
| **Kalman Filter** | $\mathcal{O}(M^2)$ | $0.00236 \pm 0.00008$ ($-26.27\text{ dB}$) | $0.03425$ ($-14.65\text{ dB}$) | $0.06493$ ($-11.88\text{ dB}$) | $\sim 0.036$ |
| **MLP-NLMS** | $\mathcal{O}(M + d_{\text{mlp}}^2)$ | $0.00450 \pm 0.00018$ ($-23.47\text{ dB}$) | $0.04668$ ($-13.31\text{ dB}$) | $0.42517$ ($-3.71\text{ dB}$) | $\sim 0.063$ |
| **LSTM-NLMS** | $\mathcal{O}(M + L d_{\text{model}}^2)$ | $0.00270 \pm 0.00011$ ($-25.68\text{ dB}$) | $0.03416$ ($-14.66\text{ dB}$) | $0.11462$ ($-9.41\text{ dB}$) | $\sim 0.067$ |
| **Mamba-NLMS (Ours)** | $\mathcal{O}(M + L d_{\text{model}} d_{\text{state}})$ | **$0.00245 \pm 0.00009$ ($-26.11\text{ dB}$)** | **$0.03172$ ($-14.99\text{ dB}$)** | **$0.03822$ ($-14.18\text{ dB}$)** | $\sim 0.106$ ($\sim 7.256^*$) |

*$^*$Note: $\sim 0.106\text{ ms}$ is measured with lightweight tensor caching; $\sim 7.256\text{ ms}$ corresponds to unoptimized sample-by-sample PyTorch dynamic graph dispatch.*

---

## 5. Key Implementation Notes
- **Ground-Truth-Free Closed-Loop Inference**: During online deployment, the Mamba modulator ingests only the 3-dimensional observable feature vector $\mathbf{z}[n] = [e[n], \|\mathbf{x}[n]\|_2^2, \hat{y}[n]]^T \in \mathbb{R}^3$ without requiring access to the unknown true system weights $\mathbf{w}^*[n]$.
- **Linear $\mathcal{O}(M)$ Scalability**: Because the neural modulator input is decoupled from the filter tap order $M$, Mamba-NLMS preserves $\mathcal{O}(M)$ linear asymptotic complexity with respect to $M$, avoiding the $\mathcal{O}(M^2)$ covariance matrix propagation bottleneck of RLS and full-state Kalman filters in high-order acoustic applications ($M = 256$).
- **Lyapunov Stability Guarantee**: The Sigmoid output layer strictly bounds the dynamic step-size within $[\mu_{\min}, \mu_{\max}] \subset (0, 2)$ ($\mu_{\min}=10^{-4}, \mu_{\max}=1.0$), guaranteeing global mean-square convergence stability under bounded excitation and noise.
