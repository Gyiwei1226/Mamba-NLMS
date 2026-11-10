import os

# Standard evaluation scenarios configured according to the paper specifications
scenarios = [
    {"case": "ar2", "snr": 20},  # Smooth drift scenario governed by an AR(2) process
    {"case": "fir", "snr": 15},  # Abrupt parameter change scenario for FIR coefficients
    {"case": "sin", "snr": 25}   # Dynamic tracking under complex sinusoidal modulation
]

print("Initializing automated batch evaluation for the Mamba-NLMS framework...")
print("Target output directory: ./results_mamba\n")

for s in scenarios:
    case = s["case"]
    snr = s["snr"]

    print("=" * 60)
    print(f"Executing Scenario: {case.upper()} | SNR: {snr} dB")
    print("=" * 60)

    # Invoke the underlying experimental execution script with controlled arguments
    exit_code = os.system(f"python run_experiments_mamba.py --case {case} --snr {snr}")

    if exit_code != 0:
        print(f"Execution Failure: Scenario {case.upper()} terminated abnormally.")
        break

print("\nAll non-stationary evaluation benchmarks completed.")
print("The generated comparative performance figures have been successfully saved to './results_mamba'.")