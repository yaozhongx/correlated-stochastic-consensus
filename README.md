# Reproducibility Package for Manuscript CSR-2026-08-0121

## Manuscript

**Finite-Time Leader--Follower Consensus via Correlated Stochastic Diffusion Design in Nonlinear Multi-Agent Systems**

Repository: <https://github.com/yaozhongx/correlated-stochastic-consensus>

This package provides the archived numerical data and Python code supporting manuscript Figures 2--4 and Tables 1--4.

## Environment

The archived results were generated with Python 3.12.13, NumPy 2.5.1, and Matplotlib 3.11.1 on Windows 11. Install the dependencies from the package root:

```text
python -m pip install -r requirements.txt
```

All commands use paths relative to the package root.

## Contents

```text
CSR-2026-08-0121_Reproducibility_Package/
|-- README.md
|-- requirements.txt
|-- certificate_checks/
|   `-- compute_manuscript_certificates.py
`-- simulation/
    |-- run_correlated_experiments.py
    |-- _model_support.py
    `-- data/
        |-- representative_trajectories.csv
        |-- rho_paths.csv
        |-- ablation_paths.csv
        |-- beta_rho_sweep.csv
        |-- step_sensitivity_by_rho.csv
        |-- threshold_sensitivity_by_rho.csv
        |-- reduced_pinning_paths.csv
        `-- metadata.json
```

`run_correlated_experiments.py` is the executable workflow. `_model_support.py` contains the model equations, summary statistics, and Figure 2 plotting utilities used by that workflow.

## Replotting Figures 2--4

From the package root, run:

```text
python simulation/run_correlated_experiments.py --replot-only
```

The command reads the archived CSV files and writes PDF, SVG, TIFF, and PNG versions of:

```text
Fig2_representative_ablation
Fig3_correlation_family
Fig4_power_exponent_certificate_sensitivity
```

## Running the simulations

A reduced smoke run is available with:

```text
python simulation/run_correlated_experiments.py --quick
```

The complete Figures 2--4 and Tables 1--3 workflow is:

```text
python simulation/run_correlated_experiments.py
```

The Table 4 reduced-pinning-gain paths are regenerated with:

```text
python simulation/run_correlated_experiments.py --reduced-pinning-only
```

Simulation commands update the corresponding files in `simulation/data`.

## R1.4 supplementary continuation check

The primary coefficient-map ablation remains the original `H=6 s` experiment. In `ablation_paths.csv`, power-law-only path `1575` therefore has an empty `hit_power_s` field and is retained in the 5000-path restricted mean with `restricted_time=6 s`. The empty field denotes that no crossing was recorded within the primary horizon; it is not replaced by the later crossing time.

As a supplementary robustness check, the same path can be replayed with the original parameters, master seed, and 5000-path Gaussian-stream layout and then continued beyond the primary horizon. Run:

```text
python -B simulation/run_correlated_experiments.py --r14-supplementary-check
```

The command recomputes the continued crossing at `T_epsilon=6.1515 s` and prints the result as JSON. It does not write or modify any CSV file and does not create a second ablation dataset.

## Certificate calculation

Run:

```text
python certificate_checks/compute_manuscript_certificates.py
```

The script evaluates the manuscript formulas and prints:

- full-pinning power slack;
- full-pinning linear slack at `rho=0` and `rho=1`;
- full-pinning cross slack;
- full-pinning mean-time upper bounds at `rho=0` and `rho=1`;
- reduced-pinning separated spectral power slack;
- reduced-pinning coupled power margin;
- reduced-pinning exact linear margin;
- reduced-pinning `d_1`;
- reduced-pinning mean-time upper bound.

The linear slack is evaluated as

```text
c_rho^2 * (2*beta*gamma^2*lambda_min^2 - gamma^2*lambda_max^2) - 2*theta
```

with `c_rho^2 = 1/(1+rho)`.

## Figure and table mapping

| Manuscript item | Data source | Code |
|---|---|---|
| Figure 2 | `representative_trajectories.csv` | `_model_support.py` |
| Figure 3 | Hitting times in `rho_paths.csv` and `ablation_paths.csv` | `run_correlated_experiments.py` |
| Figure 4 | `beta_rho_sweep.csv`; formula values from `compute_manuscript_certificates.py` | `run_correlated_experiments.py` |
| Table 1 | Full-map `rho=0` hitting-time and quadratic-variation columns in `rho_paths.csv`; power-only and linear-only columns in `ablation_paths.csv` | `run_correlated_experiments.py` |
| Table 2 | All hitting-time and quadratic-variation columns in `rho_paths.csv` | `run_correlated_experiments.py` |
| Table 3 | `step_sensitivity_by_rho.csv` and `threshold_sensitivity_by_rho.csv` | `run_correlated_experiments.py` |
| Table 4 | `reduced_pinning_paths.csv`; formula values from `compute_manuscript_certificates.py` | Both Python files |
Each path-level file uses one row per fixed-seed paired path. `metadata.json` records the model, seeds, solver, sample sizes, correlation construction, and simulation horizon. The Figure 2 representative path is simulated under the same `6 s` horizon, while `representative_trajectories.csv` archives only the plotted `0--2 s` window. The finite-horizon convention for the Table 1 ablation and its supplementary continuation check are specified below.

## Fixed numerical protocol

- Master seed: `734921`
- Representative-path seed: `418637`
- Correlation levels: `0`, `0.25`, `0.5`, `0.75`, and `1`
- Euler--Maruyama step size: `5e-4 s`
- Monte Carlo simulation horizon: `6 s`
- Figure 2 representative-path simulation horizon: `6 s`
- Figure 2 archived plotting window: `0--2 s`
- Numerical threshold: `1e-3`
- Primary sample size: 5000 paired paths per correlation level
- Brownian construction: `B1=Z1`, `B2=rho*Z1+sqrt(1-rho^2)*Z2`
- Normalization: `c_rho=(1+rho)^(-1/2)`
