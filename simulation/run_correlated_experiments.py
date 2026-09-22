"""Reproduce manuscript Figures 2--4 and Tables 1--4."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent
SUPPORT_SCRIPT = ROOT / "_model_support.py"
CERTIFICATE_SCRIPT = ROOT.parent / "certificate_checks" / "compute_manuscript_certificates.py"
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
MASTER_SEED = 734921
REPRESENTATIVE_SEED = 418637
REPRESENTATIVE_HORIZON = 6.0
R14_SUPPLEMENTARY_PATH_ID = 1575
R14_SUPPLEMENTARY_ENSEMBLE_SIZE = 5000
R14_SUPPLEMENTARY_MAX_TIME_S = 6.2


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path.name}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


support = _load_module("model_support", SUPPORT_SCRIPT)
cert = _load_module("manuscript_certificates", CERTIFICATE_SCRIPT)
ModelConfig = support.ModelConfig
BLUE = support.BLUE
ORANGE = support.ORANGE
GREEN = support.GREEN
GREY = support.GREY

RHO_VALUES = (0.0, 0.25, 0.50, 0.75, 1.0)
RHO_COLORS = {
    0.0: "#1F4E79",
    0.25: "#376A8C",
    0.5: "#5A7F9C",
    0.75: "#725F91",
    1.0: "#675080",
}
RHO_LINESTYLES = {
    0.0: "-",
    0.25: (0, (5, 1.5)),
    0.5: (0, (3, 1.5)),
    0.75: (0, (5, 1.5, 1.2, 1.5)),
    1.0: (0, (1.5, 1.2)),
}


def setup_style() -> None:
    """Set publication-safe typography and keep vector text editable."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.6,
            "lines.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
            "savefig.facecolor": "white",
        }
    )


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def export_figure(fig: plt.Figure, stem: str) -> None:
    """Export manuscript figures without PowerPoint-specific side products."""

    kwargs = {"bbox_inches": "tight", "pad_inches": 0.015}
    fig.savefig(FIG_DIR / f"{stem}.pdf", **kwargs)
    fig.savefig(FIG_DIR / f"{stem}.svg", **kwargs)
    fig.savefig(FIG_DIR / f"{stem}.tiff", dpi=600, **kwargs)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, **kwargs)
    plt.close(fig)


def c_squared(rho: float) -> float:
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1].")
    return 1.0 / (1.0 + rho)


def simulate_correlated_ensemble(
    cfg: ModelConfig,
    n_paths: int,
    seed: int,
    *,
    rho: float,
    alpha: float | None = None,
    gamma: float | None = None,
    pin_diagonal: Iterable[float] = (5.0, 5.0, 5.0, 5.0),
    epsilons: Iterable[float] | None = None,
    initial_error: np.ndarray | None = None,
) -> dict[str, object]:
    """Vectorized Euler-Maruyama simulation with correlated common signals.

    Every rho-level call consumes the same two base Gaussian arrays at every
    step. Reusing the seed therefore defines pathwise-paired rho comparisons.
    """

    alpha = cfg.alpha if alpha is None else float(alpha)
    gamma = cfg.gamma if gamma is None else float(gamma)
    q_matrix = support.pinned_matrix(pin_diagonal)
    eps_values = sorted(
        set(float(v) for v in (epsilons if epsilons is not None else [cfg.epsilon])),
        reverse=True,
    )
    smallest_eps = min(eps_values)
    hits = {eps: np.full(n_paths, np.nan) for eps in eps_values}
    j_qv_at_hit = {eps: np.full(n_paths, np.nan) for eps in eps_values}
    initial = (
        support.INITIAL_ERROR.copy()
        if initial_error is None
        else np.asarray(initial_error, dtype=float).copy()
    )
    if initial.shape != (cfg.n_followers,):
        raise ValueError("initial_error must have one scalar error per follower.")
    errors = np.repeat(initial[None, :], n_paths, axis=0)
    active = np.ones(n_paths, dtype=bool)
    numerical_failure = np.zeros(n_paths, dtype=bool)
    j_qv = np.zeros(n_paths)
    x0 = support.INITIAL_LEADER
    rng = np.random.default_rng(seed)
    sqrt_dt = math.sqrt(cfg.dt)
    c = math.sqrt(c_squared(rho))
    orthogonal_weight = math.sqrt(max(0.0, 1.0 - rho * rho))
    n_steps = int(round(cfg.horizon / cfg.dt))

    for step in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        dw1 = z1 * sqrt_dt
        dw2 = (rho * z1 + orthogonal_weight * z2) * sqrt_dt
        indices = np.flatnonzero(active)
        if indices.size == 0:
            break

        old_error = errors[indices]
        disagreement = old_error @ q_matrix
        u1 = alpha * support.sig_power(disagreement, cfg.beta)
        u2 = gamma * disagreement
        new_error = (
            old_error
            + support.error_drift(x0, old_error, cfg) * cfg.dt
            + c * (u1 * dw1[indices, None] + u2 * dw2[indices, None])
        )

        density_1 = np.sum(u1 * u1, axis=1)
        density_2 = np.sum(u2 * u2, axis=1)
        density_cross = np.sum(u1 * u2, axis=1)
        density_qv = c * c * (
            density_1 + density_2 + 2.0 * rho * density_cross
        )
        density_qv = np.maximum(density_qv, 0.0)
        j_qv[indices] += density_qv * cfg.dt

        finite = np.all(np.isfinite(new_error), axis=1)
        if not np.all(finite):
            failed = indices[~finite]
            numerical_failure[failed] = True
            active[failed] = False
            errors[failed] = np.inf

        finite_indices = indices[finite]
        if finite_indices.size:
            finite_error = new_error[finite]
            errors[finite_indices] = finite_error
            norms = np.linalg.norm(finite_error, axis=1)
            hit_time = (step + 1) * cfg.dt
            for eps in eps_values:
                selected_mask = np.isnan(hits[eps][finite_indices]) & (norms <= eps)
                selected = finite_indices[selected_mask]
                hits[eps][selected] = hit_time
                j_qv_at_hit[eps][selected] = j_qv[selected]
            absorbed = norms <= smallest_eps
            if np.any(absorbed):
                absorbed_indices = finite_indices[absorbed]
                errors[absorbed_indices] = 0.0
                active[absorbed_indices] = False

        x0 = float(x0 + support.drift(step * cfg.dt, x0, cfg) * cfg.dt)

    return {
        "hits": hits,
        "j_budget_at_hit": j_qv_at_hit,
        "j_budget": j_qv,
        "j_qv": j_qv,
        "terminal_norm": np.linalg.norm(errors, axis=1),
        "numerical_failure": numerical_failure,
        "q_matrix": q_matrix,
        "seed": seed,
        "n_paths": n_paths,
        "alpha": alpha,
        "beta": cfg.beta,
        "gamma": gamma,
        "rho": rho,
        "c_squared": c_squared(rho),
        "pin_diagonal": list(float(v) for v in pin_diagonal),
        "initial_error": initial.tolist(),
        "initial_V": float(initial @ q_matrix @ initial),
        "noise_mode": "correlated-common",
    }


def simulate_coupled_steps(
    cfg: ModelConfig,
    n_paths: int,
    seed: int,
    rho: float,
    step_sizes: Iterable[float],
) -> dict[float, dict[str, object]]:
    """Nested-grid correlated simulations coupled through the finest grid."""

    dts = tuple(sorted(set(float(v) for v in step_sizes), reverse=True))
    finest = min(dts)
    ratios = {dt: int(round(dt / finest)) for dt in dts}
    for dt, ratio in ratios.items():
        if not math.isclose(dt, ratio * finest, abs_tol=1.0e-14):
            raise ValueError("Step sizes must be integer multiples of the finest grid.")

    q_matrix = support.pinned_matrix((5.0, 5.0, 5.0, 5.0))
    levels: dict[float, dict[str, object]] = {}
    for dt in dts:
        levels[dt] = {
            "errors": np.repeat(support.INITIAL_ERROR[None, :], n_paths, axis=0),
            "active": np.ones(n_paths, dtype=bool),
            "failure": np.zeros(n_paths, dtype=bool),
            "hits": np.full(n_paths, np.nan),
            "x0": support.INITIAL_LEADER,
            "dw1": np.zeros(n_paths),
            "dw2": np.zeros(n_paths),
            "coarse_step": 0,
        }

    rng = np.random.default_rng(seed)
    sqrt_fine = math.sqrt(finest)
    c = math.sqrt(c_squared(rho))
    orthogonal_weight = math.sqrt(max(0.0, 1.0 - rho * rho))
    n_fine_steps = int(round(cfg.horizon / finest))
    for fine_step in range(n_fine_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        fine_dw1 = z1 * sqrt_fine
        fine_dw2 = (rho * z1 + orthogonal_weight * z2) * sqrt_fine
        for dt in dts:
            level = levels[dt]
            np.asarray(level["dw1"])[:] += fine_dw1
            np.asarray(level["dw2"])[:] += fine_dw2
            if (fine_step + 1) % ratios[dt] != 0:
                continue
            errors = np.asarray(level["errors"])
            active = np.asarray(level["active"])
            failure = np.asarray(level["failure"])
            hits = np.asarray(level["hits"])
            indices = np.flatnonzero(active)
            x0 = float(level["x0"])
            coarse_step = int(level["coarse_step"])
            if indices.size:
                old_error = errors[indices]
                disagreement = old_error @ q_matrix
                u1 = cfg.alpha * support.sig_power(disagreement, cfg.beta)
                u2 = cfg.gamma * disagreement
                new_error = (
                    old_error
                    + support.error_drift(x0, old_error, cfg) * dt
                    + c
                    * (
                        u1 * np.asarray(level["dw1"])[indices, None]
                        + u2 * np.asarray(level["dw2"])[indices, None]
                    )
                )
                finite = np.all(np.isfinite(new_error), axis=1)
                if not np.all(finite):
                    failed = indices[~finite]
                    failure[failed] = True
                    active[failed] = False
                    errors[failed] = np.inf
                finite_indices = indices[finite]
                if finite_indices.size:
                    finite_error = new_error[finite]
                    errors[finite_indices] = finite_error
                    crossed = np.linalg.norm(finite_error, axis=1) <= cfg.epsilon
                    selected = finite_indices[np.isnan(hits[finite_indices]) & crossed]
                    hits[selected] = (coarse_step + 1) * dt
                    absorbed = finite_indices[crossed]
                    errors[absorbed] = 0.0
                    active[absorbed] = False
            level["x0"] = float(x0 + support.drift(coarse_step * dt, x0, cfg) * dt)
            np.asarray(level["dw1"]).fill(0.0)
            np.asarray(level["dw2"]).fill(0.0)
            level["coarse_step"] = coarse_step + 1

    output: dict[float, dict[str, object]] = {}
    for dt in dts:
        level = levels[dt]
        output[dt] = {
            "hits": {cfg.epsilon: np.asarray(level["hits"])},
            "j_budget_at_hit": {cfg.epsilon: np.full(n_paths, np.nan)},
            "j_budget": np.zeros(n_paths),
            "terminal_norm": np.linalg.norm(np.asarray(level["errors"]), axis=1),
            "numerical_failure": np.asarray(level["failure"]),
            "rho": rho,
            "dt": dt,
        }
    return output


def figure_correlation_family(
    results: dict[float, dict[str, object]],
    summaries: list[dict[str, object]],
    ablation_results: dict[str, dict[str, object]],
    cfg: ModelConfig,
) -> None:
    """Correlation effect contrasted with the two one-map references."""

    # The CDF and mean are the two visual claims.  Paired changes and the
    # quadratic-variation budget are reported numerically in the manuscript,
    # where their small effect sizes are easier to compare than in weak panels.
    fig = plt.figure(figsize=(7.20, 2.65))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.34)
    ax_ecdf = fig.add_subplot(gs[0, 0])
    ax_mean = fig.add_subplot(gs[0, 1])
    axes = (ax_ecdf, ax_mean)

    finite_hits: dict[float, np.ndarray] = {}
    for rho in RHO_VALUES:
        values = np.asarray(results[rho]["hits"][cfg.epsilon], dtype=float)
        finite_hits[rho] = values[np.isfinite(values)]
    reference_hits = {
        method: np.asarray(result["hits"][cfg.epsilon], dtype=float)
        for method, result in ablation_results.items()
        if method in ("power", "linear")
    }
    all_distributions = list(finite_hits.values()) + [
        values[np.isfinite(values)] for values in reference_hits.values()
    ]
    xmax = math.ceil(max(float(np.quantile(v, 0.995)) for v in all_distributions) * 10.0) / 10.0
    for rho in RHO_VALUES:
        values = finite_hits[rho]
        x, y = support.ecdf(values)
        visible = x <= xmax
        endpoint = float(np.sum(visible)) / float(len(x))
        x_plot = np.concatenate(([0.0], x[visible], [xmax]))
        y_plot = np.concatenate(([0.0], y[visible], [endpoint]))
        is_endpoint = rho in (0.0, 1.0)
        ax_ecdf.step(
            x_plot,
            y_plot,
            where="post",
            color=RHO_COLORS[rho],
            ls=RHO_LINESTYLES[rho],
            lw=1.25 if is_endpoint else 0.72,
            alpha=1.0 if is_endpoint else 0.62,
            label=rf"$\rho={rho:g}$",
        )
        if is_endpoint:
            dkw = math.sqrt(math.log(2.0 / 0.05) / (2.0 * len(values)))
            ax_ecdf.fill_between(
                x_plot,
                np.maximum(0.0, y_plot - dkw),
                np.minimum(1.0, y_plot + dkw),
                step="post",
                color=RHO_COLORS[rho],
                alpha=0.08,
                lw=0,
            )
    for method, color, linestyle, label in (
        ("power", ORANGE, (0, (5, 2)), r"power-law only ($\rho=0$ ref.)"),
        ("linear", GREEN, (0, (1.5, 1.5)), r"linear only ($\rho=0$ ref.)"),
    ):
        values = reference_hits[method]
        values = values[np.isfinite(values)]
        x, y = support.ecdf(values)
        visible = x <= xmax
        endpoint = float(np.sum(visible)) / float(len(x))
        x_plot = np.concatenate(([0.0], x[visible], [xmax]))
        y_plot = np.concatenate(([0.0], y[visible], [endpoint]))
        ax_ecdf.step(
            x_plot,
            y_plot,
            where="post",
            color=color,
            ls=linestyle,
            lw=1.05,
            label=label,
        )
    ax_ecdf.set_xlim(0.0, xmax)
    ax_ecdf.set_ylim(0.0, 1.01)
    ax_ecdf.set_xlabel(r"Numerical hitting time $T_\varepsilon$ (s)")
    ax_ecdf.set_ylabel("Cumulative probability")
    ax_ecdf.legend(
        frameon=False,
        loc="lower right",
        ncol=2,
        handlelength=1.8,
        columnspacing=0.8,
        fontsize=5.8,
    )

    rho_array = np.array([float(row["rho"]) for row in summaries])
    means = np.array([float(row["mean_hit"]) for row in summaries])
    lows = np.array([float(row["mean_ci_low"]) for row in summaries])
    highs = np.array([float(row["mean_ci_high"]) for row in summaries])
    ax_mean.errorbar(
        rho_array,
        means,
        yerr=np.vstack((means - lows, highs - means)),
        color=BLUE,
        ecolor="#CC2F6C",
        marker="o",
        ms=3.6,
        capsize=3.5,
        capthick=1.0,
        lw=1.0,
    )
    ax_mean.set_xlabel(r"Signal correlation $\rho$")
    ax_mean.set_ylabel(r"Mean $T_\varepsilon$ (s)")
    ax_mean.set_xticks(RHO_VALUES)
    span = max(highs) - min(lows)
    ax_mean.set_ylim(min(lows) - 0.25 * span, max(highs) + 0.25 * span)

    for ax, label in zip(axes, "ab"):
        support.clean_axes(ax)
        support.panel_label(ax, label)
    export_figure(fig, "Fig3_correlation_family")


def figure_beta_correlation(
    beta_rows: list[dict[str, object]],
    cfg: ModelConfig,
) -> None:
    """Exponent sensitivity and worst-endpoint certificate slacks."""

    q_matrix = support.pinned_matrix((5.0, 5.0, 5.0, 5.0))
    fig = plt.figure(figsize=(7.20, 2.75))
    gs = fig.add_gridspec(1, 3, wspace=0.48)
    axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
    markers = {0.0: "o", 1.0: "s"}
    for rho in (0.0, 1.0):
        rows = sorted(
            [row for row in beta_rows if math.isclose(float(row["rho"]), rho)],
            key=lambda row: float(row["beta"]),
        )
        beta = np.array([float(row["beta"]) for row in rows])
        mean = np.array([float(row["mean_hit"]) for row in rows])
        low = np.array([float(row["mean_ci_low"]) for row in rows])
        high = np.array([float(row["mean_ci_high"]) for row in rows])
        certified = np.array([
            bool(cert.spectral_certificate(q_matrix, float(row["beta"]), rho)["certified"])
            for row in rows
        ])
        axes[0].plot(
            beta[certified],
            mean[certified],
            color=RHO_COLORS[rho],
            lw=0.9,
            alpha=0.85,
        )
        axes[0].errorbar(
            beta[certified],
            mean[certified],
            yerr=np.vstack((mean[certified] - low[certified], high[certified] - mean[certified])),
            color=RHO_COLORS[rho],
            marker=markers[rho],
            ms=3.2,
            capsize=2.8,
            lw=0.0,
            elinewidth=0.9,
            label=rf"$\rho={rho:g}$",
        )
        if np.any(~certified):
            axes[0].errorbar(
                beta[~certified],
                mean[~certified],
                yerr=np.vstack((mean[~certified] - low[~certified], high[~certified] - mean[~certified])),
                color=GREY,
                ecolor=GREY,
                marker=markers[rho],
                markerfacecolor="white",
                ms=3.2,
                capsize=2.8,
                lw=0.0,
                elinewidth=0.9,
            )
    axes[0].set_xlabel(r"Power-law exponent $\beta$")
    axes[0].set_ylabel(r"Mean $T_\varepsilon$ (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(
        Line2D(
            [],
            [],
            color=GREY,
            marker="o",
            markerfacecolor="white",
            lw=0.0,
            ms=3.2,
        )
    )
    labels.append("uncertified check")
    axes[0].legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        handlelength=1.4,
        fontsize=5.8,
    )

    beta_dense = np.linspace(0.82, 0.985, 260)
    for rho in (0.0, 0.5, 1.0):
        bounds = [
            float(cert.spectral_certificate(q_matrix, float(beta), rho)["mean_time_upper_bound_s"])
            for beta in beta_dense
        ]
        axes[1].plot(
            beta_dense,
            np.asarray(bounds),
            color=RHO_COLORS[rho],
            ls=RHO_LINESTYLES[rho],
            lw=0.9,
            label=rf"$\rho={rho:g}$",
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Power-law exponent $\beta$")
    axes[1].set_ylabel(r"Sufficient bound on $\mathbb{E}[T_0]$ (s)")
    axes[1].legend(frameon=False, loc="upper right", handlelength=1.5)

    certificate_rows = [
        cert.spectral_certificate(q_matrix, float(beta), 1.0)
        for beta in beta_dense
    ]
    power = np.array([float(row["power_slack"]) for row in certificate_rows])
    linear = np.array([float(row["linear_slack"]) for row in certificate_rows])
    cross = np.array([float(row["cross_slack"]) for row in certificate_rows])
    feasible = (power > 0.0) & (linear > 0.0) & (cross > 0.0)

    def first_positive_crossing(x: np.ndarray, y: np.ndarray) -> float:
        if y[0] > 0.0:
            return float(x[0])
        indices = np.flatnonzero((y[:-1] <= 0.0) & (y[1:] > 0.0))
        if len(indices) == 0:
            return math.inf
        index = int(indices[0])
        x0, x1 = float(x[index]), float(x[index + 1])
        y0, y1 = float(y[index]), float(y[index + 1])
        return x0 - y0 * (x1 - x0) / (y1 - y0)

    beta_star = max(
        first_positive_crossing(beta_dense, margin)
        for margin in (power, linear, cross)
    )
    if np.any(feasible) and math.isfinite(beta_star):
        axes[2].axvspan(
            beta_star,
            float(beta_dense[feasible][-1]),
            color="#E8EEF4",
            zorder=0,
        )
        axes[2].axvline(beta_star, color=GREY, lw=0.65, ls=(0, (2, 2)))
    axes[2].axhline(0.0, color="black", lw=0.55, ls=(0, (2, 2)))
    axes[2].plot(beta_dense, power, color=BLUE, lw=0.9, label="power")
    axes[2].plot(beta_dense, linear, color=ORANGE, lw=0.9, ls=(0, (4, 2)), label=r"linear ($\rho=1$)")
    axes[2].plot(beta_dense, cross, color=GREEN, lw=0.9, ls=(0, (1.5, 1.5)), label="cross")
    axes[2].set_xlabel(r"Power-law exponent $\beta$")
    axes[2].set_ylabel("Certificate slack")
    axes[2].legend(frameon=False, loc="upper left", handlelength=1.5)
    if math.isfinite(beta_star):
        axes[2].annotate(
            r"$\beta^\star$",
            xy=(beta_star, 0.12),
            xycoords=("data", "axes fraction"),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            color=GREY,
            rotation=90,
        )
    axes[2].text(0.98, 0.05, "certified", transform=axes[2].transAxes, ha="right", color=BLUE)

    for ax, label in zip(axes, "abc"):
        support.clean_axes(ax)
        support.panel_label(ax, label)
    export_figure(fig, "Fig4_power_exponent_certificate_sensitivity")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

TABLE3_FIELDS = ("n", "hits", "mean_hit", "mean_ci_low", "mean_ci_high")


def _select(source: dict[str, object], fields: tuple[str, ...]) -> dict[str, object]:
    return {field: source[field] for field in fields}


def run(quick: bool = False) -> None:
    setup_style()
    ensure_dirs()
    support.DATA_DIR = DATA_DIR
    support.FIG_DIR = FIG_DIR
    support.export_figure = export_figure
    cfg = ModelConfig()
    n_primary = 500 if quick else 5000
    n_secondary = 300 if quick else 2000
    n_beta = 250 if quick else 1500

    eps_values = (1.0e-2, 1.0e-3, 1.0e-4)
    rho_results: dict[float, dict[str, object]] = {}
    rho_summaries: list[dict[str, object]] = []
    for rho in RHO_VALUES:
        result = simulate_correlated_ensemble(
            cfg, n_primary, MASTER_SEED, rho=rho, epsilons=eps_values
        )
        rho_results[rho] = result
        rho_summaries.append(
            {"rho": rho, **_select(support.summarize(result, cfg.epsilon), TABLE3_FIELDS)}
        )

    path_rows: list[dict[str, object]] = []
    for index in range(n_primary):
        row: dict[str, object] = {"path_id": index + 1, "seed": MASTER_SEED}
        for rho in RHO_VALUES:
            key = f"rho_{rho:.2f}".replace(".", "p")
            hit = np.asarray(rho_results[rho]["hits"][cfg.epsilon], dtype=float)[index]
            row[f"hit_{key}_s"] = float(hit) if np.isfinite(hit) else ""
            row[f"j_at_hit_{key}"] = float(
                np.asarray(rho_results[rho]["j_budget_at_hit"][cfg.epsilon])[index]
            )
        path_rows.append(row)
    support.write_csv(DATA_DIR / "rho_paths.csv", path_rows)

    threshold_rows: list[dict[str, object]] = []
    for rho in (0.0, 1.0):
        for eps in eps_values:
            threshold_rows.append(
                {
                    "rho": rho,
                    "epsilon": eps,
                    **_select(support.summarize(rho_results[rho], eps), TABLE3_FIELDS),
                }
            )
    support.write_csv(DATA_DIR / "threshold_sensitivity_by_rho.csv", threshold_rows)

    ablation_results: dict[str, dict[str, object]] = {"full": rho_results[0.0]}
    for method, alpha, gamma in (("power", 1.0, 0.0), ("linear", 0.0, 1.0)):
        ablation_results[method] = simulate_correlated_ensemble(
            cfg, n_primary, MASTER_SEED, rho=0.0, alpha=alpha, gamma=gamma
        )
    ablation_path_rows: list[dict[str, object]] = []
    for index in range(n_primary):
        row = {"path_id": index + 1, "seed": MASTER_SEED}
        for method in ("power", "linear"):
            hit = np.asarray(ablation_results[method]["hits"][cfg.epsilon], dtype=float)[index]
            row[f"hit_{method}_s"] = float(hit) if np.isfinite(hit) else ""
            row[f"j_at_hit_{method}"] = (
                float(np.asarray(ablation_results[method]["j_qv"])[index])
                if np.isfinite(hit)
                else ""
            )
        ablation_path_rows.append(row)
    support.write_csv(DATA_DIR / "ablation_paths.csv", ablation_path_rows)

    step_sizes = (2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4)
    step_rows: list[dict[str, object]] = []
    for rho in (0.0, 1.0):
        coupled = simulate_coupled_steps(
            cfg, n_secondary, MASTER_SEED + 100, rho, step_sizes
        )
        finest_hits = np.asarray(coupled[min(step_sizes)]["hits"][cfg.epsilon], dtype=float)
        for dt in step_sizes:
            result = coupled[dt]
            hits = np.asarray(result["hits"][cfg.epsilon], dtype=float)
            mask = np.isfinite(hits) & np.isfinite(finest_hits)
            mean, low, high = support.mean_ci(hits[mask] - finest_hits[mask])
            step_rows.append(
                {
                    "rho": rho,
                    "dt_s": dt,
                    "reference_dt_s": min(step_sizes),
                    "paired_mean_difference_s": mean,
                    "paired_difference_ci_low_s": low,
                    "paired_difference_ci_high_s": high,
                    **_select(support.summarize(result, cfg.epsilon), TABLE3_FIELDS),
                }
            )
    support.write_csv(DATA_DIR / "step_sensitivity_by_rho.csv", step_rows)

    beta_values = (0.84, 0.86, 0.88, 0.90, 11.0 / 12.0, 0.94, 0.96, 0.98)
    beta_rows: list[dict[str, object]] = []
    for idx, beta in enumerate(beta_values):
        local_cfg = replace(cfg, beta=beta)
        for rho in (0.0, 1.0):
            result = simulate_correlated_ensemble(
                local_cfg, n_beta, MASTER_SEED + 300 + idx, rho=rho
            )
            beta_rows.append(
                {
                    "beta": beta,
                    "rho": rho,
                    **_select(support.summarize(result, local_cfg.epsilon), TABLE3_FIELDS),
                }
            )
    support.write_csv(DATA_DIR / "beta_rho_sweep.csv", beta_rows)

    representative_cfg = replace(cfg, horizon=REPRESENTATIVE_HORIZON)
    representative = support.simulate_representative(
        representative_cfg,
        REPRESENTATIVE_SEED,
        {
            "full": (1.0, 1.0),
            "power": (1.0, 0.0),
            "linear": (0.0, 1.0),
            "none": (0.0, 0.0),
        },
    )
    support.export_representative_csv(representative)
    support.figure_representative(representative, representative_cfg)
    figure_correlation_family(rho_results, rho_summaries, ablation_results, cfg)
    figure_beta_correlation(beta_rows, cfg)

    metadata = {
        "model": asdict(cfg),
        "master_seed": MASTER_SEED,
        "representative_seed": REPRESENTATIVE_SEED,
        "representative_simulation_horizon_s": REPRESENTATIVE_HORIZON,
        "representative_archive_window_s": support.REPRESENTATIVE_ARCHIVE_WINDOW_S,
        "rho_values": list(RHO_VALUES),
        "primary_n_per_rho": n_primary,
        "step_n_per_endpoint": n_secondary,
        "beta_n_per_setting": n_beta,
        "initial_leader": support.INITIAL_LEADER,
        "initial_error": support.INITIAL_ERROR.tolist(),
        "laplacian": support.LAPLACIAN.tolist(),
        "full_pinning": [5.0, 5.0, 5.0, 5.0],
        "solver": "Euler-Maruyama with threshold-triggered numerical absorption",
        "correlation_construction": "B1=Z1, B2=rho*Z1+sqrt(1-rho^2)*Z2, c_rho=(1+rho)^(-1/2)",
        "pairing": "All rho levels reuse the same base Gaussian innovations and path identifiers",
        "restricted_time": "tilde_T_epsilon=min(T_epsilon,H); a missing crossing is assigned H only for restricted summaries and is not recorded as a hitting time",
        "budget": "Integral of c_rho^2*(||u1||^2+||u2||^2+2*rho*u1^T*u2) up to the recorded threshold-hitting time; j_at_hit is empty when no crossing is recorded by H",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": mpl.__version__,
        "quick_mode": quick,
    }
    (DATA_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def replot_existing() -> None:
    """Regenerate Figures 2--4 from the archived data."""

    setup_style()
    ensure_dirs()
    support.DATA_DIR = DATA_DIR
    support.FIG_DIR = FIG_DIR
    support.export_figure = export_figure
    cfg = ModelConfig()
    representative_cfg = replace(cfg, horizon=REPRESENTATIVE_HORIZON)

    rep_rows = _read_csv_rows(DATA_DIR / "representative_trajectories.csv")
    rep: dict[str, np.ndarray] = {
        "time": np.array([float(row["time_s"]) for row in rep_rows]),
        "leader": np.array([float(row["leader"]) for row in rep_rows]),
        "hit_time_full_s": np.array(float(rep_rows[0]["hit_time_full_s"])),
        "state_full": np.array(
            [[float(row[f"state_full_{i}"]) for i in range(1, 5)] for row in rep_rows]
        ),
    }
    for method in ("full", "power", "linear", "none"):
        rep[f"error_{method}"] = np.array(
            [[float(row[f"error_{method}_{i}"]) for i in range(1, 5)] for row in rep_rows]
        )
    support.figure_representative(rep, representative_cfg)

    path_rows = _read_csv_rows(DATA_DIR / "rho_paths.csv")
    rho_results: dict[float, dict[str, object]] = {}
    rho_summaries: list[dict[str, object]] = []
    for rho in RHO_VALUES:
        key = f"rho_{rho:.2f}".replace(".", "p")
        values = np.array(
            [float(row[f"hit_{key}_s"]) if row[f"hit_{key}_s"] else math.nan for row in path_rows]
        )
        rho_results[rho] = {"hits": {cfg.epsilon: values}}
        mean, low, high = support.mean_ci(values)
        rho_summaries.append(
            {"rho": rho, "mean_hit": mean, "mean_ci_low": low, "mean_ci_high": high}
        )

    ablation_rows = _read_csv_rows(DATA_DIR / "ablation_paths.csv")
    ablation_results: dict[str, dict[str, object]] = {}
    for method in ("power", "linear"):
        values = np.array(
            [float(row[f"hit_{method}_s"]) if row[f"hit_{method}_s"] else math.nan for row in ablation_rows]
        )
        ablation_results[method] = {"hits": {cfg.epsilon: values}}
    figure_correlation_family(rho_results, rho_summaries, ablation_results, cfg)

    beta_rows: list[dict[str, object]] = list(
        _read_csv_rows(DATA_DIR / "beta_rho_sweep.csv")
    )
    figure_beta_correlation(beta_rows, cfg)


def run_reduced_pinning_case() -> None:
    """Regenerate the reduced-pinning-gain path data used in Table 4."""

    ensure_dirs()
    support.DATA_DIR = DATA_DIR
    cfg = ModelConfig()
    result = simulate_correlated_ensemble(
        cfg,
        5000,
        MASTER_SEED,
        rho=1.0,
        pin_diagonal=(3.0, 3.0, 3.0, 3.0),
    )
    hits = np.asarray(result["hits"][cfg.epsilon], dtype=float)
    rows = [
        {
            "path_id": index + 1,
            "seed": MASTER_SEED,
            "rho": 1.0,
            "pin_gain": 3.0,
            "hit_s": float(hits[index]) if np.isfinite(hits[index]) else "",
        }
        for index in range(hits.size)
    ]
    support.write_csv(DATA_DIR / "reduced_pinning_paths.csv", rows)
    summary = support.summarize(result, cfg.epsilon)
    print(
        json.dumps(
            {
                "n": summary["n"],
                "hits": summary["hits"],
                "mean_hit": summary["mean_hit"],
                "mean_ci_low": summary["mean_ci_low"],
                "mean_ci_high": summary["mean_ci_high"],
                "median_hit": summary["median_hit"],
                "q25_hit": summary["q25_hit"],
                "q75_hit": summary["q75_hit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_r14_supplementary_check() -> None:
    """Replay and continue the single power-law-only path discussed in R1.4.

    The generator consumes the original 5000-path Gaussian blocks at every
    time step and selects only path 1575. This preserves the exact random
    stream used by the archived ablation while avoiding any write to the
    H=6 s primary data files.
    """

    cfg = ModelConfig()
    if not math.isclose(cfg.horizon, 6.0):
        raise RuntimeError("The R1.4 check requires the primary H=6 s protocol.")

    path_index = R14_SUPPLEMENTARY_PATH_ID - 1
    rng = np.random.default_rng(MASTER_SEED)
    sqrt_dt = math.sqrt(cfg.dt)
    q_matrix = support.pinned_matrix((5.0, 5.0, 5.0, 5.0))
    error = support.INITIAL_ERROR.copy()
    x0 = support.INITIAL_LEADER
    first_hit = math.nan
    n_steps = int(round(R14_SUPPLEMENTARY_MAX_TIME_S / cfg.dt))

    for step in range(n_steps):
        z1 = rng.standard_normal(R14_SUPPLEMENTARY_ENSEMBLE_SIZE)
        rng.standard_normal(R14_SUPPLEMENTARY_ENSEMBLE_SIZE)
        disagreement = error @ q_matrix
        u1 = cfg.alpha * support.sig_power(disagreement, cfg.beta)
        error = (
            error
            + support.error_drift(x0, error, cfg) * cfg.dt
            + u1 * (z1[path_index] * sqrt_dt)
        )
        hit_time = (step + 1) * cfg.dt
        if np.linalg.norm(error) <= cfg.epsilon:
            first_hit = hit_time
            break
        x0 = float(x0 + support.drift(step * cfg.dt, x0, cfg) * cfg.dt)

    if not np.isfinite(first_hit):
        raise RuntimeError(
            f"Path {R14_SUPPLEMENTARY_PATH_ID} did not cross by "
            f"{R14_SUPPLEMENTARY_MAX_TIME_S:g} s."
        )
    if first_hit <= cfg.horizon:
        raise RuntimeError("The supplementary path unexpectedly crossed within H=6 s.")

    print(
        json.dumps(
            {
                "check": "R1.4 supplementary continuation",
                "path_id": R14_SUPPLEMENTARY_PATH_ID,
                "ensemble_size": R14_SUPPLEMENTARY_ENSEMBLE_SIZE,
                "master_seed": MASTER_SEED,
                "primary_horizon_s": cfg.horizon,
                "recorded_crossing_by_primary_horizon": False,
                "continued_hitting_time_s": round(float(first_hit), 10),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a reduced smoke test.")
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Regenerate manuscript Figures 2--4 from existing CSV files.",
    )
    parser.add_argument(
        "--reduced-pinning-only",
        action="store_true",
        help="Regenerate only the reduced-pinning-gain data for Table 4.",
    )
    parser.add_argument(
        "--r14-supplementary-check",
        action="store_true",
        help="Replay and continue the single R1.4 power-law-only path without writing data.",
    )
    args = parser.parse_args()
    if args.r14_supplementary_check:
        run_r14_supplementary_check()
    elif args.replot_only:
        replot_existing()
    elif args.reduced_pinning_only:
        run_reduced_pinning_case()
    else:
        run(quick=args.quick)


if __name__ == "__main__":
    main()
