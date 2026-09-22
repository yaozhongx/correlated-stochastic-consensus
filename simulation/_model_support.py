"""Shared model and Figure 2 utilities for the public reproduction workflow."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
BLUE = "#1F4E79"
ORANGE = "#B66A2C"
GREEN = "#2F6F64"
GREY = "#5F6872"

LAPLACIAN = np.array(
    [
        [0.50, -0.25, 0.00, -0.25],
        [-0.25, 0.50, -0.25, 0.00],
        [0.00, -0.25, 0.50, -0.25],
        [-0.25, 0.00, -0.25, 0.50],
    ],
    dtype=float,
)
INITIAL_LEADER = 1.0
INITIAL_ERROR = np.array([3.0, 2.0, -6.0, -2.0], dtype=float)
REPRESENTATIVE_ARCHIVE_WINDOW_S = 2.0


@dataclass(frozen=True)
class ModelConfig:
    n_followers: int = 4
    beta: float = 11.0 / 12.0
    alpha: float = 1.0
    gamma: float = 1.0
    theta: float = 1.5
    drift_saturation: float = 1.5
    forcing_amplitude: float = 1.0
    forcing_frequency: float = 2.0
    # The main Monte Carlo protocol uses the first refined grid. A still finer
    # The 2.5e-4 s grid is used as the coupled-refinement reference.
    dt: float = 5.0e-4
    horizon: float = 6.0
    epsilon: float = 1.0e-3


def pinned_matrix(pin_diagonal: Iterable[float]) -> np.ndarray:
    pin = np.asarray(tuple(pin_diagonal), dtype=float)
    if pin.shape != (4,):
        raise ValueError("The current experiment is defined for four followers.")
    return LAPLACIAN + np.diag(pin)


def drift(t: float, x: np.ndarray | float, cfg: ModelConfig) -> np.ndarray | float:
    return cfg.drift_saturation * np.tanh(x) - cfg.forcing_amplitude * np.cos(
        cfg.forcing_frequency * t
    )


def error_drift(x0: float, error: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    # The common external forcing cancels exactly in f(t, x0 + e) - f(t, x0).
    return cfg.drift_saturation * (np.tanh(x0 + error) - np.tanh(x0))


def sig_power(values: np.ndarray, exponent: float) -> np.ndarray:
    return np.sign(values) * np.abs(values) ** exponent


def simulate_representative(
    cfg: ModelConfig,
    seed: int,
    methods: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    n_steps = int(round(cfg.horizon / cfg.dt))
    rng = np.random.default_rng(seed)
    dw1 = rng.standard_normal(n_steps) * math.sqrt(cfg.dt)
    dw2 = rng.standard_normal(n_steps) * math.sqrt(cfg.dt)
    q_matrix = pinned_matrix((5.0, 5.0, 5.0, 5.0))
    # Store the representative path at the actual integration resolution.
    # Earlier 5-ms decimation made the short Brownian transient look polygonal.
    stride = 1
    save_steps = np.arange(0, n_steps + 1, stride)
    times = save_steps * cfg.dt
    output: dict[str, np.ndarray] = {"time": times}

    leader = np.empty(n_steps + 1, dtype=float)
    leader[0] = INITIAL_LEADER
    for step in range(n_steps):
        leader[step + 1] = leader[step] + drift(step * cfg.dt, leader[step], cfg) * cfg.dt
    output["leader"] = leader[save_steps]

    for name, (alpha, gamma) in methods.items():
        error = INITIAL_ERROR.copy()
        shadow_error = INITIAL_ERROR.copy() if name == "full" else None
        stored_error = np.empty((len(save_steps), cfg.n_followers), dtype=float)
        stored_shadow = (
            np.empty((len(save_steps), cfg.n_followers), dtype=float)
            if name == "full"
            else None
        )
        stored_j = np.empty(len(save_steps), dtype=float)
        stored_error[0] = error
        if stored_shadow is not None:
            stored_shadow[0] = shadow_error
        stored_j[0] = 0.0
        total_j = 0.0
        absorbed = False
        first_hit = math.nan
        save_index = 1
        for step in range(n_steps):
            if shadow_error is not None:
                shadow_disagreement = q_matrix @ shadow_error
                shadow_u1 = alpha * sig_power(shadow_disagreement, cfg.beta)
                shadow_u2 = gamma * shadow_disagreement
                shadow_error = (
                    shadow_error
                    + error_drift(leader[step], shadow_error, cfg) * cfg.dt
                    + shadow_u1 * dw1[step]
                    + shadow_u2 * dw2[step]
                )
            if not absorbed:
                disagreement = q_matrix @ error
                u1 = alpha * sig_power(disagreement, cfg.beta)
                u2 = gamma * disagreement
                error = (
                    error
                    + error_drift(leader[step], error, cfg) * cfg.dt
                    + u1 * dw1[step]
                    + u2 * dw2[step]
                )
                total_j += float(np.sum(u1 * u1 + u2 * u2) * cfg.dt)
                if np.linalg.norm(error) <= cfg.epsilon:
                    first_hit = (step + 1) * cfg.dt
                    error[:] = 0.0
                    absorbed = True
            if step + 1 == save_steps[save_index] if save_index < len(save_steps) else False:
                stored_error[save_index] = error
                if stored_shadow is not None:
                    stored_shadow[save_index] = shadow_error
                stored_j[save_index] = total_j
                save_index += 1
        output[f"error_{name}"] = stored_error
        output[f"state_{name}"] = output["leader"][:, None] + stored_error
        output[f"j_{name}"] = stored_j
        if stored_shadow is not None:
            output["error_full_unabsorbed"] = stored_shadow
            output["hit_time_full_s"] = np.asarray(first_hit)
    return output


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(np.mean(finite))
    if finite.size == 1:
        return mean, mean, mean
    half = 1.96 * float(np.std(finite, ddof=1)) / math.sqrt(finite.size)
    return mean, mean - half, mean + half


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = 1.96
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total**2)) / denom
    return centre - half, centre + half


def summarize(result: dict[str, object], epsilon: float) -> dict[str, float | int]:
    values = np.asarray(result["hits"][epsilon], dtype=float)  # type: ignore[index]
    finite = values[np.isfinite(values)]
    n = int(values.size)
    successes = int(finite.size)
    mean, mean_low, mean_high = mean_ci(finite)
    success_low, success_high = wilson_interval(successes, n)
    j_hit = np.asarray(result["j_budget_at_hit"][epsilon], dtype=float)  # type: ignore[index]
    finite_j_hit = j_hit[np.isfinite(j_hit)]
    j_stop = np.asarray(result["j_budget"], dtype=float)
    terminal = np.asarray(result["terminal_norm"], dtype=float)
    finite_terminal = terminal[np.isfinite(terminal)]
    return {
        "n": n,
        "hits": successes,
        "success_rate": successes / n,
        "success_ci_low": success_low,
        "success_ci_high": success_high,
        "mean_hit": mean,
        "mean_ci_low": mean_low,
        "mean_ci_high": mean_high,
        "median_hit": float(np.median(finite)) if successes else math.nan,
        "q25_hit": float(np.quantile(finite, 0.25)) if successes else math.nan,
        "q75_hit": float(np.quantile(finite, 0.75)) if successes else math.nan,
        "q95_hit": float(np.quantile(finite, 0.95)) if successes else math.nan,
        "median_j": float(np.median(finite_j_hit)) if finite_j_hit.size else math.nan,
        "q25_j": float(np.quantile(finite_j_hit, 0.25)) if finite_j_hit.size else math.nan,
        "q75_j": float(np.quantile(finite_j_hit, 0.75)) if finite_j_hit.size else math.nan,
        "median_j_to_stop_or_horizon": float(np.median(j_stop)),
        "median_terminal_norm": float(np.median(finite_terminal))
        if finite_terminal.size
        else math.inf,
        "numerical_failures": int(np.sum(result["numerical_failure"])),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.5, width=0.6)


def panel_label(ax: plt.Axes, label: str, gap_pt: float = 5.5) -> None:
    """Place a panel identifier a fixed physical distance below its panel.

    Using the complete tight bounding box, rather than an axes-coordinate
    ``y`` value, keeps the gap uniform when panels have different heights,
    tick-label depths, or x-axis titles.
    """

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel_bbox = ax.get_tightbbox(renderer)
    label_x_px = 0.5 * (ax.bbox.x0 + ax.bbox.x1)
    label_y_px = panel_bbox.y0 - gap_pt * fig.dpi / 72.0
    label_x, label_y = fig.transFigure.inverted().transform((label_x_px, label_y_px))
    fig.text(
        label_x,
        label_y,
        f"({label})",
        fontsize=7.0,
        fontweight="normal",
        ha="center",
        va="top",
        clip_on=False,
    )


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values[np.isfinite(values)])
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def figure_representative(rep: dict[str, np.ndarray], cfg: ModelConfig) -> None:
    """Representative paths with unobtrusive embedded detail insets."""

    fig = plt.figure(figsize=(7.20, 5.42))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.00, 1.22], hspace=0.38, wspace=0.38)
    ax_state = fig.add_subplot(gs[0, :])
    ax_error = fig.add_subplot(gs[1, 0])
    ax_channel = fig.add_subplot(gs[1, 1])
    t = rep["time"]

    # a, state convergence with an embedded magnification inset.  A separate
    # source rectangle/connector is intentionally omitted to reduce clutter.
    state_mask = t <= 1.20
    follower_colors = ["#0072B2", "#D55E00", "#009E73", "#8E5AA9"]
    ax_state.plot(t[state_mask], rep["leader"][state_mask], color="black", lw=0.95, label="leader")
    for idx, color in enumerate(follower_colors):
        ax_state.plot(
            t[state_mask], rep["state_full"][state_mask, idx],
            color=color, lw=0.62, label=rf"$v_{idx + 1}$",
        )
    ax_state.set_xlim(0.0, 1.20)
    state_values = np.concatenate(
        [rep["leader"][state_mask]]
        + [rep["state_full"][state_mask, idx] for idx in range(cfg.n_followers)]
    )
    state_values = state_values[np.isfinite(state_values)]
    state_span = max(float(np.ptp(state_values)), 1.0)
    state_limits = (
        float(np.min(state_values)) - 0.07 * state_span,
        float(np.max(state_values)) + 0.07 * state_span,
    )
    ax_state.set_ylim(*state_limits)
    ax_state.set_xlabel("Time (s)")
    ax_state.set_ylabel("Node voltage (V)")
    ax_state.legend(ncol=5, frameon=False, loc="upper right", columnspacing=0.8, handlelength=1.4)

    state_zoom = t <= 0.14
    inset_state = ax_state.inset_axes([0.54, 0.17, 0.36, 0.42])
    inset_state.plot(t[state_zoom], rep["leader"][state_zoom], color="black", lw=0.72)
    for idx, color in enumerate(follower_colors):
        inset_state.plot(t[state_zoom], rep["state_full"][state_zoom, idx], color=color, lw=0.55)
    inset_state.set_xlim(0.0, 0.14)
    zoom_values = np.concatenate(
        [rep["leader"][state_zoom]]
        + [rep["state_full"][state_zoom, idx] for idx in range(cfg.n_followers)]
    )
    zoom_values = zoom_values[np.isfinite(zoom_values)]
    zoom_span = max(float(np.ptp(zoom_values)), 1.0)
    inset_state.set_ylim(
        float(np.min(zoom_values)) - 0.07 * zoom_span,
        float(np.max(zoom_values)) + 0.07 * zoom_span,
    )
    inset_state.set_xticks([0.0, 0.07, 0.14])
    inset_state.set_title("0--0.14 s", pad=1.5)
    inset_state.tick_params(labelsize=6.0, length=1.8, width=0.5)
    for spine in inset_state.spines.values():
        spine.set_linewidth(0.55)
    # b, matched-method errors with the common initial interval enlarged in situ.
    error_mask = t <= 2.0
    method_style = {
        "full": (BLUE, "full two-map", "-"),
        "power": (ORANGE, "power-law only", (0, (4, 2))),
        "linear": (GREEN, "linear only", (0, (1.5, 1.5))),
        "none": (GREY, "no excitation", "-"),
    }
    error_series: dict[str, np.ndarray] = {}
    for name, (color, label, linestyle) in method_style.items():
        norm = np.linalg.norm(rep[f"error_{name}"], axis=1)
        error_series[name] = np.maximum(norm, cfg.epsilon / 2.0)
        ax_error.plot(t[error_mask], error_series[name][error_mask], color=color, ls=linestyle, lw=0.78, label=label)
    hit_time = float(np.asarray(rep["hit_time_full_s"]))
    ax_error.axvline(hit_time, color=BLUE, lw=0.50, ls=(0, (2, 2)), alpha=0.75)
    ax_error.axhline(cfg.epsilon, color="black", lw=0.55, ls=(0, (2, 2)))
    ax_error.text(1.96, cfg.epsilon * 1.20, r"$\varepsilon$", ha="right", va="bottom")
    ax_error.set_xlim(0.0, 2.0)
    ax_error.set_yscale("log")
    ax_error.set_xlabel("Time (s)")
    ax_error.set_ylabel(r"Tracking error $\|e\|_2$")
    ax_error.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.03, 0.82),
        ncol=2,
        handlelength=1.7,
        columnspacing=0.55,
        labelspacing=0.25,
        fontsize=5.7,
    )

    error_zoom = t <= 0.25
    inset_error = ax_error.inset_axes([0.49, 0.18, 0.45, 0.39])
    for name, (color, _label, linestyle) in method_style.items():
        inset_error.plot(t[error_zoom], error_series[name][error_zoom], color=color, ls=linestyle, lw=0.60)
    inset_error.axvline(hit_time, color=BLUE, lw=0.48, ls=(0, (2, 2)), alpha=0.75)
    inset_error.axhline(cfg.epsilon, color="black", lw=0.45, ls=(0, (2, 2)))
    inset_error.set_xlim(0.0, 0.25)
    inset_error.set_ylim(3.0e-4, 40.0)
    inset_error.set_yscale("log")
    inset_error.set_title("0--0.25 s", pad=1.5)
    inset_error.tick_params(labelsize=6.0, length=1.8, width=0.5)
    for spine in inset_error.spines.values():
        spine.set_linewidth(0.55)
    # c, coefficient norms and their pre-hitting ratio.
    q_matrix = pinned_matrix((5.0, 5.0, 5.0, 5.0))
    disagreement = rep["error_full"] @ q_matrix
    u1_norm = np.linalg.norm(cfg.alpha * sig_power(disagreement, cfg.beta), axis=1)
    u2_norm = np.linalg.norm(cfg.gamma * disagreement, axis=1)
    channel_mask = t <= 0.50
    floor = 1.0e-3
    ax_channel.plot(t[channel_mask], np.maximum(u1_norm[channel_mask], floor), color=ORANGE, lw=0.78, label=r"power-law $\|u_1\|_2$")
    ax_channel.plot(t[channel_mask], np.maximum(u2_norm[channel_mask], floor), color=GREEN, lw=0.78, ls=(0, (3, 1.5)), label=r"linear $\|u_2\|_2$")
    full_error = np.linalg.norm(rep["error_full"], axis=1)
    hit_indices = np.flatnonzero(full_error <= cfg.epsilon)
    if np.isfinite(hit_time):
        ax_channel.axvline(hit_time, color=BLUE, lw=0.60, ls=(0, (2, 2)))
        ax_channel.text(hit_time + 0.018, 1.8e-3, r"$T_\varepsilon$", color=BLUE, rotation=90)
    ax_channel.set_xlim(0.0, 0.50)
    ax_channel.set_yscale("log")
    ax_channel.set_xlabel("Time (s)")
    ax_channel.set_ylabel("Diffusion-coefficient norm")
    ax_channel.legend(
        frameon=False,
        loc="upper right",
        handlelength=1.7,
        labelspacing=0.25,
        fontsize=5.7,
    )

    ratio_mask = (t <= hit_time) & (u2_norm > 1.0e-12)
    ratio = u1_norm[ratio_mask] / u2_norm[ratio_mask]
    inset_ratio = ax_channel.inset_axes([0.50, 0.24, 0.41, 0.30])
    inset_ratio.plot(t[ratio_mask], ratio, color=BLUE, lw=0.65)
    inset_ratio.axhline(1.0, color=GREY, lw=0.45, ls=(0, (2, 2)))
    inset_ratio.set_xlim(0.0, max(hit_time, 0.02))
    inset_ratio.set_title(r"$\|u_1\|_2/\|u_2\|_2$", pad=1.0)
    inset_ratio.tick_params(labelsize=6.0, length=1.8, width=0.5)
    for spine in inset_ratio.spines.values():
        spine.set_linewidth(0.55)

    for ax in (ax_state, ax_error, ax_channel):
        clean_axes(ax)
    panel_label(ax_state, "a")
    panel_label(ax_error, "b")
    panel_label(ax_channel, "c")
    export_figure(fig, "Fig2_representative_ablation")

def export_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.015}
    fig.savefig(FIG_DIR / f"{stem}.pdf", **kwargs)
    fig.savefig(FIG_DIR / f"{stem}.svg", **kwargs)
    fig.savefig(FIG_DIR / f"{stem}.tiff", dpi=600, **kwargs)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, **kwargs)
    plt.close(fig)


def export_representative_csv(rep: dict[str, np.ndarray]) -> None:
    rows: list[dict[str, object]] = []
    for idx, time_value in enumerate(rep["time"]):
        if time_value > REPRESENTATIVE_ARCHIVE_WINDOW_S:
            break
        row: dict[str, object] = {
            "time_s": float(time_value),
            "leader": float(rep["leader"][idx]),
        }
        for follower in range(4):
            row[f"state_full_{follower + 1}"] = float(
                rep["state_full"][idx, follower]
            )
        for name in ("full", "power", "linear", "none"):
            for follower in range(4):
                row[f"error_{name}_{follower + 1}"] = float(
                    rep[f"error_{name}"][idx, follower]
                )
        row["hit_time_full_s"] = float(np.asarray(rep["hit_time_full_s"]))
        rows.append(row)
    write_csv(DATA_DIR / "representative_trajectories.csv", rows)
