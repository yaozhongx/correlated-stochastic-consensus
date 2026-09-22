"""Recompute the certificate values reported in manuscript CSR-2026-08-0121.

The script evaluates the closed-form quantities used in the manuscript and
prints one compact JSON result to standard output.
"""

from __future__ import annotations

import json
import math

import numpy as np


BETA = 11.0 / 12.0
ALPHA = 1.0
GAMMA = 1.0
THETA = 1.5
N_FOLLOWERS = 4
INITIAL_ERROR = np.array([3.0, 2.0, -6.0, -2.0], dtype=float)
LAPLACIAN = np.array(
    [
        [0.50, -0.25, 0.00, -0.25],
        [-0.25, 0.50, -0.25, 0.00],
        [0.00, -0.25, 0.50, -0.25],
        [-0.25, 0.00, -0.25, 0.50],
    ],
    dtype=float,
)


def pinned_matrix(pin_gain: float) -> np.ndarray:
    """Return Q=L+B for uniform direct leader access B=pin_gain*I."""

    return LAPLACIAN + pin_gain * np.eye(N_FOLLOWERS)


def spectral_power_slack(beta: float, m_q: float, M_q: float) -> float:
    """Separated spectral power slack in the closed-form certificate."""

    return (
        2.0 * beta * m_q ** (1.0 + beta)
        - M_q ** (1.0 + beta) * N_FOLLOWERS ** (1.0 - beta)
    )


def spectral_linear_slack(
    beta: float,
    m_q: float,
    M_q: float,
    rho: float,
) -> float:
    """Unscaled linear slack used in the manuscript."""

    c_squared = 1.0 / (1.0 + rho)
    return (
        c_squared
        * (2.0 * beta * GAMMA**2 * m_q**2 - GAMMA**2 * M_q**2)
        - 2.0 * THETA
    )


def spectral_cross_slack(beta: float, m_q: float, q_matrix: np.ndarray) -> float:
    """Cross-slack lower bound using the induced-norm interpolation bound."""

    chi_upper_bound = float(np.linalg.norm(q_matrix, ord=np.inf))
    return 2.0 * beta * m_q - chi_upper_bound


def exact_linear_margin(beta: float, m_q: float, M_q: float) -> float:
    """Exact linear directional margin from Proposition 3."""

    a_star = min(M_q, max(m_q, (m_q + M_q) / (4.0 * beta)))
    return 2.0 * beta * a_star**2 - (m_q + M_q) * a_star + m_q * M_q


def coupled_power_margin(beta: float, m_q: float, M_q: float) -> float:
    """Coupled power-margin lower bound from Proposition 2."""

    p = 1.0 + beta
    r = 2.0 * beta / p
    a_min = m_q ** (p / 2.0)
    a_star = (
        M_q * N_FOLLOWERS ** (1.0 - r) * r / (4.0 * beta)
    ) ** (1.0 / (2.0 - r))
    a_bar = max(a_min, a_star)
    return 2.0 * beta * a_bar**2 - M_q * N_FOLLOWERS ** (1.0 - r) * a_bar**r


def mean_time_upper_bound(
    q_matrix: np.ndarray,
    rho: float,
    certified_margin: float,
    beta: float,
) -> float:
    """Evaluate the manuscript's sufficient mean settling-time bound."""

    c_squared = 1.0 / (1.0 + rho)
    initial_v = float(INITIAL_ERROR @ q_matrix @ INITIAL_ERROR)
    return initial_v ** (1.0 - beta) / (
        c_squared * ALPHA**2 * certified_margin * (1.0 - beta)
    )


def spectral_certificate(
    q_matrix: np.ndarray,
    beta: float,
    rho: float,
) -> dict[str, float | bool]:
    """Evaluate the closed-form spectral certificate used in the manuscript."""

    eigenvalues = np.linalg.eigvalsh(q_matrix)
    m_q = float(eigenvalues[0])
    M_q = float(eigenvalues[-1])
    power = spectral_power_slack(beta, m_q, M_q)
    linear = spectral_linear_slack(beta, m_q, M_q, rho)
    cross = spectral_cross_slack(beta, m_q, q_matrix)
    certified = power > 0.0 and linear > 0.0 and cross > 0.0
    return {
        "power_slack": power,
        "linear_slack": linear,
        "cross_slack": cross,
        "certified": certified,
        "mean_time_upper_bound_s": (
            mean_time_upper_bound(q_matrix, rho, power, beta)
            if power > 0.0
            else math.inf
        ),
    }


def manuscript_certificate_values() -> dict[str, dict[str, float]]:
    """Return all certificate values numerically reported in the manuscript."""

    q_full = pinned_matrix(5.0)
    full_rho_0 = spectral_certificate(q_full, BETA, 0.0)
    full_rho_1 = spectral_certificate(q_full, BETA, 1.0)

    q_reduced = pinned_matrix(3.0)
    reduced_eigenvalues = np.linalg.eigvalsh(q_reduced)
    reduced_m = float(reduced_eigenvalues[0])
    reduced_M = float(reduced_eigenvalues[-1])
    reduced_separated_power = spectral_power_slack(BETA, reduced_m, reduced_M)
    reduced_coupled_power = coupled_power_margin(BETA, reduced_m, reduced_M)
    reduced_exact_linear = exact_linear_margin(BETA, reduced_m, reduced_M)
    reduced_c_squared = 1.0 / (1.0 + 1.0)
    reduced_d1 = GAMMA**2 * reduced_exact_linear - 2.0 * THETA / reduced_c_squared

    return {
        "full_pinning_B_5I4": {
            "power_slack": float(full_rho_0["power_slack"]),
            "linear_slack_rho_0": float(full_rho_0["linear_slack"]),
            "linear_slack_rho_1": float(full_rho_1["linear_slack"]),
            "cross_slack": float(full_rho_0["cross_slack"]),
            "mean_time_upper_bound_rho_0_s": float(
                full_rho_0["mean_time_upper_bound_s"]
            ),
            "mean_time_upper_bound_rho_1_s": float(
                full_rho_1["mean_time_upper_bound_s"]
            ),
        },
        "reduced_pinning_B_3I4_rho_1": {
            "separated_spectral_power_slack": reduced_separated_power,
            "coupled_power_margin": reduced_coupled_power,
            "exact_linear_margin": reduced_exact_linear,
            "d_1": reduced_d1,
            "mean_time_upper_bound_s": mean_time_upper_bound(
                q_reduced,
                1.0,
                reduced_coupled_power,
                BETA,
            ),
        },
    }


def main() -> None:
    print(json.dumps(manuscript_certificate_values(), indent=2))


if __name__ == "__main__":
    main()
