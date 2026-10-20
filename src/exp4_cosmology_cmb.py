"""
Experiment 4: Cosmological Spectral Broadening vs. Adiabatic FLRW Redshift,
COBE/FIRAS Blackbody Distortion Bounds, and Spectral Line Achromaticity.

Research Question:
The Wave Entities hypothesis posits that cosmic expansion and cooling consist of
an initial concentration of wave entities around a high frequency spreading out
across a wider frequency spectrum and moving toward lower frequencies
("as the energy is distributed across wider spectrum, and larger spectrum essentially
equals larger distance and more space", Notes p. 3; PDF A Sec 4; PDF B Sec 3.4).

Are "spectral broadening" (increasing frequency variance sigma_f^2) and "shift in
average frequency" (decreasing mean frequency <f>) physically equivalent to
cosmological expansion and cooling, or do they have distinct, empirically falsifiable
consequences?

Tests Conducted:
1. Planck Blackbody Preservation vs. Spectral Broadening Distortion:
   - Control (Standard FLRW Adiabatic Expansion):
     Uniform multiplicative scaling f -> f / a(t) (where a(t) = 1 + z).
     Under multiplicative scaling, a Planck blackbody spectrum I(x) propto x^3 / (e^x - 1)
     evolves to an EXACT Planck blackbody at temperature T(t) = T_0 / a(t), and its
     spectral standard deviation sigma_f(t) = sigma_f(0) / a(t) NARROWS by a factor of
     1 + z (shrinking ~1100x from recombination to today), rather than broadening!
   - Hypothesis Model A (Downward Drift + Spectral Broadening / Advection-Diffusion):
     Evolves an initial high-frequency distribution or Planck spectrum under a drift-diffusion
     operator in frequency space that simultaneously shifts <f> downward while broadening
     the spectral width sigma_f.
     Measures the resulting fractional deviation |I(f) - B_f(T_fit)| / B_peak from the
     best-fit Planck blackbody and compares against the COBE/FIRAS empirical bound
     (|Delta I / I_peak| < 5.0e-5, Fixsen et al. 1996).

2. Narrow Atomic Line Redshift Achromaticity & Line-Ratio Preservation:
   - Tracks a pair of narrow spectral lines (f_10, f_20, e.g., an alkali fine-structure
     doublet with ratio R_0 = f_20 / f_10) under:
     (i) Multiplicative cosmological redshift f(t) = f_0 / (1 + z),
     (ii) Additive downward frequency shift f(t) = f_0 - Delta f_shift,
     (iii) Nonlinear frequency-dependent dispersive drift df/dt = -c_disp * f^p (p != 1),
     (iv) Diffusive spectral broadening.
   - Compares against empirical quasar absorption-line bounds on fine-structure constant
     variation (|Delta alpha / alpha| < 1e-6) and thermal/instrumental line widths.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def planck_energy_spectrum(f: np.ndarray, T: float) -> np.ndarray:
    """
    Dimensionless Planck spectral energy density u(f, T) = (f^3) / (exp(f / T) - 1)
    in units where h = k_B = c = 1.
    """
    x = np.clip(f / max(T, 1e-12), 1e-9, 500.0)
    return (f ** 3) / np.expm1(x)


def fit_best_planck_temperature_and_scale(
    f: np.ndarray, spectrum: np.ndarray
) -> Tuple[float, float, float, np.ndarray]:
    """
    Fit an arbitrary spectral distribution `spectrum(f)` to a scaled Planck profile
    A * B(f, T_fit) and return (T_fit, A_fit, max_rel_residual_to_peak, fitted_curve).
    """
    def obj(T_try: float) -> float:
        if T_try <= 1e-5:
            return 1e30
        b = planck_energy_spectrum(f, T_try)
        scale = np.sum(spectrum * b) / max(np.sum(b * b), 1e-30)
        resid = spectrum - scale * b
        return float(np.sum(resid ** 2))

    res = minimize_scalar(obj, bounds=(0.05, 10.0), method="bounded")
    T_best = float(res.x)
    b_best = planck_energy_spectrum(f, T_best)
    A_best = float(np.sum(spectrum * b_best) / max(np.sum(b_best * b_best), 1e-30))
    fit_curve = A_best * b_best
    peak_val = max(float(np.max(fit_curve)), 1e-30)
    rms_rel_distortion = float(np.sqrt(np.mean((spectrum - fit_curve) ** 2)) / peak_val)
    max_rel_distortion = float(np.max(np.abs(spectrum - fit_curve)) / peak_val)
    return T_best, rms_rel_distortion, max_rel_distortion, fit_curve


def spectral_moments(f: np.ndarray, u: np.ndarray) -> Tuple[float, float]:
    """Compute mean frequency <f> and spectral standard deviation sigma_f."""
    df = f[1] - f[0]
    norm = np.sum(u) * df
    if norm <= 1e-30:
        return 0.0, 0.0
    p = u * df / norm
    mean_f = float(np.sum(f * p))
    var_f = float(np.sum(((f - mean_f) ** 2) * p))
    return mean_f, float(np.sqrt(max(0.0, var_f)))


def evolve_advection_diffusion_spectrum(
    f: np.ndarray,
    u0: np.ndarray,
    v_drift: float,
    D_diff: float,
    t_end: float,
    dt: float = 0.0005,
) -> np.ndarray:
    """
    Evolve spectral energy density u(f, t) on frequency grid f under:
        du/dt = v_drift * du/df + D_diff * d^2u/df^2
    using a CFL-stable finite-volume scheme with zero-flux boundaries at [f_min, f_max].
    """
    df = float(f[1] - f[0])
    # Exact stability bound for explicit advection-diffusion:
    dt_cfl = 0.35 / ((abs(v_drift) / df) + (2.0 * D_diff / (df * df)) + 1e-12)
    dt_actual = min(dt, dt_cfl)
    n_steps = max(1, int(np.ceil(t_end / dt_actual)))
    dt_step = t_end / float(n_steps)

    u = u0.astype(float).copy()
    for _ in range(n_steps):
        # Interior cell-face fluxes J_{i+1/2} = -v_drift * u_{i+1} - D_diff * (u_{i+1} - u_i)/df
        # with zero flux J_{-1/2} = J_{N-1/2} = 0 at boundaries:
        flux_face = -v_drift * u[1:] - D_diff * (u[1:] - u[:-1]) / df
        du = np.zeros_like(u)
        du[:-1] -= flux_face / df
        du[1:] += flux_face / df
        u = u + dt_step * du
        u = np.maximum(u, 0.0)
    return u


def audit_atomic_line_achromaticity() -> Dict[str, Any]:
    """
    Compare how a spectroscopic doublet with rest frequencies f1_0 = 500 THz,
    f2_0 = 600 THz (rest ratio R_0 = f2_0 / f1_0 = 1.200000) and narrow thermal
    width sigma_0 = 0.05 THz evolves to redshift z = 2.0 (where <f> drops by 3x) under:
    1. Multiplicative FLRW expansion: f -> f / (1 + z)
    2. Additive downward shift: f -> f - Delta_f (chosen so mean shifts by 3x)
    3. Nonlinear dispersive shift: df/dt = -gamma * f^2
    4. Diffusive spectral broadening with D_f * t = (5.0 THz)^2
    """
    f1_0 = 500.0
    f2_0 = 600.0
    R_0 = f2_0 / f1_0
    sigma_0 = 0.05
    z_target = 2.0
    scale_factor = 1.0 + z_target  # 3.0

    # 1. Multiplicative FLRW scaling
    f1_mult = f1_0 / scale_factor
    f2_mult = f2_0 / scale_factor
    R_mult = f2_mult / f1_mult
    z1_mult = (f1_0 / f1_mult) - 1.0
    z2_mult = (f2_0 / f2_mult) - 1.0
    sigma_mult_rel = (sigma_0 / scale_factor) / f1_mult

    # 2. Additive downward frequency shift (shifting midpoint 550 THz -> 550/3 = 183.333 THz)
    f_mid_0 = 0.5 * (f1_0 + f2_0)
    delta_add = f_mid_0 * (1.0 - 1.0 / scale_factor)  # 366.667 THz
    f1_add = f1_0 - delta_add
    f2_add = f2_0 - delta_add
    R_add = f2_add / f1_add
    z1_add = (f1_0 / f1_add) - 1.0
    z2_add = (f2_0 / f2_add) - 1.0

    # 3. Nonlinear dispersive shift df/dt = -gamma * f^2 => f(t) = f_0 / (1 + gamma * t * f_0)
    # Choose gamma * t so that f1_0 redshifts by z_target = 2.0 (i.e., gamma * t * f1_0 = 2.0)
    gt = z_target / f1_0
    f1_nl = f1_0 / (1.0 + gt * f1_0)
    f2_nl = f2_0 / (1.0 + gt * f2_0)
    R_nl = f2_nl / f1_nl
    z1_nl = (f1_0 / f1_nl) - 1.0
    z2_nl = (f2_0 / f2_nl) - 1.0

    # 4. Diffusive broadening of line width
    sigma_diff = float(np.sqrt((sigma_0 / scale_factor) ** 2 + 5.0 ** 2))
    sigma_diff_rel = sigma_diff / f1_mult

    return {
        "rest_frequencies_THz": [f1_0, f2_0],
        "rest_ratio_R0": R_0,
        "rest_relative_linewidth": float(sigma_0 / f1_0),
        "flrw_multiplicative": {
            "f1_obs": f1_mult, "f2_obs": f2_mult,
            "ratio_obs": R_mult, "ratio_error": float(abs(R_mult - R_0)),
            "z1": z1_mult, "z2": z2_mult, "achromatic_dz": float(abs(z2_mult - z1_mult)),
            "relative_linewidth": float(sigma_mult_rel),
        },
        "additive_frequency_shift": {
            "f1_obs": f1_add, "f2_obs": f2_add,
            "ratio_obs": R_add, "ratio_error": float(abs(R_add - R_0)),
            "z1": z1_add, "z2": z2_add, "achromatic_dz": float(abs(z2_add - z1_add)),
        },
        "nonlinear_dispersive_shift_f2": {
            "f1_obs": f1_nl, "f2_obs": f2_nl,
            "ratio_obs": R_nl, "ratio_error": float(abs(R_nl - R_0)),
            "z1": z1_nl, "z2": z2_nl, "achromatic_dz": float(abs(z2_nl - z1_nl)),
        },
        "diffusive_broadened_relative_linewidth": float(sigma_diff_rel),
        "line_broadening_factor": float(sigma_diff_rel / sigma_mult_rel),
    }


def run_experiment_4(output_dir: Path) -> Dict[str, Any]:
    """Run all cosmological redshift and CMB blackbody tests in Experiment 4."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    f_grid = np.linspace(0.02, 25.0, 1200)
    T_init = 2.0
    T_cooled_target = 1.0  # factor of 2 expansion (1 + z = 2)

    # 1. Initial Planck blackbody at T_init = 2.0
    u_init = planck_energy_spectrum(f_grid, T_init)
    mean_init, std_init = spectral_moments(f_grid, u_init)

    # 2. Control: Standard FLRW Adiabatic Redshift (f -> f / 2, T -> T / 2 = 1.0)
    u_flrw = planck_energy_spectrum(f_grid, T_cooled_target)
    mean_flrw, std_flrw = spectral_moments(f_grid, u_flrw)
    T_fit_flrw, rms_flrw, max_flrw, _ = fit_best_planck_temperature_and_scale(f_grid, u_flrw)

    # 3. Hypothesis Case A: Narrow high-frequency Gaussian concentration spreading via diffusion + downward drift
    u_narrow_high = np.exp(-0.5 * ((f_grid - 10.0) / 0.8) ** 2)
    u_spread_from_narrow = evolve_advection_diffusion_spectrum(
        f_grid, u_narrow_high, v_drift=3.0, D_diff=2.0, t_end=2.0, dt=0.0005
    )
    mean_narrow_init, std_narrow_init = spectral_moments(f_grid, u_narrow_high)
    mean_spread_narrow, std_spread_narrow = spectral_moments(f_grid, u_spread_from_narrow)
    T_fit_narrow, rms_narrow, max_narrow, fit_narrow_curve = fit_best_planck_temperature_and_scale(
        f_grid, u_spread_from_narrow
    )

    # 4. Hypothesis Case B: Initial Thermal Planck spectrum subjected to downward drift + diffusive broadening
    u_thermal_broadened = evolve_advection_diffusion_spectrum(
        f_grid, u_init, v_drift=1.8, D_diff=1.2, t_end=1.2, dt=0.0005
    )
    mean_th_broad, std_th_broad = spectral_moments(f_grid, u_thermal_broadened)
    T_fit_th, rms_th, max_th, fit_th_curve = fit_best_planck_temperature_and_scale(
        f_grid, u_thermal_broadened
    )

    # 5. Sweep diffusion coefficient D_diff to find maximum allowed broadening under COBE/FIRAS bound (5e-5)
    D_sweep = np.logspace(-5, 0.2, 20)
    firas_bound = 5.0e-5
    distortions_vs_D = []
    for d_val in D_sweep:
        # Pure diffusive broadening on top of a T=1.0 Planck spectrum over delta_t = 0.5
        u_test = evolve_advection_diffusion_spectrum(
            f_grid, u_flrw, v_drift=0.0, D_diff=float(d_val), t_end=0.5, dt=0.0002
        )
        _, rms_d, max_d, _ = fit_best_planck_temperature_and_scale(f_grid, u_test)
        distortions_vs_D.append(max_d)

    # 6. Atomic line achromaticity audit
    line_audit = audit_atomic_line_achromaticity()

    # --- Generate Multi-Panel Diagnostic Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), dpi=150)

    # Panel A: FLRW Adiabatic Cooling NARROWS the Spectrum vs Hypothesis Broadening
    ax = axes[0, 0]
    ax.plot(f_grid, u_init / np.max(u_init), "k--", lw=1.8,
            label=f"Initial Hot Blackbody ($T_0=2.0, \\langle f\\rangle={mean_init:.2f}, \\sigma_f={std_init:.2f}$)")
    ax.plot(f_grid, u_flrw / np.max(u_flrw), color="#2ca02c", lw=2.2,
            label=f"FLRW Cooled ($T=1.0, \\langle f\\rangle={mean_flrw:.2f}, \\sigma_f={std_flrw:.2f}$ — NARROWER)")
    ax.plot(f_grid, u_thermal_broadened / np.max(u_thermal_broadened), color="#d62728", lw=2.0,
            label=f"Drift + Broadening ($\\langle f\\rangle={mean_th_broad:.2f}, \\sigma_f={std_th_broad:.2f}$ — BROADER)")
    ax.set_xlim(0, 20)
    ax.set_xlabel("Frequency $f$ [dimensionless $h f / k_B T_{\\mathrm{ref}}$]")
    ax.set_ylabel("Peak-Normalized Energy Density $u(f) / u_{\\max}$")
    ax.set_title("A. Physical Cooling Narrows $\\sigma_f \\propto T$; Spectral Broadening Distorts It")
    ax.legend(loc="upper right", fontsize=8.0)
    ax.grid(True, alpha=0.3)

    # Panel B: High-Frequency Narrow Cluster Spreading vs Best-Fit Planck Curve
    ax = axes[0, 1]
    scale_n = np.max(fit_narrow_curve)
    ax.plot(f_grid, u_narrow_high * 0.5, color="gray", ls=":", lw=1.5,
            label="Initial High-$f$ Wave Entity Cluster ($f_0=10.0$)")
    ax.plot(f_grid, u_spread_from_narrow / scale_n, color="#d62728", lw=2.2,
            label="Evolved Advective-Diffusive Spectrum")
    ax.plot(f_grid, fit_narrow_curve / scale_n, color="#1f77b4", ls="--", lw=1.8,
            label=f"Best-Fit Planck Blackbody ($T_{{\\mathrm{{fit}}}}={T_fit_narrow:.2f}$, Max Err $= {max_narrow*100:.1f}\\%$)")
    ax.set_xlim(0, 16)
    ax.set_xlabel("Frequency $f$")
    ax.set_ylabel("Normalized Spectral Intensity")
    ax.set_title("B. Diffusive Spreading from Single Frequency Fails Planck Blackbody Shape")
    ax.legend(loc="upper right", fontsize=8.0)
    ax.grid(True, alpha=0.3)

    # Panel C: Spectral Distortion vs Diffusive Broadening Coefficient (COBE/FIRAS Bound)
    ax = axes[1, 0]
    ax.loglog(D_sweep, distortions_vs_D, "o-", color="#9467bd", lw=2.0, markersize=5,
              label="Peak Fractional Distortion $|\\Delta I|_{\\max} / I_{\\mathrm{peak}}$")
    ax.axhline(firas_bound, color="#d62728", ls="--", lw=2.0,
               label="COBE/FIRAS Upper Limit ($5.0 \\times 10^{-5}$, Fixsen et al. 1996)")
    ax.fill_between(D_sweep, firas_bound, 1.0, color="#d62728", alpha=0.12, label="Empirically Excluded Regime")
    ax.set_xlabel("Spectral Broadening Diffusion Coefficient $D_f$")
    ax.set_ylabel("Fractional Blackbody Distortion $|\\Delta I| / I_{\\mathrm{peak}}$")
    ax.set_title("C. COBE/FIRAS Blackbody Constraint Rules Out Diffusive Spectral Broadening")
    ax.set_ylim(1e-6, 1.0)
    ax.legend(loc="lower right", fontsize=8.0)
    ax.grid(True, which="both", alpha=0.3)

    # Panel D: Spectroscopic Doublet Redshift Achromaticity Test (z = 2.0)
    ax = axes[1, 1]
    models_lbl = [
        "Rest Frame\n($z=0$)",
        "FLRW Scaling\n$f \\to f/(1+z)$",
        "Additive Shift\n$f \\to f - \\Delta f$",
        "Nonlinear Drift\n$\\dot{f} = -\\gamma f^2$",
    ]
    ratios = [
        line_audit["rest_ratio_R0"],
        line_audit["flrw_multiplicative"]["ratio_obs"],
        line_audit["additive_frequency_shift"]["ratio_obs"],
        line_audit["nonlinear_dispersive_shift_f2"]["ratio_obs"],
    ]
    colors_bar = ["gray", "#2ca02c", "#d62728", "#ff7f0e"]
    bars = ax.bar(models_lbl, ratios, color=colors_bar, width=0.5)
    ax.axhline(line_audit["rest_ratio_R0"], color="k", ls="--", lw=1.4, label=f"Invariant Atomic Ratio $R_0 = {line_audit['rest_ratio_R0']:.3f}$")
    for b, val in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width() / 2.0, val + 0.03, f"{val:.3f}", ha="center", fontweight="bold")
    ax.set_ylim(0.8, 1.95)
    ax.set_ylabel("Observed Doublet Frequency Ratio $f_2 / f_1$")
    ax.set_title("D. Atomic Doublet Ratio Preservation Requires Multiplicative Scaling")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = fig_dir / "exp4_cosmology_cmb.png"
    plt.savefig(fig_path)
    plt.close(fig)

    summary = {
        "experiment": "exp4_cosmology_cmb",
        "planck_blackbody_vs_spectral_broadening": {
            "initial_hot_blackbody": {"T": T_init, "mean_f": mean_init, "std_f": std_init},
            "flrw_adiabatic_cooled": {
                "T": T_cooled_target, "mean_f": mean_flrw, "std_f": std_flrw,
                "width_change_factor": float(std_flrw / std_init),
                "max_planck_distortion": max_flrw,
            },
            "narrow_high_f_diffusive_spreading": {
                "initial_mean_f": mean_narrow_init, "initial_std_f": std_narrow_init,
                "final_mean_f": mean_spread_narrow, "final_std_f": std_spread_narrow,
                "best_fit_T": T_fit_narrow,
                "rms_planck_distortion": rms_narrow,
                "max_planck_distortion": max_narrow,
                "violation_factor_over_COBE_FIRAS": float(max_narrow / firas_bound),
            },
            "thermal_with_diffusive_broadening": {
                "final_mean_f": mean_th_broad, "final_std_f": std_th_broad,
                "best_fit_T": T_fit_th,
                "rms_planck_distortion": rms_th,
                "max_planck_distortion": max_th,
                "violation_factor_over_COBE_FIRAS": float(max_th / firas_bound),
            },
            "cobe_firas_limit": firas_bound,
            "finding": (
                "Spectral broadening and downward shift of average frequency are NOT physically "
                "equivalent to cosmological cooling. Under standard adiabatic expansion T -> T/(1+z), "
                f"the spectral width sigma_f NARROWS proportionally to T (from {std_init:.3f} to "
                f"{std_flrw:.3f} for 1+z=2). By contrast, diffusive spreading from a narrow high-frequency "
                "state or diffusive broadening of a thermal state produces non-Planckian distributions "
                f"with fractional residuals of {max_narrow:.3f} and {max_th:.3f}, exceeding the "
                f"COBE/FIRAS empirical bound (5e-5) by factors of {max_narrow/firas_bound:.0f}x and "
                f"{max_th/firas_bound:.0f}x."
            ),
        },
        "spectroscopic_achromaticity_audit": {
            "results": line_audit,
            "finding": (
                "Preserving invariant atomic frequency ratios f2/f1 across cosmological distances "
                "strictly requires uniform multiplicative frequency scaling f(t) = f0 / a(t). "
                "Additive downward shifts alter a 1.200 doublet ratio to "
                f"{line_audit['additive_frequency_shift']['ratio_obs']:.3f} at z=2 (producing "
                f"spurious redshift discrepancy Delta z = {line_audit['additive_frequency_shift']['achromatic_dz']:.3f}), "
                "while nonlinear dispersive drifts alter it to "
                f"{line_audit['nonlinear_dispersive_shift_f2']['ratio_obs']:.3f} and diffusive "
                f"spreading broadens narrow absorption lines by {line_audit['line_broadening_factor']:.1f}x."
            ),
        },
        "figure_path": str(fig_path),
    }

    with open(output_dir / "exp4_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "results"
    res = run_experiment_4(out)
    print(json.dumps(res["planck_blackbody_vs_spectral_broadening"]["finding"], indent=2))
    print(json.dumps(res["spectroscopic_achromaticity_audit"]["finding"], indent=2))
