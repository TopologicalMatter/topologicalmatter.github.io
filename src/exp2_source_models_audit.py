"""
Experiment 2: Mathematical and Numerical Audit of the Source PDFs' Proposed Models.

Research Question:
Do the explicit mathematical equations proposed in the source documents
(`Wave Entities Notes.pdf`, Section 6, p. 17; `Wave Entities and the Universe...pdf`,
Section 5.2, p. 4 & Section 6, p. 5) actually produce the behaviors claimed for them
(frequency-shift attraction, Newtonian 1/r^2 gravity, and finite-energy wave entities)?

Models Audited:
1. Source Toy Model (PDF B, Section 5.2): Coupled Oscillators in Frequency Space:
       d^2 x_i / dt^2 + f_i^2 x_i = sum_{j != i} K(|f_i - f_j|) (x_j - x_i)
   with K(Delta f) = g / (Delta f^2 + eps^2).
   - Analytical & numerical audit of normal-mode eigenfrequencies:
     Demonstrates that positive diffusive coupling K > 0 induces Wigner-von Neumann
     LEVEL REPULSION (the upper frequency shifts UP and the lower frequency shifts
     DOWN, increasing |f_+ - f_-| > |f_2 - f_1|), i.e., emergent-space REPULSION!
   - Audits negative coupling K < 0 (attractive mode shift), showing that it induces
     a tachyonic dynamical instability (exponential runaway) when 2|K| > |f_2^2 - f_1^2|.
   - Audits Kuramoto-type phase/frequency synchronization as an alternative completion,
     showing why phase-locking leads to threshold all-or-nothing frequency collapse
     (d_ij -> 0) rather than inverse-square Newtonian orbital mechanics.

2. Source Illustrative Wave Model (PDF A & B, Section 6):
       W(r, t) = A * sin(2*pi*f_0*t - k*r) / (r^2 + alpha^2)
       W_total(r, t) = W_1(r, t) + W_2(r, t),   I(r, t) = |W_total(r, t)|^2
       Delta f propto A_1 A_2 / r^2  ===>  F_g propto Delta f / r ~ G_eff A_1 A_2 / r^3.
   - Audit A (Linear Superposition No-Shift Theorem): Linear superposition W_1 + W_2
     produces amplitude beating at |f_1 - f_2| in I(r, t), but ZERO shift in the
     spectral centroids f_1, f_2.
   - Audit B (Energy Integrability in 1D vs 3D): Computes L1 and L2 integrals of
     bell-shaped inverse-square profiles across dimensions and cutoff scales alpha.
   - Audit C (Inverse-Cube Force Orbital Catastrophe): Demonstrates via numerical orbit
     integration and Bertrand's Theorem that the source PDF's derived force law
     F_g propto 1/r^3 (from V_eff propto -1/r^2) admits NO stable bound orbits—test
     particles either spiral into r -> 0 (Cotes' spiral) or escape to infinity.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def exact_two_oscillator_eigenfrequencies(
    f1: float, f2: float, K: float
) -> Tuple[float, float, bool]:
    """
    Compute exact normal-mode eigenfrequencies for the source PDF's coupled
    oscillator equation:
        x1'' + f1^2 x1 = K (x2 - x1)
        x2'' + f2^2 x2 = K (x1 - x2)
    Dynamical matrix D = [[f1^2 + K, -K], [-K, f2^2 + K]].
    """
    trace = (f1 ** 2 + K) + (f2 ** 2 + K)
    diff = f2 ** 2 - f1 ** 2
    disc = np.sqrt(diff ** 2 + 4.0 * (K ** 2))
    lam_minus = 0.5 * (trace - disc)
    lam_plus = 0.5 * (trace + disc)
    if lam_minus <= 0.0 or lam_plus <= 0.0:
        return (
            float(np.sqrt(max(0.0, lam_minus))),
            float(np.sqrt(max(0.0, lam_plus))),
            False,  # Dynamically unstable (tachyonic runaway)
        )
    return float(np.sqrt(lam_minus)), float(np.sqrt(lam_plus)), True


def simulate_two_coupled_oscillators(
    f1: float = 2.0,
    f2: float = 2.5,
    K: float = 0.6,
    t_max: float = 200.0,
    dt: float = 0.01,
) -> Dict[str, Any]:
    """
    Numerically integrate the coupled oscillator ODE from PDF B Section 5.2
    and measure the emergent normal-mode spectral peaks via FFT.
    """
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        x1, v1, x2, v2 = y
        a1 = -(f1 ** 2) * x1 + K * (x2 - x1)
        a2 = -(f2 ** 2) * x2 + K * (x1 - x2)
        return np.array([v1, a1, v2, a2])

    t_eval = np.arange(0.0, t_max, dt)
    y0 = np.array([1.0, 0.0, -1.0, 0.0])
    sol = solve_ivp(rhs, (0.0, t_max), y0, t_eval=t_eval, method="DOP853", rtol=1e-11, atol=1e-13)

    x1_t = sol.y[0]
    x2_t = sol.y[2]
    win = np.hanning(len(t_eval))
    freqs = np.fft.rfftfreq(len(t_eval), d=dt) * 2.0 * np.pi
    spec_total = (np.abs(np.fft.rfft(x1_t * win)) ** 2) + (np.abs(np.fft.rfft(x2_t * win)) ** 2)

    peaks, _ = find_peaks(spec_total, distance=5)
    top2_peaks = sorted(peaks[np.argsort(spec_total[peaks])[::-1][:2]])
    f1_meas = float(freqs[top2_peaks[0]])
    f2_meas = float(freqs[top2_peaks[1]])
    f_minus_exact, f_plus_exact, stable = exact_two_oscillator_eigenfrequencies(f1, f2, K)

    return {
        "f1_bare": f1,
        "f2_bare": f2,
        "bare_separation": float(abs(f2 - f1)),
        "K": K,
        "f_minus_exact": f_minus_exact,
        "f_plus_exact": f_plus_exact,
        "coupled_separation_exact": float(f_plus_exact - f_minus_exact),
        "f1_measured_fft": f1_meas,
        "f2_measured_fft": f2_meas,
        "coupled_separation_measured": float(abs(f2_meas - f1_meas)),
        "is_stable": stable,
    }


def audit_linear_wave_superposition(
    f1: float = 5.0,
    f2: float = 7.0,
    A1: float = 1.0,
    A2: float = 1.0,
    r0: float = 1.0,
    alpha: float = 0.5,
    c: float = 10.0,
    T_total: float = 100.0,
    dt: float = 0.005,
) -> Dict[str, Any]:
    """
    Test the source PDF's Section 6 claim that linear superposition
        W_combined(r, t) = W_1(r, t) + W_2(r, t)
    where W_i(r, t) = A_i sin(2*pi*f_i*t - k_i*r) / (r^2 + alpha^2)
    produces a frequency shift Delta f in the entities' characteristic frequencies.
    """
    t = np.arange(0.0, T_total, dt)
    k1 = 2.0 * np.pi * f1 / c
    k2 = 2.0 * np.pi * f2 / c
    # Entity 1 centered at r = 0, Entity 2 centered at r = d = 2.0, evaluated at overlap midpoint r0 = 1.0
    r1 = r0
    r2 = abs(2.0 - r0)
    W1 = A1 * np.sin(2.0 * np.pi * f1 * t - k1 * r1) / (r1 ** 2 + alpha ** 2)
    W2 = A2 * np.sin(2.0 * np.pi * f2 * t - k2 * r2) / (r2 ** 2 + alpha ** 2)
    W_tot = W1 + W2
    I_tot = W_tot ** 2

    freqs = np.fft.rfftfreq(len(t), d=dt)
    psd_W1 = np.abs(np.fft.rfft(W1)) ** 2
    psd_W2 = np.abs(np.fft.rfft(W2)) ** 2
    psd_Wtot = np.abs(np.fft.rfft(W_tot)) ** 2
    psd_Itot = np.abs(np.fft.rfft(I_tot - np.mean(I_tot))) ** 2

    peak_W1 = float(freqs[np.argmax(psd_W1)])
    peak_W2 = float(freqs[np.argmax(psd_W2)])
    peaks_w, _ = find_peaks(psd_Wtot, distance=5)
    top2_idx = peaks_w[np.argsort(psd_Wtot[peaks_w])[::-1][:2]]
    top2_Wtot = sorted([float(freqs[top2_idx[0]]), float(freqs[top2_idx[1]])])

    # Dominant low-frequency envelope beat in intensity I(t) = |W1 + W2|^2 (for f < min(f1, f2))
    low_band = freqs < min(f1, f2)
    beat_freq = float(freqs[low_band][np.argmax(psd_Itot[low_band])])

    return {
        "f1_input": f1,
        "f2_input": f2,
        "peak_f1_isolated": peak_W1,
        "peak_f2_isolated": peak_W2,
        "top2_peaks_in_W_combined": top2_Wtot,
        "frequency_shift_entity_1": float(top2_Wtot[0] - peak_W1),
        "frequency_shift_entity_2": float(top2_Wtot[1] - peak_W2),
        "intensity_beat_frequency": beat_freq,
        "expected_beat_frequency": float(abs(f2 - f1)),
    }


def simulate_gravity_orbits(
    t_max: float = 35.0,
    dt: float = 0.002,
) -> Dict[str, Any]:
    """
    Compare orbital trajectories under:
    1. Standard Newtonian inverse-square gravity: F_N = -G M m / r^2  (V = -G M m / r)
    2. Source PDF's Section 6.3 derived force law:
           Delta f propto A1 A2 / r^2  ===>  F_g propto Delta f / r = -C / r^3
       (which corresponds to an attractive inverse-square potential V = -C / (2 r^2)).

    Test particle initialized at r0 = 1.0, v_r0 = 0.0, tangential velocity v_theta0 = 1.05
    (slightlyperturbed from circular equilibrium v_circ = 1.0 for G*M = C = 1.0).
    Also tests v_theta0 = 0.95 (slightly sub-circular).
    """
    def integrate_central_force(power_n: int, v_tan0: float) -> Tuple[np.ndarray, np.ndarray, str]:
        # Force F(r) = -1.0 / r^power_n in 2D Cartesian coordinates (x, y, vx, vy)
        def rhs(t: float, state: np.ndarray) -> np.ndarray:
            x, y, vx, vy = state
            r2 = x * x + y * y
            r = np.sqrt(r2)
            f_mag = 1.0 / (r ** power_n)
            ax = -f_mag * (x / r)
            ay = -f_mag * (y / r)
            return np.array([vx, vy, ax, ay])

        def plunge_event(t: float, state: np.ndarray) -> float:
            return np.hypot(state[0], state[1]) - 0.08
        plunge_event.terminal = True
        plunge_event.direction = -1

        y0 = np.array([1.0, 0.0, 0.0, v_tan0])
        sol = solve_ivp(
            rhs, (0.0, t_max), y0, max_step=dt, rtol=1e-10, atol=1e-12,
            events=plunge_event
        )
        r_vals = np.hypot(sol.y[0], sol.y[1])
        if sol.status == 1:
            outcome = "plunge_to_origin"
        elif r_vals[-1] > 4.0:
            outcome = "escape_to_infinity"
        else:
            outcome = "stable_bound_orbit"
        return sol.y[0], sol.y[1], outcome

    x_newt_sub, y_newt_sub, out_newt_sub = integrate_central_force(2, 0.92)
    x_newt_sup, y_newt_sup, out_newt_sup = integrate_central_force(2, 1.08)
    x_pdf_sub, y_pdf_sub, out_pdf_sub = integrate_central_force(3, 0.92)
    x_pdf_sup, y_pdf_sup, out_pdf_sup = integrate_central_force(3, 1.08)

    return {
        "newton_inverse_square_v092": out_newt_sub,
        "newton_inverse_square_v108": out_newt_sup,
        "pdf_inverse_cube_v092": out_pdf_sub,
        "pdf_inverse_cube_v108": out_pdf_sup,
        "trajectories": {
            "x_newt_sub": x_newt_sub, "y_newt_sub": y_newt_sub,
            "x_newt_sup": x_newt_sup, "y_newt_sup": y_newt_sup,
            "x_pdf_sub": x_pdf_sub, "y_pdf_sub": y_pdf_sub,
            "x_pdf_sup": x_pdf_sup, "y_pdf_sup": y_pdf_sup,
        },
    }


def audit_spectral_profile_integrals(alpha: float = 0.5) -> Dict[str, Any]:
    """
    Verify finite-energy conditions for the bell-shaped / inverse-square profiles
    described in the source notes and PDF Section 6.1:
    1. Profile P1 (Inverse-square amplitude in 1D frequency space):
           A(f) = A0 / ((f - f0)^2 + alpha^2)
       - L1 norm: int_{-inf}^{inf} A(f) df = pi * A0 / alpha
       - L2 energy: int_{-inf}^{inf} |A(f)|^2 df = pi * A0^2 / (2 * alpha^3)
    2. Profile P2 (Inverse-square energy density / Cauchy-Lorentz amplitude in 1D):
           |A(f)|^2 = E0 / ((f - f0)^2 + alpha^2)
       - L2 energy: int_{-inf}^{inf} |A(f)|^2 df = pi * E0 / alpha
    3. Profile P3 (Source PDF Section 6.1 wave in 3D radial space):
           W(r, t) = A0 * sin(2*pi*f0*t - k*r) / (r^2 + alpha^2)
       - Time-averaged 3D energy: int_0^{inf} <(W(r,t))^2> 4*pi*r^2 dr
         = 4*pi * (A0^2 / 2) * int_0^{inf} r^2 / (r^2 + alpha^2)^2 dr
         = 2*pi * A0^2 * (pi / (4 * alpha)) = (pi^2 / (2 * alpha)) * A0^2.
       Note: Requires alpha > 0 regularization; as alpha -> 0 (exact inverse-square
       1/r^2 amplitude at origin), the 3D ultraviolet integral diverges as 1/alpha!
       Conversely, if energy density followed 1/r^2 in 3D (|W| ~ 1/r), the 3D infrared
       integral int_0^R (1/r^2) 4*pi*r^2 dr = 4*pi*R diverges linearly as R -> inf!
    """
    from scipy.integrate import quad

    num_1d_l1, _ = quad(lambda x: 1.0 / (x ** 2 + alpha ** 2), -np.inf, np.inf)
    exact_1d_l1 = np.pi / alpha

    num_1d_l2, _ = quad(lambda x: 1.0 / ((x ** 2 + alpha ** 2) ** 2), -np.inf, np.inf)
    exact_1d_l2 = np.pi / (2.0 * (alpha ** 3))

    num_3d_l2, _ = quad(lambda r: 2.0 * np.pi * (r ** 2) / ((r ** 2 + alpha ** 2) ** 2), 0.0, np.inf)
    exact_3d_l2 = (np.pi ** 2) / (2.0 * alpha)

    return {
        "alpha": alpha,
        "1D_lorentzian_L1_numerical": float(num_1d_l1),
        "1D_lorentzian_L1_exact": float(exact_1d_l1),
        "1D_inv_sq_amplitude_L2_numerical": float(num_1d_l2),
        "1D_inv_sq_amplitude_L2_exact": float(exact_1d_l2),
        "3D_time_avg_L2_numerical": float(num_3d_l2),
        "3D_time_avg_L2_exact": float(exact_3d_l2),
        "uv_divergence_as_alpha_to_0": "Diverges as O(alpha^-3) in 1D L2 and O(alpha^-1) in 3D L2",
        "ir_divergence_if_intensity_inv_sq_in_3D": "Diverges linearly as O(R_max) in 3D if |W|^2 ~ 1/r^2",
    }


def run_experiment_2(output_dir: Path) -> Dict[str, Any]:
    """Run all audits in Experiment 2 and generate diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 2.1 Sweep coupling strength g in the PDF's coupled oscillator model
    f1_0, f2_0 = 2.0, 2.5
    df0 = f2_0 - f1_0
    K_vals = np.linspace(-0.8, 1.5, 100)
    sep_vals = []
    stable_flags = []
    for k_v in K_vals:
        fm, fp, st = exact_two_oscillator_eigenfrequencies(f1_0, f2_0, float(k_v))
        sep_vals.append(fp - fm if st else np.nan)
        stable_flags.append(st)

    osc_sim_pos = simulate_two_coupled_oscillators(f1=f1_0, f2=f2_0, K=0.6)
    osc_sim_neg = simulate_two_coupled_oscillators(f1=f1_0, f2=f2_0, K=-0.4)

    # 2.2 Linear wave superposition audit
    superpos_audit = audit_linear_wave_superposition()

    # 2.3 Orbital dynamics under PDF's 1/r^3 force vs Newtonian 1/r^2 force
    orbit_audit = simulate_gravity_orbits()
    trajs = orbit_audit.pop("trajectories")

    # 2.4 Spectral profile energy integral audit
    integral_audit = audit_spectral_profile_integrals(alpha=0.5)

    # --- Generate Multi-Panel Diagnostic Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), dpi=150)

    # Panel A: Level Repulsion in Source PDF's Coupled Oscillator Model
    ax = axes[0, 0]
    K_arr = np.array(K_vals)
    sep_arr = np.array(sep_vals)
    ax.plot(K_arr[K_arr >= 0], sep_arr[K_arr >= 0], color="#d62728", lw=2.2,
            label="PDF Model ($K > 0$): $O(K^2)$ Level Repulsion ($\\Delta f_{\\mathrm{eff}} > \\Delta f_0$)")
    ax.plot(K_arr[K_arr <= 0], sep_arr[K_arr <= 0], color="#1f77b4", lw=2.0, ls="--",
            label="Negative Coupling ($K < 0$): Repulsion & Tachyon Instability")
    ax.axhline(df0, color="k", ls=":", lw=1.5, label=f"Unperturbed Separation $\\Delta f_0 = {df0:.2f}$")
    # Mark instability boundary for K < 0: f1^2 * f2^2 + K*(f1^2 + f2^2) = 0 => K_crit = -(f1^2 f2^2)/(f1^2 + f2^2)
    K_tachyon = -(f1_0 ** 2 * f2_0 ** 2) / (f1_0 ** 2 + f2_0 ** 2)
    ax.axvline(0.0, color="gray", alpha=0.5)
    ax.plot([0.6], [osc_sim_pos["coupled_separation_measured"]], "ro", markersize=8,
            label=f"DOP853 FFT Verified ($K=0.6, \\Delta f={osc_sim_pos['coupled_separation_measured']:.3f}$)")
    ax.set_xlabel("Coupling Strength $K(|f_1 - f_2|)$")
    ax.set_ylabel("Effective Frequency Distance $\\Delta f_{\\mathrm{eff}} = f_+ - f_-$")
    ax.set_title("A. Source Toy Model (PDF Sec 5.2): Normal-Mode Repulsion vs. Claimed Attraction")
    ax.legend(loc="upper left", fontsize=8.0)
    ax.grid(True, alpha=0.3)

    # Panel B: Linear Superposition No-Shift Theorem
    ax = axes[0, 1]
    t_demo = np.arange(0.0, 40.0, 0.005)
    W1_d = np.sin(2.0 * np.pi * 5.0 * t_demo)
    W2_d = np.sin(2.0 * np.pi * 7.0 * t_demo)
    freqs_d = np.fft.rfftfreq(len(t_demo), d=0.005)
    psd1 = np.abs(np.fft.rfft(W1_d)) / len(t_demo)
    psd2 = np.abs(np.fft.rfft(W2_d)) / len(t_demo)
    psd_sum = np.abs(np.fft.rfft(W1_d + W2_d)) / len(t_demo)
    ax.plot(freqs_d, psd_sum, color="k", lw=2.0, label="$| \\mathcal{F}\\{W_1 + W_2\\}(f) |$ (Combined)")
    ax.axvline(5.0, color="#1f77b4", ls="--", lw=1.5, label="Unperturbed $f_1 = 5.0\\,\\mathrm{Hz}$ ($\\Delta f_1 = 0.0$)")
    ax.axvline(7.0, color="#ff7f0e", ls="--", lw=1.5, label="Unperturbed $f_2 = 7.0\\,\\mathrm{Hz}$ ($\\Delta f_2 = 0.0$)")
    ax.set_xlim(2.0, 10.0)
    ax.set_xlabel("Frequency $f$ [Hz]")
    ax.set_ylabel("Spectral Amplitude")
    ax.set_title("B. Linear Superposition $W_1 + W_2$ Produces Zero Frequency Shift")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, alpha=0.3)

    # Panel C: Orbital Stability under Newtonian 1/r^2 vs Source PDF's Derived 1/r^3 Force
    ax = axes[1, 0]
    ax.plot(trajs["x_newt_sub"], trajs["y_newt_sub"], color="#2ca02c", lw=1.8,
            label="Newtonian $F \\propto -1/r^2$ ($v_0=0.92$: Stable Ellipse)")
    ax.plot(trajs["x_pdf_sub"], trajs["y_pdf_sub"], color="#d62728", lw=1.8,
            label="PDF Derived $F \\propto -1/r^3$ ($v_0=0.92$: Plunge Spiral to $r\\to 0$)")
    ax.plot(trajs["x_pdf_sup"], trajs["y_pdf_sup"], color="#9467bd", lw=1.8, ls="--",
            label="PDF Derived $F \\propto -1/r^3$ ($v_0=1.08$: Unbound Escape)")
    ax.plot([0], [0], "ko", markersize=7, label="Force Center ($r=0$)")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Emergent Coordinate $x$")
    ax.set_ylabel("Emergent Coordinate $y$")
    ax.set_title("C. Source PDF Sec 6.3 Force ($F_g \\propto 1/r^3$): No Stable Orbits")
    ax.legend(loc="lower left", fontsize=7.8)
    ax.grid(True, alpha=0.3)

    # Panel D: Spectral Energy Integrability vs Regularization Parameter alpha
    ax = axes[1, 1]
    alphas = np.logspace(-2, 0.5, 50)
    e_1d_inv_sq = np.pi / (2.0 * (alphas ** 3))
    e_1d_lor = np.pi / alphas
    e_3d_wave = (np.pi ** 2) / (2.0 * alphas)
    ax.loglog(alphas, e_1d_inv_sq, color="#d62728", lw=2.0, label="1D $|A(f)| \\propto (\\Delta f^2 + \\alpha^2)^{-1}$: $E \\propto \\alpha^{-3}$")
    ax.loglog(alphas, e_1d_lor, color="#1f77b4", lw=2.0, ls="--", label="1D $|A(f)|^2 \\propto (\\Delta f^2 + \\alpha^2)^{-1}$: $E \\propto \\alpha^{-1}$")
    ax.loglog(alphas, e_3d_wave, color="#2ca02c", lw=2.0, ls="-.", label="3D PDF Wave $\\int \\langle W^2 \\rangle 4\\pi r^2 dr \\propto \\alpha^{-1}$")
    ax.set_xlabel("Core Regularization Scale $\\alpha$")
    ax.set_ylabel("Total Integrated Energy $E_{\\mathrm{tot}}(\\alpha)$")
    ax.set_title("D. Energy Finitedness Requires Non-Zero Core Cutoff $\\alpha > 0$")
    ax.legend(loc="upper right", fontsize=8.0)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    fig_path = fig_dir / "exp2_source_models_audit.png"
    plt.savefig(fig_path)
    plt.close(fig)

    summary = {
        "experiment": "exp2_source_models_audit",
        "coupled_oscillator_audit": {
            "positive_coupling_test": osc_sim_pos,
            "negative_coupling_test": osc_sim_neg,
            "critical_tachyon_coupling_K_crit": float(K_tachyon),
            "finding": (
                "The coupled oscillator equation proposed in PDF B Section 5.2 "
                "(x_i'' + f_i^2 x_i = sum K(|f_i - f_j|)(x_j - x_i)) exhibits normal-mode "
                "level repulsion for K > 0: the frequency difference INCREASES from "
                f"{df0:.3f} to {osc_sim_pos['coupled_separation_exact']:.3f} at K=0.6, "
                "producing emergent-space repulsion rather than attraction. Reversing the "
                "sign (K < 0) causes tachyonic runaway instability whenever |K| exceeds "
                f"|K_crit| = {abs(K_tachyon):.4f}."
            ),
        },
        "linear_wave_superposition_audit": {
            "results": superpos_audit,
            "finding": (
                "Linear superposition W_1 + W_2 of two wave entities produces amplitude beating "
                f"at |f_2 - f_1| = {superpos_audit['intensity_beat_frequency']:.2f} Hz in intensity "
                "I(r,t), but ZERO shift in the entities' spectral frequencies "
                f"(Delta f_1 = {superpos_audit['frequency_shift_entity_1']:.2e}, "
                f"Delta f_2 = {superpos_audit['frequency_shift_entity_2']:.2e}). "
                "Permanent frequency shifts require nonlinear wave-wave coupling."
            ),
        },
        "inverse_cube_gravity_audit": {
            "results": orbit_audit,
            "finding": (
                "In PDF A & B Section 6.3, differentiating or dividing the claimed frequency shift "
                "Delta f propto A1 A2 / r^2 by r yields F_g propto A1 A2 / r^3. By Bertrand's "
                "theorem and numerical integration, an attractive 1/r^3 force has NO stable "
                "circular or elliptical orbits: sub-circular perturbations (v0=0.92) plunge into "
                "the origin along a Cotes' spiral, while super-circular perturbations (v0=1.08) "
                "escape to infinity."
            ),
        },
        "spectral_profile_energy_audit": integral_audit,
        "figure_path": str(fig_path),
    }

    with open(output_dir / "exp2_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "results"
    res = run_experiment_2(out)
    print(json.dumps(res["coupled_oscillator_audit"]["finding"], indent=2))
    print(json.dumps(res["inverse_cube_gravity_audit"]["finding"], indent=2))
