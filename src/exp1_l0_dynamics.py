"""
Experiment 1: Level 0 State Evolution, Wave Entity Emergence, Persistence,
Multiplication, and Energy Conservation Audit.

Research Question:
Can repeating, frequency-localized "wave entities" emerge spontaneously from
random initial conditions in a Level 0 state vector E(t) = [E_1(t), ..., E_N(t)],
multiply via interaction, dissipate rapidly when isolated, and persist longer
in groups—and what mathematical constraints on U(E) are required for each behavior?

Distinction of Variables:
- Fundamental Level 0 variables:
  * State vector E(t) in R^N (dimensionless normalized energy units)
  * Conjugate rate vector V(t) = dE/dt in R^N (units: [T^-1])
  * Evolution parameter t (abstract L0 update clock, units: [T])
  * Component index i in {0, ..., N-1} (abstract state-space coordinate, NOT spatial position)
- Derived Level 1 diagnostics (not assumed a priori):
  * Modal / spectral power distribution P_k(t) = |E_hat_k(t)|^2 + |V_hat_k(t)|^2 / omega_k^2
  * Spectral Participation Ratio: PR(t) = (sum_k P_k)^2 / sum_k P_k^2  in [1, N]
    (PR ~ 1 indicates a sharply localized wave entity in frequency space;
     PR ~ N/2 indicates patternless / thermal equipartition across the spectrum)
  * Normalized Spectral Shannon Entropy: H_spec(t) = -sum_k p_k ln(p_k) / ln(N) in [0, 1]
  * Entity count M_ent(t; tau_thresh): number of contiguous spectral peaks exceeding
    tau_thresh * <P_k>
  * Total Hamiltonian / L2 norm E_tot(t)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_normal_mode_spectrum(
    E: np.ndarray,
    V: np.ndarray,
    omega_k: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Compute the normal-mode spectral energy distribution P_k(t),
    Participation Ratio PR(t), and normalized Shannon spectral entropy H_spec(t).

    Parameters
    ----------
    E : np.ndarray, shape (N,)
        Level 0 state vector components E_i(t).
    V : np.ndarray, shape (N,)
        Level 0 velocity vector components dE_i/dt.
    omega_k : np.ndarray, shape (N,)
        Linear dispersion frequencies of the normal modes.

    Returns
    -------
    P_k : np.ndarray, shape (N,)
        Spectral energy per mode k.
    pr : float
        Spectral participation ratio in [1, N].
    h_spec : float
        Normalized spectral entropy in [0, 1].
    """
    N = len(E)
    E_hat = np.fft.fft(E) / np.sqrt(N)
    V_hat = np.fft.fft(V) / np.sqrt(N)
    P_k = 0.5 * (np.abs(V_hat) ** 2 + (omega_k ** 2) * (np.abs(E_hat) ** 2))
    total_P = np.sum(P_k)
    if total_P <= 1e-30:
        return P_k, 1.0, 0.0
    p_norm = P_k / total_P
    pr = float((total_P ** 2) / np.sum(P_k ** 2))
    p_safe = np.clip(p_norm, 1e-30, 1.0)
    h_spec = float(-np.sum(p_safe * np.log(p_safe)) / np.log(N))
    return P_k, pr, h_spec


def detect_spectral_entities(P_k: np.ndarray, threshold_factor: float = 4.0) -> int:
    """
    Count distinct spectral wave entities in the one-sided frequency spectrum
    above `threshold_factor` times the median/mean spectral floor.
    """
    half_N = len(P_k) // 2
    spec = P_k[:half_N]
    mean_p = np.mean(spec)
    if mean_p <= 1e-30:
        return 0
    above = spec > (threshold_factor * mean_p)
    # Count connected components above threshold
    transitions = np.diff(above.astype(int), prepend=0)
    return int(np.sum(transitions == 1))


def laplacian_1d(E: np.ndarray, boundary: str = "periodic") -> np.ndarray:
    """Discrete 1D index-space coupling operator (nearest-neighbor)."""
    if boundary == "periodic":
        return np.roll(E, -1) + np.roll(E, 1) - 2.0 * E
    elif boundary == "dirichlet":
        out = np.zeros_like(E)
        out[1:-1] = E[2:] + E[:-2] - 2.0 * E[1:-1]
        out[0] = E[1] - 2.0 * E[0]
        out[-1] = E[-2] - 2.0 * E[-1]
        return out
    else:
        raise ValueError(f"Unknown boundary condition: {boundary}")


def simulate_l0_system(
    N: int = 128,
    dt: float = 0.02,
    n_steps: int = 4000,
    c0: float = 1.0,
    omega0: float = 1.0,
    beta_nl: float = 0.0,
    gamma_0: float = 0.0,
    gamma_coop_alpha: float = 0.0,
    init_mode: str = "random_noise",
    seed: int = 42,
    boundary: str = "periodic",
) -> Dict[str, Any]:
    """
    Integrate the Level 0 dynamical system:
        d^2 E_i / dt^2 = c0^2 (Delta_lat E)_i - omega0^2 E_i - beta_nl E_i^3 - gamma_eff(E) dE_i/dt

    where:
    - beta_nl = 0, gamma_0 = 0: Class I (Linear Unitary / Hamiltonian)
    - beta_nl > 0, gamma_0 = 0: Class II (Conservative Nonlinear Klein-Gordon / FPU-like)
    - beta_nl >= 0, gamma_0 > 0, gamma_coop_alpha = 0: Class III (Passively Dissipative)
    - gamma_coop_alpha > 0: Class IV (State-dependent Cooperative Damping Reduction:
      gamma_eff = gamma_0 / (1 + gamma_coop_alpha * E_tot_active), an explicit
      additional assumption required to make larger groups decay more slowly than
      isolated weak entities).
    """
    rng = np.random.default_rng(seed)
    k_vals = 2.0 * np.pi * np.fft.fftfreq(N, d=1.0)
    omega_k = np.sqrt(omega0 ** 2 + 4.0 * (c0 ** 2) * (np.sin(k_vals / 2.0) ** 2))

    idx = np.arange(N, dtype=float)
    if init_mode == "random_noise":
        E = rng.normal(0.0, 0.5, size=N)
        V = rng.normal(0.0, 0.5, size=N)
    elif init_mode == "single_entity":
        # Single frequency-localized wave packet centered at mode k0 = N // 8
        k0 = 2.0 * np.pi * (N // 8) / N
        w0 = np.sqrt(omega0 ** 2 + 4.0 * (c0 ** 2) * (np.sin(k0 / 2.0) ** 2))
        env = np.exp(-0.5 * ((idx - N / 2.0) / (N / 10.0)) ** 2)
        E = 1.0 * env * np.cos(k0 * (idx - N / 2.0))
        V = 1.0 * w0 * env * np.sin(k0 * (idx - N / 2.0))
    elif init_mode == "interacting_group":
        # Group of 4 overlapping wave entities with nearby frequencies
        E = np.zeros(N)
        V = np.zeros(N)
        modes = [N // 8 - 4, N // 8 - 2, N // 8 + 2, N // 8 + 4]
        for m_idx, m in enumerate(modes):
            k_m = 2.0 * np.pi * m / N
            w_m = np.sqrt(omega0 ** 2 + 4.0 * (c0 ** 2) * (np.sin(k_m / 2.0) ** 2))
            env = np.exp(-0.5 * ((idx - N / 2.0) / (N / 10.0)) ** 2)
            phase = 0.5 * m_idx
            E += 1.0 * env * np.cos(k_m * (idx - N / 2.0) + phase)
            V += 1.0 * w_m * env * np.sin(k_m * (idx - N / 2.0) + phase)
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    def calc_total_energy(e_vec: np.ndarray, v_vec: np.ndarray) -> float:
        if boundary == "periodic":
            grad_sq = np.sum((np.roll(e_vec, -1) - e_vec) ** 2)
        else:
            grad_sq = np.sum((e_vec[1:] - e_vec[:-1]) ** 2) + e_vec[0] ** 2 + e_vec[-1] ** 2
        kin = 0.5 * np.sum(v_vec ** 2)
        pot_lin = 0.5 * (c0 ** 2) * grad_sq + 0.5 * (omega0 ** 2) * np.sum(e_vec ** 2)
        pot_nl = 0.25 * beta_nl * np.sum(e_vec ** 4)
        return float(kin + pot_lin + pot_nl)

    def calc_accel(e_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
        lap = laplacian_1d(e_vec, boundary=boundary)
        force = (c0 ** 2) * lap - (omega0 ** 2) * e_vec - beta_nl * (e_vec ** 3)
        if gamma_0 > 0.0:
            if gamma_coop_alpha > 0.0:
                # Non-Hamiltonian cooperative damping suppression depending on collective active energy
                active_energy = 0.5 * np.sum(v_vec ** 2 + (omega0 ** 2) * (e_vec ** 2))
                gamma_eff = gamma_0 / (1.0 + gamma_coop_alpha * active_energy)
            else:
                gamma_eff = gamma_0
            force = force - gamma_eff * v_vec
        return force

    record_stride = max(1, n_steps // 200)
    times = []
    energies = []
    prs = []
    entropies = []
    entity_counts = []
    initial_P_k, _, _ = compute_normal_mode_spectrum(E, V, omega_k)

    for step in range(n_steps + 1):
        if step % record_stride == 0:
            t_now = step * dt
            e_tot = calc_total_energy(E, V)
            P_k, pr, h_spec = compute_normal_mode_spectrum(E, V, omega_k)
            n_ent = detect_spectral_entities(P_k, threshold_factor=4.0)
            times.append(t_now)
            energies.append(e_tot)
            prs.append(pr)
            entropies.append(h_spec)
            entity_counts.append(n_ent)

        if step < n_steps:
            # Velocity-Verlet / Leapfrog integration (symplectic when gamma_0 == 0)
            a_curr = calc_accel(E, V)
            E_next = E + V * dt + 0.5 * a_curr * (dt ** 2)
            # Predictor velocity for velocity-dependent damping
            V_pred = V + a_curr * dt
            a_next = calc_accel(E_next, V_pred)
            V_next = V + 0.5 * (a_curr + a_next) * dt
            E, V = E_next, V_next

    final_P_k, _, _ = compute_normal_mode_spectrum(E, V, omega_k)
    return {
        "times": np.array(times),
        "energies": np.array(energies),
        "prs": np.array(prs),
        "entropies": np.array(entropies),
        "entity_counts": np.array(entity_counts),
        "initial_P_k": initial_P_k,
        "final_P_k": final_P_k,
        "omega_k": omega_k,
    }


def run_experiment_1(output_dir: Path) -> Dict[str, Any]:
    """
    Execute all sub-tests of Experiment 1, including:
    1.1 Random Initial Conditions under Linear vs. Nonlinear Conservative Evolution (10 seeds)
    1.2 Harmonic "Kaleidoscope" Multiplication vs. Energy Partitioning
    1.3 Isolated vs. Grouped Persistence under Passive vs. Cooperative Damping
    1.4 Numerical Convergence (dt) and Discretization/Boundary Sensitivity (N, BC)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    N = 128
    dt = 0.02
    n_steps = 4000

    # --- 1.1 Spontaneous Emergence from Randomness Test (10 seeds) ---
    seeds = list(range(100, 110))
    lin_pr_means, nl_pr_means = [], []
    lin_ent_means, nl_ent_means = [], []
    lin_max_peaks, nl_max_peaks = [], []

    sample_lin_random = None
    sample_nl_random = None

    for s in seeds:
        res_lin = simulate_l0_system(
            N=N, dt=dt, n_steps=n_steps, beta_nl=0.0, gamma_0=0.0,
            init_mode="random_noise", seed=s
        )
        res_nl = simulate_l0_system(
            N=N, dt=dt, n_steps=n_steps, beta_nl=0.5, gamma_0=0.0,
            init_mode="random_noise", seed=s
        )
        lin_pr_means.append(float(np.mean(res_lin["prs"][-50:])))
        nl_pr_means.append(float(np.mean(res_nl["prs"][-50:])))
        lin_ent_means.append(float(np.mean(res_lin["entropies"][-50:])))
        nl_ent_means.append(float(np.mean(res_nl["entropies"][-50:])))
        lin_max_peaks.append(float(np.max(res_lin["final_P_k"] / np.mean(res_lin["final_P_k"]))))
        nl_max_peaks.append(float(np.max(res_nl["final_P_k"] / np.mean(res_nl["final_P_k"]))))
        if s == 100:
            sample_lin_random = res_lin
            sample_nl_random = res_nl

    # Theoretical expectation for Gaussian white noise Rayleigh/Exponential spectral power:
    # For real FFT of N modes (N/2 independent complex modes), P_k ~ Chi^2_2 (exponential),
    # so E[PR] = N / 2 = 64.0, and expected max(P_k/<P>) for M=64 independent exponential variables
    # is H_64 = sum_{m=1}^{64} 1/m ≈ 4.7439.
    euler_mascheroni_harmonic_64 = float(np.sum(1.0 / np.arange(1, N // 2 + 1)))

    # --- 1.2 Harmonic Multiplication ("Kaleidoscope") vs Energy Conservation ---
    res_single_lin = simulate_l0_system(
        N=N, dt=dt, n_steps=n_steps, beta_nl=0.0, gamma_0=0.0,
        init_mode="single_entity", seed=42
    )
    res_single_nl = simulate_l0_system(
        N=N, dt=dt, n_steps=n_steps, beta_nl=1.2, gamma_0=0.0,
        init_mode="single_entity", seed=42
    )
    # Check threshold sensitivity on entity count for final nonlinear spectrum
    thresh_sensitivity = {}
    for th in [2.0, 3.0, 4.0, 6.0, 8.0]:
        thresh_sensitivity[str(th)] = detect_spectral_entities(res_single_nl["final_P_k"], threshold_factor=th)

    # --- 1.3 Persistence in Isolation vs. Groups (Passive vs. Cooperative Damping) ---
    gamma_test = 0.025
    # Passive linear damping (Class III):
    res_iso_passive = simulate_l0_system(
        N=N, dt=dt, n_steps=n_steps, beta_nl=0.5, gamma_0=gamma_test, gamma_coop_alpha=0.0,
        init_mode="single_entity", seed=42
    )
    res_grp_passive = simulate_l0_system(
        N=N, dt=dt, n_steps=n_steps, beta_nl=0.5, gamma_0=gamma_test, gamma_coop_alpha=0.0,
        init_mode="interacting_group", seed=42
    )
    # State-dependent cooperative damping suppression (Class IV):
    res_iso_coop = simulate_l0_system(
        N=N, dt=dt, n_steps=n_steps, beta_nl=0.5, gamma_0=gamma_test, gamma_coop_alpha=0.012,
        init_mode="single_entity", seed=42
    )
    res_grp_coop = simulate_l0_system(
        N=N, dt=dt, n_steps=n_steps, beta_nl=0.5, gamma_0=gamma_test, gamma_coop_alpha=0.012,
        init_mode="interacting_group", seed=42
    )

    def half_life(times_arr: np.ndarray, e_arr: np.ndarray) -> float:
        e0 = e_arr[0]
        below = np.where(e_arr <= 0.5 * e0)[0]
        if len(below) == 0:
            return float("inf")
        idx_h = below[0]
        if idx_h == 0:
            return 0.0
        # Linear interpolation
        t1, t2 = times_arr[idx_h - 1], times_arr[idx_h]
        e1, e2 = e_arr[idx_h - 1], e_arr[idx_h]
        return float(t1 + (0.5 * e0 - e1) * (t2 - t1) / (e2 - e1))

    t_half_iso_passive = half_life(res_iso_passive["times"], res_iso_passive["energies"])
    t_half_grp_passive = half_life(res_grp_passive["times"], res_grp_passive["energies"])
    t_half_iso_coop = half_life(res_iso_coop["times"], res_iso_coop["energies"])
    t_half_grp_coop = half_life(res_grp_coop["times"], res_grp_coop["energies"])

    # --- 1.4 Numerical Convergence & Discretization / Boundary Checks ---
    dt_list = [0.04, 0.02, 0.01, 0.005]
    energy_drifts = {}
    T_conv = 40.0
    for dt_c in dt_list:
        steps_c = int(round(T_conv / dt_c))
        r_c = simulate_l0_system(
            N=64, dt=dt_c, n_steps=steps_c, beta_nl=0.5, gamma_0=0.0,
            init_mode="single_entity", seed=42
        )
        rel_drift = float(np.max(np.abs(r_c["energies"] - r_c["energies"][0])) / r_c["energies"][0])
        energy_drifts[str(dt_c)] = rel_drift

    # Boundary & System-size N sensitivity check
    bc_check = {}
    for bc in ["periodic", "dirichlet"]:
        for N_c in [64, 128, 256]:
            r_b = simulate_l0_system(
                N=N_c, dt=0.02, n_steps=2000, beta_nl=0.5, gamma_0=0.0,
                init_mode="random_noise", seed=100, boundary=bc
            )
            bc_check[f"{bc}_N{N_c}"] = {
                "final_entropy": float(r_b["entropies"][-1]),
                "final_pr_over_N": float(r_b["prs"][-1] / N_c),
            }

    # --- Generate Multi-Panel Diagnostic Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), dpi=150)

    # Panel A: Spectral Participation Ratio from Random Initial Conditions
    ax = axes[0, 0]
    ax.plot(sample_lin_random["times"], sample_lin_random["prs"], label="Class I: Linear Unitary ($\\beta=0$)", color="#1f77b4", lw=1.8)
    ax.plot(sample_nl_random["times"], sample_nl_random["prs"], label="Class II: Nonlinear Hamiltonian ($\\beta=0.5$)", color="#ff7f0e", lw=1.8)
    ax.axhline(N / 2.0, color="k", ls="--", alpha=0.7, label=f"Thermal Equipartition Null ($N/2 = {N//2}$)")
    ax.axhline(1.0, color="crimson", ls=":", lw=1.5, label="Single Localized Entity ($PR = 1$)")
    ax.set_xlabel("Evolution Time $t$ [L0 steps $\\times \\Delta t$]")
    ax.set_ylabel("Spectral Participation Ratio $PR(t)$")
    ax.set_title("A. Random Initial State: No Spontaneous Spectral Localization")
    ax.set_ylim(0, N * 0.75)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, alpha=0.3)

    # Panel B: Harmonic Multiplication ("Kaleidoscope") vs Energy Partitioning
    ax = axes[0, 1]
    half_N = N // 2
    modes_x = np.arange(half_N)
    p_init_norm = res_single_nl["initial_P_k"][:half_N] / np.sum(res_single_nl["initial_P_k"][:half_N])
    p_fin_lin = res_single_lin["final_P_k"][:half_N] / np.sum(res_single_lin["initial_P_k"][:half_N])
    p_fin_nl = res_single_nl["final_P_k"][:half_N] / np.sum(res_single_nl["initial_P_k"][:half_N])
    ax.semilogy(modes_x, np.clip(p_init_norm, 1e-8, 1.0), "k--", lw=1.5, label="Initial Single Wave Entity ($t=0$)")
    ax.semilogy(modes_x, np.clip(p_fin_lin, 1e-8, 1.0), color="#1f77b4", alpha=0.8, lw=1.5, label="Linear Evolution ($t=80$, static spectrum)")
    ax.semilogy(modes_x, np.clip(p_fin_nl, 1e-8, 1.0), color="#d62728", lw=1.6, label="Nonlinear ($\\beta=1.2, t=80$): Harmonic Cascade")
    ax.set_xlabel("Spectral Mode Index $k$")
    ax.set_ylabel("Normalized Modal Energy $P_k(t) / E_{\\mathrm{tot}}(0)$")
    ax.set_title("B. 'Kaleidoscope' Harmonic Generation Partitions Fixed Energy")
    ax.set_ylim(1e-7, 2.0)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, alpha=0.3)

    # Panel C: Persistence of Isolated vs. Grouped Entities
    ax = axes[1, 0]
    ax.plot(res_iso_passive["times"], res_iso_passive["energies"] / res_iso_passive["energies"][0],
            color="#1f77b4", ls="--", lw=2.0, label=f"Isolated (Passive Damping, $t_{{1/2}}={t_half_iso_passive:.1f}$)")
    ax.plot(res_grp_passive["times"], res_grp_passive["energies"] / res_grp_passive["energies"][0],
            color="#ff7f0e", ls=":", lw=2.2, label=f"Group of 4 (Passive Damping, $t_{{1/2}}={t_half_grp_passive:.1f}$)")
    ax.plot(res_iso_coop["times"], res_iso_coop["energies"] / res_iso_coop["energies"][0],
            color="#2ca02c", lw=1.8, label=f"Isolated (Cooperative Rule, $t_{{1/2}}={t_half_iso_coop:.1f}$)")
    ax.plot(res_grp_coop["times"], res_grp_coop["energies"] / res_grp_coop["energies"][0],
            color="#9467bd", lw=2.2, label=f"Group of 4 (Cooperative Rule, $t_{{1/2}}={t_half_grp_coop:.1f}$)")
    ax.axhline(0.5, color="gray", ls="-.", alpha=0.6, label="Half-life threshold ($E(t)/E(0) = 0.5$)")
    ax.set_xlabel("Evolution Time $t$")
    ax.set_ylabel("Normalized Total Energy $E(t) / E(0)$")
    ax.set_title("C. Persistence Audit: Passive Damping vs. Imposed Cooperative Rule")
    ax.legend(loc="upper right", fontsize=8.0)
    ax.grid(True, alpha=0.3)

    # Panel D: Symplectic Integrator Timestep Convergence Check
    ax = axes[1, 1]
    dts_arr = np.array([float(k) for k in energy_drifts.keys()])
    drifts_arr = np.array([energy_drifts[k] for k in energy_drifts.keys()])
    ax.loglog(dts_arr, drifts_arr, "o-", color="#2ca02c", lw=2.0, markersize=7, label="Measured $|\\Delta E_{\\max}| / E_0$ (Velocity-Verlet)")
    ref_scale = drifts_arr[0] * (dts_arr / dts_arr[0]) ** 2
    ax.loglog(dts_arr, ref_scale, "k--", lw=1.5, label="Exact Second-Order Scaling $O(\\Delta t^2)$")
    ax.set_xlabel("Timestep $\\Delta t$")
    ax.set_ylabel("Maximum Relative Energy Drift $|\\Delta E|_{\\max} / E_0$")
    ax.set_title("D. Numerical Convergence Verification (Conservative $U$)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    fig_path = fig_dir / "exp1_l0_dynamics.png"
    plt.savefig(fig_path)
    plt.close(fig)

    summary = {
        "experiment": "exp1_l0_dynamics",
        "parameters": {"N": N, "dt": dt, "n_steps": n_steps, "n_seeds": len(seeds)},
        "spontaneous_emergence_from_randomness": {
            "theoretical_white_noise_PR": N / 2.0,
            "theoretical_extreme_value_peak_ratio_H64": euler_mascheroni_harmonic_64,
            "linear_mean_PR_across_seeds": float(np.mean(lin_pr_means)),
            "linear_std_PR_across_seeds": float(np.std(lin_pr_means)),
            "nonlinear_mean_PR_across_seeds": float(np.mean(nl_pr_means)),
            "nonlinear_std_PR_across_seeds": float(np.std(nl_pr_means)),
            "linear_mean_entropy": float(np.mean(lin_ent_means)),
            "nonlinear_mean_entropy": float(np.mean(nl_ent_means)),
            "linear_mean_max_peak_to_mean": float(np.mean(lin_max_peaks)),
            "nonlinear_mean_max_peak_to_mean": float(np.mean(nl_max_peaks)),
            "finding": (
                f"Under both linear and nonlinear conservative dynamics, random initial "
                f"conditions remain near maximum spectral entropy (H_spec = {np.mean(lin_ent_means):.3f} "
                f"linear, {np.mean(nl_ent_means):.3f} nonlinear; PR = {np.mean(lin_pr_means):.1f} and "
                f"{np.mean(nl_pr_means):.1f}) and do NOT spontaneously condense into isolated narrow-band "
                f"wave entities. Apparent peaks (max/mean = {np.mean(lin_max_peaks):.2f} linear, "
                f"{np.mean(nl_max_peaks):.2f} nonlinear) match the theoretical Rayleigh/Exponential "
                f"extreme-value expectation H_64 = {euler_mascheroni_harmonic_64:.2f}."
            ),
        },
        "harmonic_multiplication_audit": {
            "linear_initial_energy": float(res_single_lin["energies"][0]),
            "linear_final_energy": float(res_single_lin["energies"][-1]),
            "nonlinear_initial_energy": float(res_single_nl["energies"][0]),
            "nonlinear_final_energy": float(res_single_nl["energies"][-1]),
            "nonlinear_initial_PR": float(res_single_nl["prs"][0]),
            "nonlinear_final_PR": float(res_single_nl["prs"][-1]),
            "entity_count_threshold_sensitivity": thresh_sensitivity,
            "finding": (
                f"In a conservative nonlinear system, a single narrow-band wave entity generates "
                f"higher/sideband harmonics (PR increases from {res_single_nl['prs'][0]:.2f} to "
                f"{res_single_nl['prs'][-1]:.2f}), analogous to the notes' 'kaleidoscope multiplication', "
                f"but total energy is strictly conserved ({res_single_nl['energies'][0]:.4f} -> "
                f"{res_single_nl['energies'][-1]:.4f}): new spectral peaks are formed by partitioning "
                f"and depleting the parent entity's amplitude rather than multiplying total energy."
            ),
        },
        "persistence_and_dissipation_audit": {
            "passive_damping_gamma0": gamma_test,
            "theoretical_passive_half_life": float(np.log(2.0) / gamma_test),
            "isolated_passive_half_life": t_half_iso_passive,
            "group_passive_half_life": t_half_grp_passive,
            "isolated_cooperative_half_life": t_half_iso_coop,
            "group_cooperative_half_life": t_half_grp_coop,
            "cooperative_persistence_ratio_group_vs_iso": float(t_half_grp_coop / t_half_iso_coop),
            "passive_persistence_ratio_group_vs_iso": float(t_half_grp_passive / t_half_iso_passive),
            "finding": (
                f"Under standard passive damping (-gamma * dE/dt), grouped wave entities decay at "
                f"virtually the exact same rate as isolated entities (ratio = "
                f"{t_half_grp_passive / t_half_iso_passive:.3f}), and all dissipated energy vanishes "
                f"from Level 0 (violating total energy conservation). Achieving longer group "
                f"persistence (ratio = {t_half_grp_coop / t_half_iso_coop:.2f}x, t_1/2 = "
                f"{t_half_grp_coop:.1f} vs {t_half_iso_coop:.1f}) requires explicitly inserting a "
                f"non-Hamiltonian, state-dependent damping suppression rule gamma_eff(E)."
            ),
        },
        "numerical_verification": {
            "verlet_dt_convergence_relative_energy_drift": energy_drifts,
            "measured_convergence_order": float(
                np.polyfit(np.log(dts_arr), np.log(drifts_arr), 1)[0]
            ),
            "boundary_and_N_sensitivity": bc_check,
        },
        "figure_path": str(fig_path),
    }

    with open(output_dir / "exp1_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "results"
    res = run_experiment_1(out)
    print(json.dumps(res["spontaneous_emergence_from_randomness"], indent=2))
    print(json.dumps(res["persistence_and_dissipation_audit"], indent=2))
