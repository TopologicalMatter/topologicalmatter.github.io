"""
Automated Verification and Unit Test Suite for the Wave Entities Research Codebase.

Tests mathematical identities, numerical convergence, conservation laws,
dimensionality theorems, and reproducibility across all four experiment modules.
"""

from __future__ import annotations
import numpy as np
import pytest

from src.exp1_l0_dynamics import (
    compute_normal_mode_spectrum,
    detect_spectral_entities,
    simulate_l0_system,
)
from src.exp2_source_models_audit import (
    exact_two_oscillator_eigenfrequencies,
    simulate_two_coupled_oscillators,
    audit_linear_wave_superposition,
    audit_spectral_profile_integrals,
)
from src.exp3_emergent_space_geometry import (
    classical_mds_eigenvalues,
    cayley_menger_3d_volume_squared,
    evaluate_momentum_vs_inertia_interpretations,
)
from src.exp4_cosmology_cmb import (
    planck_energy_spectrum,
    fit_best_planck_temperature_and_scale,
    spectral_moments,
    evolve_advection_diffusion_spectrum,
    audit_atomic_line_achromaticity,
)
from src.exp5_approximate_wave_entity_model import (
    compute_qball_energetics_and_profiles,
    simulate_triadic_phase_locking,
    audit_relativistic_wave_entity_kinematics,
    audit_long_range_mediator_gravity,
)


def test_exp1_symplectic_energy_conservation():
    """Verify that Conservative Class I and II systems conserve energy to O(dt^2)."""
    res_coarse = simulate_l0_system(N=64, dt=0.02, n_steps=500, beta_nl=0.5, gamma_0=0.0, init_mode="single_entity")
    res_fine = simulate_l0_system(N=64, dt=0.005, n_steps=2000, beta_nl=0.5, gamma_0=0.0, init_mode="single_entity")
    drift_coarse = np.max(np.abs(res_coarse["energies"] - res_coarse["energies"][0])) / res_coarse["energies"][0]
    drift_fine = np.max(np.abs(res_fine["energies"] - res_fine["energies"][0])) / res_fine["energies"][0]
    assert drift_fine < 1e-5
    # Refining dt by 4x should reduce energy drift by ~16x (second-order symplectic Verlet)
    assert drift_coarse / drift_fine > 12.0


def test_exp1_linear_spectral_stationarity():
    """In a linear time-invariant L0 system, normal-mode energies P_k are exact constants of motion."""
    res = simulate_l0_system(N=64, dt=0.01, n_steps=1000, beta_nl=0.0, gamma_0=0.0, init_mode="random_noise", seed=7)
    np.testing.assert_allclose(res["initial_P_k"], res["final_P_k"], rtol=1e-3, atol=1e-5)


def test_exp2_coupled_oscillator_level_repulsion():
    """Verify that positive diffusive coupling K > 0 in the PDF's oscillator model increases frequency separation."""
    f1, f2 = 2.0, 2.5
    fm, fp, stable = exact_two_oscillator_eigenfrequencies(f1, f2, K=0.6)
    assert stable is True
    assert (fp - fm) > (f2 - f1)

    sim = simulate_two_coupled_oscillators(f1=f1, f2=f2, K=0.6, t_max=150.0, dt=0.01)
    assert abs(sim["coupled_separation_measured"] - sim["coupled_separation_exact"]) < 0.05


def test_exp2_linear_superposition_zero_shift():
    """Verify that linear wave superposition W1 + W2 produces zero frequency shift in spectral peaks."""
    audit = audit_linear_wave_superposition(f1=5.0, f2=7.0, T_total=50.0, dt=0.005)
    assert abs(audit["frequency_shift_entity_1"]) < 1e-12
    assert abs(audit["frequency_shift_entity_2"]) < 1e-12
    assert abs(audit["intensity_beat_frequency"] - 2.0) < 1e-12


def test_exp2_spectral_integrals_exact_match():
    """Verify numerical vs analytical L1 and L2 integrals of inverse-square profiles."""
    res = audit_spectral_profile_integrals(alpha=0.5)
    assert abs(res["1D_lorentzian_L1_numerical"] - res["1D_lorentzian_L1_exact"]) < 1e-10
    assert abs(res["1D_inv_sq_amplitude_L2_numerical"] - res["1D_inv_sq_amplitude_L2_exact"]) < 1e-10
    assert abs(res["3D_time_avg_L2_numerical"] - res["3D_time_avg_L2_exact"]) < 1e-10


def test_exp3_scalar_frequency_distance_is_strictly_1d():
    """Prove numerically that scalar frequency distance d_ij = |f_i - f_j| has Rank-1 Gram matrix and zero 3D volume."""
    f = np.array([1.0, 2.4, 5.1, 8.9, 12.3, 15.0])
    D = np.abs(f[:, None] - f[None, :])
    eigs, ratios = classical_mds_eigenvalues(D)
    assert abs(ratios[0] - 1.0) < 1e-14
    assert np.all(np.abs(ratios[1:]) < 1e-14)
    vol_sq = cayley_menger_3d_volume_squared(D[:4, :4])
    assert abs(vol_sq) < 1e-14


def test_exp3_momentum_vs_inertia_mechanics():
    """Verify that resistance to df/dt stops immediately after force ceases, unlike Newtonian inertia."""
    kin = evaluate_momentum_vs_inertia_interpretations()
    assert abs(kin["post_pulse_velocity_t5"]["case1_tethered_Delta_f"]) < 1e-12
    assert abs(kin["post_pulse_velocity_t5"]["case2_aristotelian_df_dt"]) < 1e-12
    assert abs(kin["post_pulse_velocity_t5"]["case3_newtonian_d2f_dt2"] - 2.0) < 0.05


def test_exp4_planck_adiabatic_scaling_narrows_width():
    """Verify that cooling a Planck blackbody from T=2 to T=1 halves its spectral width and preserves Planck shape."""
    f = np.linspace(0.01, 25.0, 1000)
    u2 = planck_energy_spectrum(f, 2.0)
    u1 = planck_energy_spectrum(f, 1.0)
    m2, s2 = spectral_moments(f, u2)
    m1, s1 = spectral_moments(f, u1)
    assert abs((s1 / s2) - 0.5) < 0.01
    T_fit, rms_err, max_err, _ = fit_best_planck_temperature_and_scale(f, u1)
    assert abs(T_fit - 1.0) < 1e-4
    assert max_err < 1e-6


def test_exp4_advection_diffusion_stability():
    """Verify that the CFL-substepped finite-volume advection-diffusion solver is stable and finite."""
    f = np.linspace(0.02, 25.0, 600)
    u0 = np.exp(-0.5 * ((f - 10.0) / 0.8) ** 2)
    u_fin = evolve_advection_diffusion_spectrum(f, u0, v_drift=3.0, D_diff=2.0, t_end=1.5)
    assert np.all(np.isfinite(u_fin))
    assert np.all(u_fin >= 0.0)
    # Total integrated energy conserved to machine precision under zero-flux boundaries
    np.testing.assert_allclose(np.sum(u_fin), np.sum(u0), rtol=1e-10)


def test_exp4_atomic_doublet_achromaticity():
    """Verify that multiplicative scaling preserves doublet ratios while additive/nonlinear shifts violate them."""
    res = audit_atomic_line_achromaticity()
    assert res["flrw_multiplicative"]["ratio_error"] < 1e-14
    assert res["flrw_multiplicative"]["achromatic_dz"] < 1e-14
    assert res["additive_frequency_shift"]["ratio_error"] > 0.3
    assert res["nonlinear_dispersive_shift_f2"]["ratio_error"] > 0.1


def test_exp5_qball_stability_threshold():
    """Verify that in AWEM, a single Wave Entity (Q=35) is above the mass gap while a 3-Entity Proton (Q=105) is bound below it."""
    res = compute_qball_energetics_and_profiles()
    assert res["single_entity_Q35"]["E_over_m0Q"] > 1.0
    assert res["single_entity_Q35"]["radiation_decay_rate"] > 0.0
    assert res["composite_proton_Q105"]["E_over_m0Q"] < 1.0
    assert res["composite_proton_Q105"]["omega_over_m0"] < 1.0
    assert res["composite_proton_Q105"]["radiation_decay_rate"] == 0.0


def test_exp5_triadic_phase_locking_and_relativistic_kinematics():
    """Verify 120-degree triadic phase neutralization, relativistic momentum integral, and 1/r^2 mediator gravity."""
    triad = simulate_triadic_phase_locking()
    assert triad["N3_final_total_color_dipole_norm"] < 1e-3
    np.testing.assert_allclose(triad["N3_pairwise_phase_angles_deg"], [120.0, 120.0, 120.0], atol=0.1)

    rel = audit_relativistic_wave_entity_kinematics()
    assert rel["max_relative_momentum_error"] < 1e-12

    grav = audit_long_range_mediator_gravity()
    assert abs(grav["far_field_freq_shift_exponent"] - (-1.0)) < 1e-3
    assert abs(grav["far_field_force_exponent"] - (-2.0)) < 1e-3
    assert grav["orbit_is_stable_bound_ellipse"] is True


def test_exp6_topological_quantization_and_spin_half_rotor():
    """Verify exact integer topological winding B=1, Derrick virial equilibrium, and Finkelstein-Rubinstein spin-1/2."""
    from src.exp6_evolved_topological_wave_model import (
        solve_topological_skyrme_wave_entity,
        quantize_finkelstein_rubinstein_rotor,
    )
    skyrme = solve_topological_skyrme_wave_entity()
    assert abs(skyrme["B_measured"] - 1.0) < 1e-5
    assert abs(skyrme["B_perturbed"] - 1.0) < 1e-4
    assert skyrme["virial_theorem_relative_residual"] < 1e-14

    rotor = quantize_finkelstein_rubinstein_rotor(skyrme)
    assert abs(rotor["phase_factor_at_2pi_rotation"] - (-1.0)) < 1e-12
    assert abs(rotor["phase_factor_at_4pi_rotation"] - 1.0) < 1e-12
    assert abs(rotor["pauli_exchange_sign"] - (-1.0)) < 1e-12
    assert rotor["mu_ratio_relative_error_pct"] < 5.0


def test_exp6_tensor_refractive_gr_and_moduli_bell_violation():
    """Verify PPN gamma=1 light bending, perihelion precession, and S = 2*sqrt(2) Bell violation on moduli space."""
    from src.exp6_evolved_topological_wave_model import (
        audit_tensor_refractive_gravity_ppn,
        audit_moduli_space_bell_entanglement,
    )
    ppn = audit_tensor_refractive_gravity_ppn()
    assert abs(ppn["mean_scalar_deflection_ratio"] - 0.5) < 1e-4
    assert abs(ppn["mean_tensor_deflection_ratio"] - 1.0) < 1e-4
    assert ppn["perihelion_precession_relative_error"] < 0.025

    bell = audit_moduli_space_bell_entanglement()
    assert abs(bell["S_moduli_max_tsirelson"] - 2.0 * np.sqrt(2.0)) < 1e-12
    assert bell["S_separable_lhv_max"] <= 2.0 + 1e-12
    assert bell["violates_bell_inequality"] is True


def test_manuscript_single_source_of_truth_sync():
    """Verify that TOPOLOGICAL_WAVE_SOLITON_PAPER.md, arxiv_submission/main.tex, and EPJC_submission/ are in exact sync with master .tex."""
    import os
    from src.build_arxiv_package import (
        TEX_PATH,
        MD_PATH,
        ARXIV_DIR,
        EPJC_DIR,
        EPJC_SOURCE_DIR,
        convert_latex_to_markdown,
        strip_mermaid_comments,
        run_arxiv_compliance_audit,
    )
    with open(TEX_PATH, "r", encoding="utf-8") as f:
        master_tex = f.read()
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()
    with open(os.path.join(ARXIV_DIR, "main.tex"), "r", encoding="utf-8") as f:
        arxiv_tex = f.read()
    with open(os.path.join(EPJC_SOURCE_DIR, "main.tex"), "r", encoding="utf-8") as f:
        epjc_tex = f.read()

    assert convert_latex_to_markdown(master_tex) == md_text
    assert strip_mermaid_comments(master_tex) == arxiv_tex
    assert epjc_tex == arxiv_tex
    run_arxiv_compliance_audit(master_tex, arxiv_tex, md_text)
    companion_pdf = os.path.join(os.path.dirname(TEX_PATH), "NON_SPECIALIST_COMPANION_GUIDE.pdf")
    anc_companion_pdf = os.path.join(ARXIV_DIR, "anc", "NON_SPECIALIST_COMPANION_GUIDE.pdf")
    assert os.path.exists(companion_pdf) and os.path.getsize(companion_pdf) > 0
    assert os.path.exists(anc_companion_pdf) and os.path.getsize(anc_companion_pdf) > 0

    # Verify EPJC submission package files
    for epjc_req in (
        os.path.join(EPJC_DIR, "TOPOLOGICAL_WAVE_SOLITON_PAPER.pdf"),
        os.path.join(EPJC_DIR, "cover_letter.pdf"),
        os.path.join(EPJC_DIR, "cover_letter.md"),
        os.path.join(EPJC_DIR, "declarations.md"),
        os.path.join(EPJC_DIR, "suggested_reviewers.md"),
        os.path.join(EPJC_DIR, "REFERENCE_VERIFICATION.md"),
        os.path.join(EPJC_DIR, "REFERENCE_VERIFICATION.csv"),
        os.path.join(EPJC_DIR, "FRONT_MATTER_CHECK.md"),
        os.path.join(EPJC_DIR, "FIGURE_CHECK.md"),
        os.path.join(EPJC_DIR, "EPJC_portal_metadata.txt"),
        os.path.join(EPJC_DIR, "SUBMISSION_CHECKLIST.md"),
        os.path.join(EPJC_DIR, "EPJC_submission_package.zip"),
    ):
        assert os.path.exists(epjc_req) and os.path.getsize(epjc_req) > 0

    assert r"\bibitem{ref54}" in master_tex
    assert "Google Gemini AI (Gemini 4 Argon High)" in master_tex
    for epjc_text_file in ("declarations.md", "EPJC_portal_metadata.txt", "cover_letter.md"):
        with open(os.path.join(EPJC_DIR, epjc_text_file), "r", encoding="utf-8") as f:
            txt_content = f.read()
        assert "Google Gemini AI (Gemini 4 Argon High)" in txt_content
        assert "GPT-5.4" not in txt_content
        assert "Claude Opus 4.6" not in txt_content

    root_dir = os.path.dirname(TEX_PATH)
    fig7_paper = os.path.join(root_dir, "results", "figures", "paper", "fig7_spatial_soliton_and_hopfion_topology.png")
    fig7_arxiv = os.path.join(ARXIV_DIR, "fig7_spatial_soliton_and_hopfion_topology.png")
    fig7_epjc = os.path.join(EPJC_SOURCE_DIR, "fig7_spatial_soliton_and_hopfion_topology.png")
    assert os.path.exists(fig7_paper) and os.path.getsize(fig7_paper) > 0
    assert os.path.exists(fig7_arxiv) and os.path.getsize(fig7_arxiv) > 0
    assert os.path.exists(fig7_epjc) and os.path.getsize(fig7_epjc) > 0
    assert "fig7_spatial_soliton_and_hopfion_topology.png" in master_tex
    assert "fig7_spatial_soliton_and_hopfion_topology.png" in md_text

    companion_md_path = os.path.join(root_dir, "NON_SPECIALIST_COMPANION_GUIDE.md")
    with open(companion_md_path, "r", encoding="utf-8") as f:
        companion_md = f.read()
    for cfig in (
        "companion_fig1_wave_knots_and_quarks.png",
        "companion_fig2_spin_motion_and_nuclei.png",
        "companion_fig3_gravity_and_cosmos.png",
    ):
        cfig_path = os.path.join(root_dir, "results", "figures", "companion", cfig)
        assert os.path.exists(cfig_path) and os.path.getsize(cfig_path) > 0
        assert cfig in companion_md

