"""
Master Runner for the Wave Entities Hypothesis Evaluation Suite.

Executes all four quantitative research experiments, aggregates diagnostics
into `results/all_experiments_summary.json`, and generates all figures in
`results/figures/`.
"""

from __future__ import annotations
import json
import time
from pathlib import Path

from src.exp1_l0_dynamics import run_experiment_1
from src.exp2_source_models_audit import run_experiment_2
from src.exp3_emergent_space_geometry import run_experiment_3
from src.exp4_cosmology_cmb import run_experiment_4
from src.exp5_approximate_wave_entity_model import run_experiment_5
from src.exp6_evolved_topological_wave_model import run_experiment_6


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print("[1/6] Running Experiment 1: Level 0 State Dynamics, Emergence & Persistence Audit...")
    res1 = run_experiment_1(results_dir)

    print("[2/6] Running Experiment 2: Source PDF Mathematical Models Audit...")
    res2 = run_experiment_2(results_dir)

    print("[3/6] Running Experiment 3: Emergent Space Geometry & Kinematic Degeneracy Audit...")
    res3 = run_experiment_3(results_dir)

    print("[4/6] Running Experiment 4: Cosmological Spectral Spreading & CMB Blackbody Audit...")
    res4 = run_experiment_4(results_dir)

    print("[5/6] Running Experiment 5: Approximate Wave Entity Model (AWEM) & Proton Composite Synthesis...")
    res5 = run_experiment_5(results_dir)

    print("[6/6] Running Experiment 6: Evolved Topological–Spinor–Metric Wave Entity Model (AWEM v2)...")
    res6 = run_experiment_6(results_dir)

    elapsed = time.perf_counter() - t0
    combined = {
        "suite_version": "1.2.0",
        "execution_time_seconds": round(elapsed, 3),
        "exp1_l0_dynamics": res1,
        "exp2_source_models_audit": res2,
        "exp3_emergent_space_geometry": res3,
        "exp4_cosmology_cmb": res4,
        "exp5_approximate_wave_entity_model": res5,
        "exp6_evolved_topological_wave_model": res6,
    }

    out_file = results_dir / "all_experiments_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"\nAll 6 experiments completed in {elapsed:.2f}s.")
    print(f"Summary written to: {out_file}")


if __name__ == "__main__":
    main()
