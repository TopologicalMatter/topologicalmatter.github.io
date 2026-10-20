[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23512788.svg)](https://doi.org/10.5281/zenodo.23512788) | Live portal: https://topologicalmatter.github.io/

# Wave Entities Hypothesis — Mathematical & Numerical Evaluation Suite

This repository contains the reproducible mathematical modeling, numerical simulation, and verification software developed to evaluate the **Wave Entities** hypothesis from the primary source documents:
- [`Wave Entities Notes.pdf`](./Wave%20Entities%20Notes.pdf) (20 pages: author's original notes, pp. 1–13; OpenAI-processed paper draft Version A, pp. 13–20)
- [`Wave Entities and the Universe_ A Hypothetical Framework for a Cyclic, Wave-Based Reality.pdf`](./Wave%20Entities%20and%20the%20Universe_%20A%20Hypothetical%20Framework%20for%20a%20Cyclic,%20Wave-Based%20Reality.pdf) (6 pages: AI-processed paper draft Version B with proposed differential equation completions)

---

## 1. Architecture & Separation of Variables

To prevent circular validation, all models strictly separate:
1. **Fundamental Level 0 State Variables**:
   - State vector $E(t) = [E_1(t), E_2(t), \dots, E_N(t)] \in \mathbb{R}^N$ (dimensionless normalized energy amplitude)
   - Conjugate rate vector $V(t) = \dot{E}(t) \in \mathbb{R}^N$ (units: $[T^{-1}]$)
   - Update parameter $t \in \mathbb{R}^+$ (abstract Level 0 clock, units: $[T]$)
   - Component index $i \in \{0, \dots, N-1\}$ (abstract state-space coordinate, explicitly distinguished from emergent physical space)
2. **Derived Level 1 Spectral Quantities**:
   - Normal-mode spectral power $P_k(t) = \frac{1}{2}\left(|\hat{V}_k(t)|^2 + \omega_k^2 |\hat{E}_k(t)|^2\right)$
   - Spectral Participation Ratio $\mathrm{PR}(t) = \frac{(\sum_k P_k)^2}{\sum_k P_k^2} \in [1, N]$
   - Normalized Shannon Spectral Entropy $H_{\mathrm{spec}}(t) = -\frac{1}{\ln N} \sum_k p_k \ln p_k \in [0, 1]$
3. **Derived Emergent-Space & Physical Observables**:
   - Pairwise frequency distances $d_{ij} = |f_i - f_j|$ or spectral overlap metrics $d_{ij} = \sqrt{-\ln \mathcal{O}_{ij}}$
   - Classical Multidimensional Scaling (MDS) Gram eigenvalues $\lambda_k$ and Cayley-Menger 3-simplex volume determinants
   - Kinematic derivatives ($\dot{f}_i, \ddot{f}_i$) and cosmological spectral distributions $u(f, t)$

---

## 2. Experiments Implemented

| Module | Research Question Tested | Key Analytical & Numerical Diagnostics |
| :--- | :--- | :--- |
| [`src/exp1_l0_dynamics.py`](./src/exp1_l0_dynamics.py) | Can localized wave entities emerge spontaneously from random Level 0 noise, multiply via interaction, and persist longer in groups than in isolation? | Compares Class I (linear unitary), Class II (conservative nonlinear FPU/Klein-Gordon), Class III (passive linear damping), and Class IV (imposed state-dependent cooperative damping). Includes 10-seed Monte Carlo null tests, threshold sensitivity, boundary/size ($N$) checks, and $O(\Delta t^2)$ symplectic convergence verification. |
| [`src/exp2_source_models_audit.py`](./src/exp2_source_models_audit.py) | Do the explicit equations proposed in the source PDFs (coupled frequency oscillators in PDF B Sec. 5.2; wave interference & $1/r^3$ gravity in PDF A/B Sec. 6) produce their claimed behaviors? | Proves & verifies normal-mode **level repulsion** ($\Delta f_{\mathrm{eff}} > \Delta f_0$) for diffusive oscillator coupling $K > 0$, tachyonic instability for $K < 0$, zero carrier-frequency shift under linear superposition $W_1 + W_2$, $L^1/L^2$ spectral norm convergence/divergence, and Cotes'-spiral orbital collapse/escape under $F_g \propto 1/r^3$. |
| [`src/exp3_emergent_space_geometry.py`](./src/exp3_emergent_space_geometry.py) | Can frequency differences recover 3D Euclidean geometry and consistent Galilean/relativistic kinematics without circularly assuming space? | Computes MDS eigenvalue spectra and Cayley-Menger determinants (proving $\mathrm{rank}(G)=1$ for scalar frequency distance), Kruskal Stress-1 on $3\times 3\times 3$ 3D lattices, Aristotelian drag vs. Newtonian inertia under "momentum = resistance to frequency change", and the position–frequency–identity trilemma. |
| [`src/exp4_cosmology_cmb.py`](./src/exp4_cosmology_cmb.py) | Is spectral broadening toward lower frequencies physically equivalent to cosmological expansion and cooling? | Compares FLRW multiplicative scaling $f \to f/(1+z)$ (which **narrows** $\sigma_f \propto T$) against advective-diffusive spectral broadening, quantifying COBE/FIRAS Planck blackbody distortions ($|\Delta I|/I_{\mathrm{peak}} < 5\times 10^{-5}$) and quasar atomic doublet ratio violations. |

---

## 3. Reproducibility & Execution Instructions

### Dependencies
- Python $\ge 3.11$ (tested on Python `3.13.8`)
- `numpy`, `scipy`, `matplotlib`, `pytest` (see [`requirements.txt`](./requirements.txt))

### Running the Verification Test Suite and Experiments
```powershell
# Run automated mathematical and numerical unit tests
.\.venv\Scripts\python.exe -m pytest -v

# Execute all four experiments, export JSON metrics, and generate diagnostic figures
.\.venv\Scripts\python.exe run_all_experiments.py
```

### Outputs Generated
- [`results/all_experiments_summary.json`](./results/all_experiments_summary.json) (and individual `exp1_results.json` – `exp4_results.json`)
- [`results/figures/exp1_l0_dynamics.png`](./results/figures/exp1_l0_dynamics.png)
- [`results/figures/exp2_source_models_audit.png`](./results/figures/exp2_source_models_audit.png)
- [`results/figures/exp3_emergent_space_geometry.png`](./results/figures/exp3_emergent_space_geometry.png)
- [`results/figures/exp4_cosmology_cmb.png`](./results/figures/exp4_cosmology_cmb.png)
