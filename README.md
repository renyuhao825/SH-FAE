# SH-FAE: Soil Hydraulic Functional AutoEncoder

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **SH-FAE** (**S**oil **H**ydraulic **F**unctional **A**uto**E**ncoder), a physics-constrained functional autoencoder for soil-water retention curves (SWRC) and hydraulic conductivity functions (HCF), and its application to differentiable Richards equation inversion.

---

## 📌 Highlights

- **Infinite-Dimensional Functional Parameterization**: Unlike conventional autoencoders that map fixed-grid vectors, SH-FAE maps partial point observations to continuous constitutive functions that can be queried at arbitrary water potential or saturation coordinates.
- **Strict Physical Hard Constraints**: The functional decoder predicts non-negative densities and integrates them via cumulative trapezoid integration, strictly guaranteeing:
  - **Monotonicity**: $\frac{d S_e}{dx} \le 0$ and $\frac{d G_K}{du} \ge 0$ everywhere.
  - **Boundedness**: $S_e \in [0, 1]$ and $G_K \in [0, 1]$.
  - **Exact Boundary Endpoints**: $S_e(x_{\min}) = 1$, $S_e(x_{\max}) = 0$, $G_K(0) = 0$, $G_K(1) = 1$.
- **Synthetic Soil Hydraulic Library**: Includes a data generator covering 50,000 paired SWRC-HCF samples across 8 distinct constitutive families (van Genuchten-Mualem, Brooks-Corey, Kosugi, Fredlund-Xing, PDI-like dry-end extension, Durner bimodal, Nonparametric monotonic densities, and Boundary-stress samples).
- **Differentiable 1D Richards PDE Solver**: Cell-centered finite volume formulation with backward Euler time integration, Newton-Raphson iteration, and Armijo backtracking line search, allowing end-to-end backpropagation through the transient hydraulic flow process.
- **Fast & Well-Conditioned Latent Inversion**: Inversion only optimizes a compact 64-dimensional latent token $z$ and 3 physical scale parameters ($\theta_r, \theta_s, K_s$), dramatically shrinking the search space and speeding up convergence compared to high-dimensional neural network weights.

---

## 📁 Repository Structure

```text
SH-FAE/
├── configs/                          # Config files (YAML)
│   ├── data_generation.yaml          # Library generation configuration
│   ├── train_ae.yaml                 # SH-FAE autoencoder training configuration
│   ├── invert_curve_lu.yaml          # Direct expressivity inversion for Lu (2016) SWRC
│   ├── invert_curve_gou.yaml         # Direct expressivity inversion for Gou (2023) HCF
│   ├── invert_synthetic.yaml         # Synthetic wetting-drying PDE inversion (Case 2)
│   └── invert_laboratory.yaml        # Laboratory upward infiltration PDE inversion (Case 3)
├── data/                             # Data directory
│   ├── README.md                     # Data formats and schema documentation
│   └── laboratory/                   # Laboratory benchmark data (AZ7 sandy loam)
│       ├── observations.npz          # Transient water-content time series
│       ├── metadata.json             # Sensor locations & experimental configuration
│       └── figure10a_digitized_wrc.csv # Independent drying-branch SWRC validation
├── sh_fae/                           # Core Python package
│   ├── data/                         # Data generation and sampling
│   │   ├── families.py               # 8 soil hydraulic constitutive models
│   │   ├── generator.py              # Synthetic library generator & quality filters
│   │   └── dataset.py                # Dataset and dynamic context/query point sampler
│   ├── models/                       # Deep learning modules
│   │   ├── embed.py                  # PointEmbed (coordinate, channel, value embedding)
│   │   ├── encoder.py                # Transformer set encoder with self-attention
│   │   ├── decoder.py                # Constrained density-integration decoder
│   │   └── fae.py                    # End-to-end SH-FAE model class
│   ├── physics/                      # Differentiable forward physics
│   │   ├── grid.py                   # 1D finite-volume cell-centered grid with ghost cells
│   │   ├── material.py               # Hydraulic material conversions
│   │   ├── constitutive.py           # Callable constitutive bridge from latent z to solver
│   │   └── solver.py                 # Fully differentiable Richards solver (Newton + linesearch)
│   ├── inversion/                    # Inverse problem engines
│   │   ├── transforms.py             # Bounded tanh parameter transformations
│   │   ├── direct.py                 # Direct curve expressivity inversion engine
│   │   └── dynamic.py                # PDE-constrained dynamic latent inversion engine
│   └── utils/                        # Utilities
│       ├── config.py                 # Dataclass config manager
│       └── metrics.py                # Evaluation metrics (RMSE, n-RMSE, R^2, monotonicity)
├── scripts/                          # Executable command-line workflows
│   ├── 01_generate_data.py           # CLI for synthetic constitutive library generation
│   ├── 02_train.py                   # CLI for training SH-FAE
│   ├── 03_eval.py                    # CLI for evaluating trained checkpoints & curve plotting
│   ├── 04_invert_curve.py            # CLI for expressivity inversion on unseen models
│   └── 05_invert_pde.py              # CLI for dynamic Richards PDE inversion (Synthetic & Lab)
├── demo.py                           # 1-minute end-to-end sanity verification test
├── pyproject.toml                    # Package metadata & build definition
├── requirements.txt                  # Python dependencies
└── LICENSE                           # MIT License
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/SH-FAE.git
cd SH-FAE
```

### 2. Create and activate a virtual environment
```bash
conda create -n sh-fae python=3.10 -y
conda activate sh-fae
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

---

## ⚡ 1-Minute Quickstart (Demo)

Run the included standalone demo to test the entire pipeline (mini data generation $\rightarrow$ mini autoencoder training $\rightarrow$ direct curve inversion $\rightarrow$ differentiable Richards solver with backpropagation) in under a minute:

```bash
python demo.py
```

Expected output:
```text
=================================================================
      SH-FAE: Quick End-to-End Verification Demo
=================================================================

[1/4] Generating mini synthetic library (50 samples)...
      Created mini library at: mini_library.npz

[2/4] Testing SH-FAE model forward pass and training loop...
      15 training steps passed! Final step loss: 6.8055e-02

[3/4] Testing direct curve latent inversion...
      Curve inversion completed! Data misfit loss: 6.1977e-02

[4/4] Testing differentiable Richards equation solver...
      Richards solver finished! Converged: True
      Gradient w.r.t. latent code z: norm = 9.3565e-02

=================================================================
  All tests passed successfully in 2.59 seconds!
  Repository is ready for production and GitHub publication.
=================================================================
```

---

## 🚀 Step-by-Step Reproduction Guide

### Step 1: Generate Synthetic Soil Hydraulic Library
Generate 50,000 physically admissible SWRC-HCF pairs covering 8 constitutive families:
```bash
python scripts/01_generate_data.py --n-samples 50000 --output data/swcc_hcf_50k.npz
```
*Note: You can also generate a smaller dataset for quick prototyping by passing `--n-samples 1000`.*

### Step 2: Train the SH-FAE Functional Autoencoder
Train the Transformer set encoder and constrained density-integration decoder:
```bash
python scripts/02_train.py \
    --config configs/train_ae.yaml \
    --data-path data/swcc_hcf_50k.npz \
    --output-dir outputs/ae_50k_lat64 \
    --device auto
```
This script saves the model checkpoint (`checkpoint_final.pt`) and computes the empirical latent distribution statistics (`trainset_latent_stats.npz`), which serve as the empirical prior $z_{\mathrm{ref}}$ and initial state $z_0$ for subsequent inversions.

### Step 3: Evaluate Model Reconstruction
Evaluate the trained checkpoint on the validation set and visualize reconstructed SWRC and HCF curves:
```bash
python scripts/03_eval.py \
    --checkpoint outputs/ae_50k_lat64/checkpoint_final.pt \
    --data-path data/swcc_hcf_50k.npz \
    --split val \
    --out-dir outputs/eval
```

### Step 4: Expressive Ability Test (Unseen Constitutive Models)
Test whether the pretrained decoder can express constitutive shapes not explicitly included in the pretraining library:

- **Unseen SWRC (Lu 2016 generalized adsorption-capillarity model)**:
  ```bash
  python scripts/04_invert_curve.py \
      --checkpoint outputs/ae_50k_lat64/checkpoint_final.pt \
      --mode lu_swrc \
      --stats-path outputs/ae_50k_lat64/trainset_latent_stats.npz \
      --out-dir outputs/invert_direct
  ```
- **Unseen HCF (Gou et al. 2023 adsorbed film + capillary flow model)**:
  ```bash
  python scripts/04_invert_curve.py \
      --checkpoint outputs/ae_50k_lat64/checkpoint_final.pt \
      --mode gou_hcf \
      --stats-path outputs/ae_50k_lat64/trainset_latent_stats.npz \
      --out-dir outputs/invert_direct
  ```

### Step 5: Differentiable Richards PDE Inversion

#### Case A: Synthetic Wetting-Drying Column (Paper Case 2)
Inverts SWRC, HCF, and physical scales from transient water-content observations across 3 wetting-drying cycles in a 1.5 m soil column:
```bash
python scripts/05_invert_pde.py \
    --checkpoint outputs/ae_50k_lat64/checkpoint_final.pt \
    --case synthetic \
    --steps 1500 \
    --noise-std 0.01 \
    --stats-path outputs/ae_50k_lat64/trainset_latent_stats.npz \
    --out-dir outputs/invert_pde
```

#### Case B: Laboratory Upward Infiltration Experiment (Paper Case 3)
Inverts the constitutive curves from real laboratory optical sensor data on an AZ7 sandy loam column (Sadeghi et al. 2017 / Bandai et al. 2024):
```bash
python scripts/05_invert_pde.py \
    --checkpoint outputs/ae_50k_lat64/checkpoint_final.pt \
    --case laboratory \
    --data-dir data/laboratory \
    --steps 1500 \
    --stats-path outputs/ae_50k_lat64/trainset_latent_stats.npz \
    --out-dir outputs/invert_pde
```

---

## 🔬 Mathematical Formulations

### 1. Dimensionless Function Representation
- **SWRC**: Represented by effective saturation $S_e \in [0, 1]$ over logarithmic suction coordinate:
  $$x = \log_{10}\left(\frac{-\psi}{1\,\mathrm{cm}}\right)$$
- **HCF**: Represented as a normalized relative permeability $G_K \in [0, 1]$ over saturation $u = S_e \in [0, 1]$:
  $$G_K(u) = \frac{\log_{10} k_r(u) - y_{K,\min}}{0 - y_{K,\min}}, \quad y_{K,\min} = -12$$

### 2. Functional Decoder with Hard Constraints
Given latent vector $z$, the decoder predicts positive densities via Softplus:
$$q_z(x) = \mathrm{Softplus}(D_S(z, x)) + \epsilon, \quad r_z(u) = \mathrm{Softplus}(D_K(z, u)) + \epsilon$$
and integrates them:
$$\hat{S}_e(x; z) = 1 - \frac{\int_{x_{\min}}^x q_z(\xi) \, d\xi}{\int_{x_{\min}}^{x_{\max}} q_z(\xi) \, d\xi}, \quad \hat{G}_K(u; z) = \frac{\int_0^u r_z(v) \, dv}{\int_0^1 r_z(v) \, dv}$$

### 3. Richards Equation & Inversion Objective
$$\frac{\partial \theta(\psi)}{\partial t} = \frac{\partial}{\partial z}\left[ \frac{\rho g}{\mu} K k_r(\psi) \left( \frac{\partial \psi}{\partial z} + 1 \right) \right]$$
Inversion searches for optimal latent code $z$ and unconstrained physical scales $\eta$:
$$\min_{z, \eta} \mathcal{L} = \frac{1}{N_{\mathrm{obs}}} \sum_{j=1}^{N_{\mathrm{obs}}} \left( \frac{\hat{\theta}_j(z, \eta) - \theta_{\mathrm{obs}, j}}{\sigma_j} \right)^2 + \lambda_z \|z - z_{\mathrm{ref}}\|_2^2 + \lambda_\eta \|\eta - \eta_{\mathrm{ref}}\|_2^2$$

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
