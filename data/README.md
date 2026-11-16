# SH-FAE Data Directory

This folder contains datasets and observation records used for training and testing SH-FAE.

## 1. Synthetic Training Library
- Generated via `python scripts/01_generate_data.py --n-samples 50000 --output data/swcc_hcf_50k.npz`.
- Contains 50,000 paired SWRC ($S_e(x)$) and HCF ($G_K(u)$) curves across 8 diverse constitutive families (VGM, Brooks-Corey, Kosugi, Fredlund-Xing, PDI-like VG, Durner bimodal, Nonparametric monotonic density, and Boundary-stress).
- NPZ Structure:
  - `curves/se_x`: Effective saturation across suction $x \in [-1, 8]$
  - `curves/gkr_u`: Normalized relative permeability across saturation $u \in [0, 1]$
  - `curves/ykr_u`: Log relative permeability $y_K(u) = \log_{10} k_r(u)$
  - `scales/theta_r`, `scales/theta_s`, `scales/log10_Ks_cm_day`: Dimensional scale parameters
  - `metadata/family_id`, `metadata/split`: Soil family identifier and train (0) / val (1) / test (2) split flags.

## 2. Laboratory Inversion Observations (`laboratory/`)
- `observations.npz`: 300-minute upward infiltration volumetric water content time series from Sadeghi et al. (2017) / Bandai et al. (2024) across 8 optical sensor depths in an AZ7 sandy loam column.
- `metadata.json`: Experimental metadata and column geometry.
- `figure10a_digitized_wrc.csv`: Independent drying-branch water retention measurements (Tempe cell & WP4) for post-inversion reference.
