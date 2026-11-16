"""Script 01: Generate synthetic soil hydraulic constitutive library."""

from __future__ import annotations

import argparse
from pathlib import Path

from sh_fae.data.generator import generate_library, save_library_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic SWRC-HCF training library.")
    parser.add_argument("--n-samples", type=int, default=50000, help="Number of constitutive pairs (default: 50000)")
    parser.add_argument("--nx", type=int, default=384, help="Grid points for SWRC suction x (default: 384)")
    parser.add_argument("--nu", type=int, default=128, help="Grid points for HCF saturation u (default: 128)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--y-min", type=float, default=-12.0, help="Minimum log10 kr floor (default: -12.0)")
    parser.add_argument("--output", type=str, default="data/swcc_hcf_50k.npz", help="Output .npz file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Generating {args.n_samples} soil hydraulic constitutive pairs...")
    arrays = generate_library(
        n_samples=args.n_samples,
        nx=args.nx,
        nu=args.nu,
        seed=args.seed,
        y_min=args.y_min,
        progress=True,
    )
    out_path = Path(args.output)
    save_library_npz(arrays, out_path)
    print(f"Successfully saved synthetic library to {out_path.resolve()}!")
    print(f"Dataset arrays: {list(arrays.keys())}")


if __name__ == "__main__":
    main()
