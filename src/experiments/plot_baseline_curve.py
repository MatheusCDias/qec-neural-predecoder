"""Plot a saved baseline curve CSV with its 95% confidence intervals."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_curve(csv_path: str | Path, output_path: str | Path) -> None:
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for distance in sorted({int(row["distance"]) for row in rows}):
        points = sorted(
            (row for row in rows if int(row["distance"]) == distance),
            key=lambda row: float(row["physical_error_rate"]),
        )
        x = [float(row["physical_error_rate"]) for row in points]
        y = [float(row["p_L"]) for row in points]
        low = [max(0.0, value - float(row["p_L_ci95_low"])) for value, row in zip(y, points)]
        high = [max(0.0, float(row["p_L_ci95_high"]) - value) for value, row in zip(y, points)]
        ax.errorbar(x, y, yerr=[low, high], marker="o", capsize=3, label=f"d={distance}")
    ax.set(xscale="log", yscale="log", xlabel="Physical error rate p", ylabel="Logical error rate p_L")
    ax.set_title("Rotated surface-code memory — PyMatching baseline")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/baseline_curve.csv")
    parser.add_argument("--output", default="data/baseline_curve.png")
    args = parser.parse_args()
    plot_curve(args.input, args.output)
    print(f"Gráfico gravado em {args.output}")


if __name__ == "__main__":
    main()
