"""Reproducible baseline sweep for rotated surface-code memory circuits."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

from src.decoders.baseline_mwpm import PyMatchingDecoder
from src.quantum.circuits import build_rotated_surface_code_circuit, sample_syndromes


def wilson_interval(errors: int, shots: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a binomial logical-error rate."""
    rate = errors / shots
    denominator = 1 + z * z / shots
    center = (rate + z * z / (2 * shots)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / shots + z * z / (4 * shots**2)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def run_baseline_curve(
    output_path: str | Path,
    distances: tuple[int, ...] = (3, 5, 7),
    noise_rates: tuple[float, ...] = (0.001, 0.003, 0.005, 0.01, 0.015, 0.02, 0.03),
    shots: int = 10_000,
    seed: int = 20261028,
    basis: str = "Z",
) -> list[dict[str, int | float | str]]:
    """Sample and decode each sweep point with deterministic, distinct seeds."""
    if shots <= 0:
        raise ValueError("shots deve ser maior que zero.")
    rows = []
    for distance in distances:
        for rate_index, noise_rate in enumerate(noise_rates):
            point_seed = seed + distance * 100_000 + rate_index
            circuit = build_rotated_surface_code_circuit(
                distance=distance, rounds=distance, noise_rate=noise_rate, basis=basis
            )
            batch = sample_syndromes(circuit, num_shots=shots, seed=point_seed)
            decoder = PyMatchingDecoder(circuit)
            # Warm up once, then report the median of repeated full-batch runs.
            decoder.decode_batch(batch.syndromes[: min(256, shots)], batch.observables[: min(256, shots)])
            repeats = [
                decoder.decode_batch(batch.syndromes, batch.observables)
                for _ in range(3)
            ]
            result = repeats[-1]
            decode_seconds = statistics.median(item.elapsed_time_seconds for item in repeats)
            ci_low, ci_high = wilson_interval(result.logical_errors, shots)
            rows.append(
                {
                    "distance": distance,
                    "rounds": distance,
                    "basis": basis,
                    "physical_error_rate": noise_rate,
                    "shots": shots,
                    "seed": point_seed,
                    "logical_errors": result.logical_errors,
                    "p_L": result.logical_error_rate,
                    "p_L_ci95_low": ci_low,
                    "p_L_ci95_high": ci_high,
                    "decode_seconds": decode_seconds,
                    "decode_us_per_shot": decode_seconds / shots * 1e6,
                }
            )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/baseline_curve.csv")
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    rows = run_baseline_curve(args.output, shots=args.shots, seed=args.seed)
    print(f"Gravados {len(rows)} pontos em {args.output}")


if __name__ == "__main__":
    main()
