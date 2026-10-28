"""In-memory supervised training for the d=3 local MicroCNN predecoder."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymatching
import stim
import torch
from torch import nn

from src.models.cnn_predecoder import MicroCNN, count_trainable_parameters
from src.models.lattice import DetectorLattice
from src.quantum.circuits import build_rotated_surface_code_circuit, sample_syndromes
from src.quantum.dem import DEMMechanisms


@dataclass
class DEMTrainingData:
    """All examples are held as NumPy arrays in RAM for the training run."""

    inputs: np.ndarray
    detectors: np.ndarray
    observables: np.ndarray
    local_targets: np.ndarray
    mechanisms: DEMMechanisms
    local_mechanism_indices: np.ndarray
    dem: stim.DetectorErrorModel
    lattice: DetectorLattice


class FocalBCELoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        if not 0 < alpha < 1 or gamma < 0:
            raise ValueError("Focal BCE requer 0 < alpha < 1 e gamma >= 0.")
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t).pow(self.gamma) * bce).mean()


def generate_dem_training_data(
    shots: int = 10_000,
    seed: int = 314159,
    distance: int = 3,
    noise_rate: float = 0.005,
) -> DEMTrainingData:
    """Generate DEM syndrome/error labels and tensor inputs fully in memory."""
    if distance != 3:
        raise ValueError("O pipeline inicial está deliberadamente limitado a d=3.")
    if shots < 2:
        raise ValueError("Use pelo menos dois shots para separar treino e validação.")

    circuit = build_rotated_surface_code_circuit(
        distance=distance, rounds=distance, noise_rate=noise_rate
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    lattice = DetectorLattice.from_circuit(circuit, distance=distance, rounds=distance)
    mechanisms = DEMMechanisms.from_dem(dem)
    local_indices = np.flatnonzero(mechanisms.local_mask(lattice.coordinates, max_span=1))
    if len(local_indices) == 0:
        raise ValueError("Nenhum mecanismo local disponível para supervisão.")

    detectors, observables, sampled_errors = dem.compile_sampler(seed=seed).sample(
        shots=shots, return_errors=True
    )
    if sampled_errors is None:
        raise RuntimeError("O sampler Stim não retornou ativações de mecanismos.")
    inputs = lattice.tensorize(detectors)
    targets = sampled_errors[:, local_indices].copy()
    return DEMTrainingData(
        inputs=inputs,
        detectors=detectors,
        observables=observables,
        local_targets=targets,
        mechanisms=mechanisms,
        local_mechanism_indices=local_indices,
        dem=dem,
        lattice=lattice,
    )


def _decode_metrics(
    model: MicroCNN,
    inputs: torch.Tensor,
    detectors: np.ndarray,
    observables: np.ndarray,
    data: DEMTrainingData,
    matcher: pymatching.Matching,
    device: torch.device,
    threshold: float,
    target_labels: np.ndarray | None = None,
) -> dict[str, float | None]:
    model.eval()
    selected = []
    with torch.no_grad():
        for start in range(0, len(inputs), 256):
            logits = model(inputs[start : start + 256].to(device))
            selected.append((torch.sigmoid(logits) >= threshold).cpu().numpy())
    predicted_local = np.concatenate(selected, axis=0)
    local_detector_masks = data.mechanisms.detector_masks[
        data.local_mechanism_indices
    ].astype(np.int32)
    local_observable_masks = data.mechanisms.observable_masks[
        data.local_mechanism_indices
    ].astype(np.int32)
    predicted_detector_correction = (
        predicted_local.astype(np.int32) @ local_detector_masks
    ) % 2
    predicted_observable_correction = (
        predicted_local.astype(np.int32) @ local_observable_masks
    ) % 2
    residual = np.logical_xor(detectors, predicted_detector_correction.astype(np.bool_))
    mwpm_predictions = matcher.decode_batch(residual)
    final_predictions = np.logical_xor(
        mwpm_predictions, predicted_observable_correction.astype(np.bool_)
    )
    baseline_predictions = matcher.decode_batch(detectors)
    baseline_errors = np.any(baseline_predictions != observables, axis=1)
    hybrid_errors = np.any(final_predictions != observables, axis=1)
    has_target_labels = target_labels is not None
    if target_labels is None:
        target_labels = data.local_targets[: len(inputs)]
    true_positive = int(np.count_nonzero(predicted_local & target_labels))
    predicted_positive = int(np.count_nonzero(predicted_local))
    actual_positive = int(np.count_nonzero(target_labels))
    return {
        "predicted_mechanisms_per_shot": float(predicted_local.sum(axis=1).mean()),
        "predicted_mechanism_rate": float(predicted_local.mean()),
        "sparsification_rate": float(1.0 - predicted_local.mean()),
        "target_mechanism_rate": (
            float(target_labels.mean()) if has_target_labels else None
        ),
        "mechanism_precision": (
            true_positive / max(1, predicted_positive) if has_target_labels else None
        ),
        "mechanism_recall": (
            true_positive / max(1, actual_positive) if has_target_labels else None
        ),
        "baseline_p_L": float(baseline_errors.mean()),
        "hybrid_p_L": float(hybrid_errors.mean()),
    }


def train_microcnn(
    shots: int = 10_000,
    epochs: int = 20,
    batch_size: int = 256,
    seed: int = 314159,
    threshold: float = 0.5,
    output_dir: str | Path = "data/microcnn_d3",
) -> dict[str, object]:
    """Train on d=3, p=0.005 DEM samples and save checkpoint/history."""
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs e batch_size devem ser maiores que zero.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold deve estar no intervalo [0, 1].")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = generate_dem_training_data(shots=shots, seed=seed)
    model = MicroCNN.from_lattice_and_dem(data.lattice, data.mechanisms)
    local_indices = model.local_mechanism_indices.cpu().numpy()
    if not np.array_equal(local_indices, data.local_mechanism_indices):
        raise RuntimeError("A ordem dos mecanismos locais divergiu entre dados e modelo.")

    generator = np.random.default_rng(seed + 1)
    permutation = generator.permutation(shots)
    validation_size = max(1, int(round(shots * 0.2)))
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]
    inputs = torch.from_numpy(data.inputs)
    targets = torch.from_numpy(data.local_targets.astype(np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = FocalBCELoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    matcher = pymatching.Matching.from_detector_error_model(data.dem)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_order = generator.permutation(training_indices)
        train_losses = []
        for start in range(0, len(train_order), batch_size):
            indices = train_order[start : start + batch_size]
            batch_inputs = inputs[indices].to(device)
            batch_targets = targets[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_inputs), batch_targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for start in range(0, len(validation_indices), batch_size):
                indices = validation_indices[start : start + batch_size]
                logits = model(inputs[indices].to(device))
                validation_losses.append(
                    float(criterion(logits, targets[indices].to(device)).cpu())
                )
        val_data_indices = validation_indices
        epoch_metrics = _decode_metrics(
            model=model,
            inputs=inputs[val_data_indices],
            detectors=data.detectors[val_data_indices],
            observables=data.observables[val_data_indices],
            data=data,
            matcher=matcher,
            device=device,
            threshold=threshold,
            target_labels=data.local_targets[val_data_indices],
        )
        history.append(
            {
                "epoch": epoch,
                "train_focal_bce": float(np.mean(train_losses)),
                "validation_focal_bce": float(np.mean(validation_losses)),
                **epoch_metrics,
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={history[-1]['train_focal_bce']:.6f} "
            f"val_loss={history[-1]['validation_focal_bce']:.6f} "
            f"sparse={epoch_metrics['sparsification_rate']:.4f} "
            f"hybrid_pL={epoch_metrics['hybrid_p_L']:.4f}"
        )

    # Final holdout comes from the original circuit sampler, not the DEM sampler.
    circuit = build_rotated_surface_code_circuit(distance=3, rounds=3, noise_rate=0.005)
    circuit_batch = sample_syndromes(
        circuit, num_shots=validation_size, seed=seed + 2
    )
    circuit_inputs = torch.from_numpy(data.lattice.tensorize(circuit_batch.syndromes))
    circuit_validation = _decode_metrics(
        model=model,
        inputs=circuit_inputs,
        detectors=circuit_batch.syndromes,
        observables=circuit_batch.observables,
        data=data,
        matcher=matcher,
        device=device,
        threshold=threshold,
        target_labels=None,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "distance": 3,
            "rounds": 3,
            "noise_rate": 0.005,
            "seed": seed,
            "threshold": threshold,
            "num_trainable_parameters": count_trainable_parameters(model),
            "local_mechanism_indices": local_indices,
        },
        output / "microcnn_d3.pt",
    )
    summary = {
        "distance": 3,
        "noise_rate": 0.005,
        "shots": shots,
        "training_shots": len(training_indices),
        "validation_shots": len(validation_indices),
        "seed": seed,
        "device": str(device),
        "num_local_mechanisms": len(local_indices),
        "num_trainable_parameters": count_trainable_parameters(model),
        "final_epoch": history[-1],
        "circuit_validation": circuit_validation,
        "history_csv": str(output / "history.csv"),
        "checkpoint": str(output / "microcnn_d3.pt"),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(
        "circuit_validation: "
        f"baseline_pL={circuit_validation['baseline_p_L']:.4f} "
        f"hybrid_pL={circuit_validation['hybrid_p_L']:.4f} "
        f"sparse={circuit_validation['sparsification_rate']:.4f}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="data/microcnn_d3")
    args = parser.parse_args()
    train_microcnn(
        shots=args.shots,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
