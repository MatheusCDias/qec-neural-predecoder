"""Map Stim detector order into a padded spatiotemporal tensor lattice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim


@dataclass(frozen=True)
class DetectorLattice:
    """Coordinate map for a circuit's ``(x, y, t)`` detector annotations."""

    distance: int
    rounds: int
    coordinates: np.ndarray  # (num_detectors, 3), scaled integer (x, y, t)
    voxel_indices: np.ndarray  # (num_detectors, 3), indices (t, y, x)
    occupancy: np.ndarray  # (rounds + 1, distance + 1, distance + 1)
    static_channels: np.ndarray  # (4, rounds + 1, distance + 1, distance + 1)

    @classmethod
    def from_circuit(
        cls, circuit: stim.Circuit, distance: int, rounds: int | None = None
    ) -> "DetectorLattice":
        if rounds is None:
            rounds = distance
        if distance <= 0 or rounds <= 0:
            raise ValueError("distance e rounds devem ser inteiros positivos.")

        raw = circuit.get_detector_coordinates()
        if len(raw) != circuit.num_detectors:
            raise ValueError(
                "Todos os detectores precisam ter coordenadas Stim explícitas."
            )
        coords = np.empty((circuit.num_detectors, 3), dtype=np.int64)
        for detector in range(circuit.num_detectors):
            if detector not in raw or len(raw[detector]) < 3:
                raise ValueError(f"Coordenadas (x, y, t) ausentes para D{detector}.")
            xyz = np.asarray(raw[detector][:3], dtype=np.float64)
            if not np.all(np.isfinite(xyz)):
                raise ValueError(f"Coordenadas não finitas para D{detector}.")
            scaled = np.array([xyz[0] / 2, xyz[1] / 2, xyz[2]])
            rounded = np.rint(scaled)
            if not np.allclose(scaled, rounded, atol=1e-8):
                raise ValueError(
                    f"Coordenadas de D{detector} incompatíveis com a malha: {xyz}."
                )
            coords[detector] = rounded.astype(np.int64)

        # Stim returns coordinates as x,y,t; tensor voxel order is t,y,x.
        voxel_indices = coords[:, [2, 1, 0]].copy()
        limits = np.array([rounds, distance, distance], dtype=np.int64)
        if np.any(voxel_indices < 0) or np.any(voxel_indices > limits):
            raise ValueError("Coordenada de detector fora da malha configurada.")
        if len({tuple(v) for v in voxel_indices.tolist()}) != circuit.num_detectors:
            raise ValueError("Dois detectores ocupam a mesma célula da malha.")

        shape = (rounds + 1, distance + 1, distance + 1)
        occupancy = np.zeros(shape, dtype=np.bool_)
        occupancy[tuple(voxel_indices.T)] = True

        static = np.zeros((4, *shape), dtype=np.float32)
        static[0] = occupancy  # valid-detector mask
        for it, iy, ix in voxel_indices:
            static[1, it, iy, ix] = float(
                ix == 0 or ix == distance or iy == 0 or iy == distance
            )
            static[2, it, iy, ix] = float(it == 0 or it == rounds)
            static[3, it, iy, ix] = 1.0 if (ix + iy) % 2 else -1.0

        return cls(
            distance=distance,
            rounds=rounds,
            coordinates=coords,
            voxel_indices=voxel_indices,
            occupancy=occupancy,
            static_channels=static,
        )

    def tensorize(self, syndromes: np.ndarray) -> np.ndarray:
        """Return float32 ``(B,5,T,Y,X)``: syndrome plus four static channels."""
        syndromes = np.asarray(syndromes)
        if syndromes.ndim != 2 or syndromes.shape[1] != len(self.voxel_indices):
            raise ValueError(
                "syndromes deve ter formato "
                f"(batch, {len(self.voxel_indices)})."
            )
        if syndromes.dtype != np.bool_:
            raise TypeError("syndromes deve ser uma matriz booleana.")
        batch = syndromes.shape[0]
        result = np.zeros(
            (batch, 5, self.rounds + 1, self.distance + 1, self.distance + 1),
            dtype=np.float32,
        )
        result[:, 0, self.voxel_indices[:, 0], self.voxel_indices[:, 1], self.voxel_indices[:, 2]] = syndromes
        result[:, 1:] = self.static_channels[None, ...]
        return result
