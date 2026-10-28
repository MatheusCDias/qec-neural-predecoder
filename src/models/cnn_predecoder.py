"""Small local 3D CNN that scores local Stim DEM error mechanisms."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.models.lattice import DetectorLattice
from src.quantum.dem import DEMMechanisms


class MicroCNN(nn.Module):
    """3D local encoder plus a shared mechanism-scoring head.

    Input is a float tensor from :class:`DetectorLattice` with shape
    ``(batch, 5, rounds+1, distance+1, distance+1)``. The occupancy mask is
    channel 1. Output logits have one column for each local DEM mechanism.
    """

    def __init__(
        self,
        detector_flat_indices: np.ndarray,
        mechanism_detector_indices: np.ndarray,
        mechanism_detector_mask: np.ndarray,
        mechanism_features: np.ndarray,
        local_mechanism_indices: np.ndarray,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(5, 12, kernel_size=3, padding=1),
            nn.GroupNorm(3, 12),
            nn.ReLU(),
            nn.Conv3d(12, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv3d(16, 12, kernel_size=3, padding=1),
            nn.GroupNorm(3, 12),
            nn.ReLU(),
        )
        self.mechanism_head = nn.Sequential(
            nn.Linear(29, 32), nn.ReLU(), nn.Linear(32, 1)
        )

        self.register_buffer(
            "detector_flat_indices",
            torch.as_tensor(detector_flat_indices, dtype=torch.long),
        )
        self.register_buffer(
            "mechanism_detector_indices",
            torch.as_tensor(mechanism_detector_indices, dtype=torch.long),
        )
        self.register_buffer(
            "mechanism_detector_mask",
            torch.as_tensor(mechanism_detector_mask, dtype=torch.bool),
        )
        self.register_buffer(
            "mechanism_features",
            torch.as_tensor(mechanism_features, dtype=torch.float32),
        )
        self.register_buffer(
            "local_mechanism_indices",
            torch.as_tensor(local_mechanism_indices, dtype=torch.long),
        )

        if self.mechanism_features.ndim != 2 or self.mechanism_features.shape[1] != 5:
            raise ValueError("mechanism_features deve ter shape (mecanismos_locais, 5).")
        if self.mechanism_detector_indices.shape != self.mechanism_detector_mask.shape:
            raise ValueError("Índices e máscara dos detectores por mecanismo divergem.")
        if self.mechanism_detector_indices.shape[0] != self.mechanism_features.shape[0]:
            raise ValueError("Os metadados devem corresponder aos mecanismos locais.")
        if torch.any(self.mechanism_detector_mask.sum(dim=1) == 0):
            raise ValueError("Todo mecanismo local precisa atingir ao menos um detector.")

    @classmethod
    def from_lattice_and_dem(
        cls,
        lattice: DetectorLattice,
        mechanisms: DEMMechanisms,
        max_span: int = 1,
    ) -> "MicroCNN":
        """Construct static mechanism gathers and metadata for one code/DEM."""
        if mechanisms.num_detectors != len(lattice.voxel_indices):
            raise ValueError("A lattice e o DEM descrevem números distintos de detectores.")
        local_indices = np.flatnonzero(
            mechanisms.local_mask(lattice.coordinates, max_span=max_span)
        )
        if len(local_indices) == 0:
            raise ValueError("O DEM não contém mecanismos locais candidatos.")

        detector_voxels = lattice.voxel_indices
        tensor_shape = (
            lattice.rounds + 1,
            lattice.distance + 1,
            lattice.distance + 1,
        )
        detector_flat_indices = np.ravel_multi_index(
            detector_voxels.T, tensor_shape
        ).astype(np.int64)

        local_masks = mechanisms.detector_masks[local_indices]
        supports = [np.flatnonzero(mask) for mask in local_masks]
        max_support = max(map(len, supports))
        support_indices = np.zeros((len(supports), max_support), dtype=np.int64)
        support_mask = np.zeros((len(supports), max_support), dtype=np.bool_)
        mechanism_features = np.zeros((len(supports), 5), dtype=np.float32)
        coordinates = lattice.coordinates
        local_probabilities = mechanisms.probabilities[local_indices]

        for row, support in enumerate(supports):
            support_indices[row, : len(support)] = support
            support_mask[row, : len(support)] = True
            points = coordinates[support]
            span = np.ptp(points, axis=0) if len(points) > 1 else np.zeros(3)
            mechanism_features[row] = (
                span[0],
                span[1],
                span[2],
                len(support),
                -np.log(max(float(local_probabilities[row]), 1e-12)),
            )

        return cls(
            detector_flat_indices=detector_flat_indices,
            mechanism_detector_indices=support_indices,
            mechanism_detector_mask=support_mask,
            mechanism_features=mechanism_features,
            local_mechanism_indices=local_indices,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 5 or tensor.shape[1] != 5:
            raise ValueError("Entrada MicroCNN deve ter formato (B, 5, T, Y, X).")
        encoded = self.encoder(tensor)
        batch, channels, time, height, width = encoded.shape
        flat = encoded.permute(0, 2, 3, 4, 1).reshape(
            batch, time * height * width, channels
        )
        detector_features = flat[:, self.detector_flat_indices, :]
        gathered = detector_features[:, self.mechanism_detector_indices, :]
        mask = self.mechanism_detector_mask[None, :, :, None]
        numeric_mask = mask.to(dtype=gathered.dtype)
        mean_features = (gathered * numeric_mask).sum(dim=2) / numeric_mask.sum(
            dim=2
        ).clamp_min(1)
        max_features = gathered.masked_fill(~mask, torch.finfo(gathered.dtype).min).max(
            dim=2
        ).values
        static_features = self.mechanism_features.unsqueeze(0).expand(batch, -1, -1)
        features = torch.cat((mean_features, max_features, static_features), dim=-1)
        return self.mechanism_head(features).squeeze(-1)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
