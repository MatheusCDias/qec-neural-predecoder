"""Extract binary detector/observable masks for Stim DEM error mechanisms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim


@dataclass(frozen=True)
class DEMMechanisms:
    """One row per DEM ``error`` instruction, in Stim sampler order."""

    detector_masks: np.ndarray  # (num_mechanisms, num_detectors), bool
    observable_masks: np.ndarray  # (num_mechanisms, num_observables), bool
    probabilities: np.ndarray  # (num_mechanisms,), float64
    num_detectors: int
    num_observables: int

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> "DEMMechanisms":
        # Flattening expands REPEAT blocks and applies their detector shifts,
        # preserving the order used by CompiledDemSampler.return_errors.
        flattened = dem.flattened()
        instructions = [item for item in flattened if item.type == "error"]
        if len(instructions) != dem.num_errors:
            raise ValueError(
                "A contagem de instruções error do DEM achatado diverge de num_errors."
            )

        detector_masks = np.zeros((len(instructions), dem.num_detectors), dtype=np.bool_)
        observable_masks = np.zeros(
            (len(instructions), dem.num_observables), dtype=np.bool_
        )
        probabilities = np.empty(len(instructions), dtype=np.float64)

        for mechanism, instruction in enumerate(instructions):
            args = instruction.args_copy()
            if len(args) != 1 or not 0 <= args[0] <= 1:
                raise ValueError(f"Probabilidade inválida no mecanismo {mechanism}: {args}.")
            probabilities[mechanism] = args[0]
            for target in instruction.targets_copy():
                if target.is_separator():
                    # Separators mark graphlike components of one correlated fault.
                    # The physical activation still toggles their total XOR support.
                    continue
                if target.is_relative_detector_id():
                    index = target.val
                    if not 0 <= index < dem.num_detectors:
                        raise ValueError(f"Detector D{index} fora dos limites do DEM.")
                    detector_masks[mechanism, index] ^= True
                elif target.is_logical_observable_id():
                    index = target.val
                    if not 0 <= index < dem.num_observables:
                        raise ValueError(f"Observável L{index} fora dos limites do DEM.")
                    observable_masks[mechanism, index] ^= True
                else:
                    raise ValueError(
                        f"Target DEM não suportado no mecanismo {mechanism}: {target}."
                    )

        return cls(
            detector_masks=detector_masks,
            observable_masks=observable_masks,
            probabilities=probabilities,
            num_detectors=dem.num_detectors,
            num_observables=dem.num_observables,
        )

    @property
    def num_mechanisms(self) -> int:
        return self.detector_masks.shape[0]

    def local_mask(
        self, detector_coordinates: np.ndarray, max_span: int = 1
    ) -> np.ndarray:
        """Select detector-bearing mechanisms local in scaled ``(x,y,t)`` cells.

        Mechanisms with no detector support are excluded because they cannot be
        identified from a syndrome. Boundary mechanisms with one detector are
        local by construction.
        """
        coordinates = np.asarray(detector_coordinates)
        if coordinates.shape != (self.num_detectors, 3):
            raise ValueError(
                f"detector_coordinates deve ter formato ({self.num_detectors}, 3)."
            )
        if max_span < 0:
            raise ValueError("max_span deve ser não negativo.")
        local = np.zeros(self.num_mechanisms, dtype=np.bool_)
        for index, mask in enumerate(self.detector_masks):
            support = coordinates[mask]
            if len(support) == 0:
                continue
            span = np.ptp(support, axis=0)
            local[index] = bool(np.all(span <= max_span))
        return local

    def reconstruct(
        self,
        activations: np.ndarray,
        mechanism_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recompose detector and observable parity from bool mechanism activations."""
        activations = np.asarray(activations)
        if activations.ndim != 2 or activations.dtype != np.bool_:
            raise TypeError("activations deve ser uma matriz booleana 2D.")
        if mechanism_indices is None:
            if activations.shape[1] != self.num_mechanisms:
                raise ValueError(
                    f"Esperadas {self.num_mechanisms} ativações de mecanismos; "
                    f"recebidas {activations.shape[1]}."
                )
            detector_masks = self.detector_masks
            observable_masks = self.observable_masks
        else:
            mechanism_indices = np.asarray(mechanism_indices, dtype=np.int64)
            if mechanism_indices.ndim != 1 or np.any(
                (mechanism_indices < 0) | (mechanism_indices >= self.num_mechanisms)
            ):
                raise ValueError("mechanism_indices contém índices DEM inválidos.")
            if activations.shape[1] != len(mechanism_indices):
                raise ValueError(
                    "O número de colunas das ativações deve corresponder a "
                    "mechanism_indices."
                )
            detector_masks = self.detector_masks[mechanism_indices]
            observable_masks = self.observable_masks[mechanism_indices]

        detectors = (activations.astype(np.int32) @ detector_masks.astype(np.int32)) % 2
        observables = (
            activations.astype(np.int32) @ observable_masks.astype(np.int32)
        ) % 2
        return detectors.astype(np.bool_), observables.astype(np.bool_)
