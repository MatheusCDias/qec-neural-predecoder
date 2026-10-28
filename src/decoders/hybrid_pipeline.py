"""Shared contract and timing path for residual-syndrome predecoders."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import stim

from src.decoders.baseline_mwpm import DecodingResult, PyMatchingDecoder


@dataclass
class PredecoderOutput:
    """Result of a predecoder stage.

    ``residual_syndromes`` is passed to MWPM. ``observable_corrections`` is
    XOR-composed with MWPM's prediction, so removing a defect must always be
    accompanied by the corresponding logical correction information.
    """

    residual_syndromes: np.ndarray
    observable_corrections: np.ndarray


class SyndromePredecoder(Protocol):
    def predecode(self, syndromes: np.ndarray) -> PredecoderOutput:
        """Return residual detector events and accumulated observable flips."""


class IdentityPredecoder:
    """No-op control used to measure pipeline overhead before a model is added."""

    def __init__(self, num_observables: int = 1):
        self.num_observables = num_observables

    def predecode(self, syndromes: np.ndarray) -> PredecoderOutput:
        return PredecoderOutput(
            residual_syndromes=syndromes.copy(),
            observable_corrections=np.zeros(
                (syndromes.shape[0], self.num_observables), dtype=np.bool_
            ),
        )


@dataclass
class HybridDecodingResult:
    predictions: np.ndarray
    logical_errors: int
    logical_error_rate: float
    predecoder_seconds: float
    mwpm_seconds: float
    end_to_end_seconds: float
    end_to_end_us_per_shot: float


class HybridDecoder:
    """Runs a predecoder followed by PyMatching and composes logical corrections."""

    def __init__(self, circuit: stim.Circuit, predecoder: SyndromePredecoder):
        self.predecoder = predecoder
        self.mwpm = PyMatchingDecoder(circuit)
        self.num_detectors = circuit.num_detectors
        self.num_observables = circuit.num_observables

    def decode_batch(
        self, syndromes: np.ndarray, actual_observables: np.ndarray
    ) -> HybridDecodingResult:
        started = time.perf_counter()
        pre_started = time.perf_counter()
        output = self.predecoder.predecode(syndromes)
        predecoder_seconds = time.perf_counter() - pre_started
        if output.residual_syndromes.shape != syndromes.shape:
            raise ValueError("A síndrome residual deve manter o formato da entrada.")
        expected_correction_shape = (syndromes.shape[0], self.num_observables)
        if output.observable_corrections.shape != expected_correction_shape:
            raise ValueError(
                "observable_corrections deve ter formato "
                f"{expected_correction_shape}; recebido "
                f"{output.observable_corrections.shape}."
            )
        if (
            output.residual_syndromes.dtype != np.bool_
            or output.observable_corrections.dtype != np.bool_
        ):
            raise TypeError("Saídas do pré-decodificador devem ser booleanas.")

        mwpm_result: DecodingResult = self.mwpm.decode_batch(
            output.residual_syndromes, actual_observables
        )
        predictions = np.logical_xor(
            mwpm_result.predictions, output.observable_corrections
        )
        logical_errors = int(np.count_nonzero(np.any(predictions != actual_observables, axis=1)))
        end_to_end_seconds = time.perf_counter() - started
        shots = syndromes.shape[0]
        return HybridDecodingResult(
            predictions=predictions,
            logical_errors=logical_errors,
            logical_error_rate=logical_errors / shots,
            predecoder_seconds=predecoder_seconds,
            mwpm_seconds=mwpm_result.elapsed_time_seconds,
            end_to_end_seconds=end_to_end_seconds,
            end_to_end_us_per_shot=end_to_end_seconds / shots * 1e6,
        )
