"""Módulo de decodificação clássica via Minimum Weight Perfect Matching (MWPM) usando PyMatching v2."""

import time
from dataclasses import dataclass

import numpy as np
import pymatching
import stim


@dataclass
class DecodingResult:
    """Resultados de uma rodada de decodificação em lote."""

    predictions: np.ndarray  # Predições do observável (N_shots, N_observables)
    logical_errors: int  # Número total de predições incorretas
    logical_error_rate: float  # Taxa de erro lógico (p_L)
    elapsed_time_seconds: float  # Tempo total de decodificação
    time_per_shot_microseconds: float  # Latência média por tiro em µs


class PyMatchingDecoder:
    """Wrapper para o decodificador Sparse Blossom do PyMatching v2."""

    def __init__(self, circuit: stim.Circuit):
        """Inicializa o decodificador compilando o Detector Error Model (DEM).

        Args:
            circuit: Circuito do Stim contendo as anotações de detectores e ruído.
        """
        self.circuit = circuit
        # Extrai o modelo de erro do detector (grafo DEM) diretamente do circuito
        self.dem = circuit.detector_error_model(decompose_errors=True)
        # Compila o grafo esparso de emparelhamento (Sparse Blossom)
        self.matcher = pymatching.Matching.from_detector_error_model(self.dem)

    def decode_batch(
        self, syndromes: np.ndarray, actual_observables: np.ndarray
    ) -> DecodingResult:
        """Decodifica uma matriz de síndromes e calcula a taxa de erro lógico.

        Args:
            syndromes: Matriz booleana de detectores (N_shots, N_detectores).
            actual_observables: Vetor booleano de observáveis reais (N_shots, 1).

        Returns:
            DecodingResult contendo as métricas de tempo e taxa de erro.
        """
        syndromes = np.asarray(syndromes)
        actual_observables = np.asarray(actual_observables)

        if syndromes.ndim != 2:
            raise ValueError("syndromes deve ter formato (num_shots, num_detectors).")
        if actual_observables.ndim != 2:
            raise ValueError(
                "actual_observables deve ter formato (num_shots, num_observables)."
            )

        num_shots = syndromes.shape[0]
        if num_shots == 0:
            raise ValueError("O lote deve conter ao menos um shot.")
        if syndromes.shape[1] != self.circuit.num_detectors:
            raise ValueError(
                f"Esperados {self.circuit.num_detectors} detectores por shot; "
                f"recebidos {syndromes.shape[1]}."
            )
        if actual_observables.shape != (num_shots, self.circuit.num_observables):
            raise ValueError(
                "actual_observables deve ter formato "
                f"({num_shots}, {self.circuit.num_observables}); "
                f"recebido {actual_observables.shape}."
            )
        if syndromes.dtype != np.bool_ or actual_observables.dtype != np.bool_:
            raise TypeError("syndromes e actual_observables devem ser matrizes booleanas.")

        start_time = time.perf_counter()
        # O PyMatching v2 aceita diretamente matrizes 2D numpy com instruções C++ otimizadas
        predictions = self.matcher.decode_batch(syndromes)
        elapsed = time.perf_counter() - start_time

        # Um erro lógico ocorre quando a predição diverge do observável real
        discrepancies = np.any(predictions != actual_observables, axis=1)
        num_errors = int(np.sum(discrepancies))
        error_rate = num_errors / num_shots
        lat_us = (elapsed / num_shots) * 1e6

        return DecodingResult(
            predictions=predictions,
            logical_errors=num_errors,
            logical_error_rate=error_rate,
            elapsed_time_seconds=elapsed,
            time_per_shot_microseconds=lat_us,
        )
