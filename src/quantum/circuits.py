"""Módulo para geração e compilação de circuitos de Rotated Surface Code via Stim."""

from dataclasses import dataclass

import numpy as np
import stim


@dataclass
class QECSampleBatch:
    """Contêiner para os dados amostrados de um circuito de correção de erros."""

    syndromes: np.ndarray  # Matriz booleana de detectores (N_shots, N_detectores)
    observables: np.ndarray  # Vetor booleano de flips lógicos (N_shots, N_observables)
    num_detectors: int
    num_shots: int


def build_rotated_surface_code_circuit(
    distance: int,
    rounds: int | None = None,
    noise_rate: float = 0.005,
    basis: str = "Z",
) -> stim.Circuit:
    """Gera um circuito de Rotated Surface Code sob ruído de circuito (circuit-level noise).

    Args:
        distance: Distância do código (d = 3, 5, 7).
        rounds: Número de rodadas de medições de estabilizadores (padrão: d).
        noise_rate: Probabilidade de erro físico por porta/medição (ex.: 0.005 = 0.5%).
        basis: Base lógica a ser medida ('Z' para memória de bit-flip, 'X' para phase-flip).

    Returns:
        stim.Circuit compilado com os detectores e observáveis anotados.
    """
    if rounds is None:
        rounds = distance

    # O Stim possui um gerador nativo otimizado para Rotated Surface Codes com circuit-level noise
    circuit_type = f"surface_code:rotated_memory_{basis.lower()}"

    circuit = stim.Circuit.generated(
        circuit_type,
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=noise_rate,
        after_reset_flip_probability=noise_rate,
        before_measure_flip_probability=noise_rate,
        before_round_data_depolarization=noise_rate,
    )
    return circuit


def sample_syndromes(
    circuit: stim.Circuit, num_shots: int = 10_000, seed: int | None = None
) -> QECSampleBatch:
    """Amostra síndromes (detectores) e o observável lógico verdadeiro usando instruções SIMD/AVX."""
    if num_shots <= 0:
        raise ValueError("num_shots deve ser maior que zero.")
    sampler = circuit.compile_detector_sampler(seed=seed)
    syndromes, observables = sampler.sample(
        shots=num_shots, separate_observables=True
    )

    return QECSampleBatch(
        syndromes=syndromes,
        observables=observables,
        num_detectors=circuit.num_detectors,
        num_shots=num_shots,
    )
