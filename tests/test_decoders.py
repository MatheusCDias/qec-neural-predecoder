"""Testes de validação para o decodificador PyMatching baseline."""

import numpy as np
import pytest

from src.decoders.baseline_mwpm import PyMatchingDecoder
from src.decoders.hybrid_pipeline import HybridDecoder, IdentityPredecoder
from src.quantum.circuits import (
    build_rotated_surface_code_circuit,
    sample_syndromes,
)


def test_pymatching_decoding_pipeline():
    """Valida se o PyMatching consegue decodificar um lote para d=3."""
    # Ruído físico típico (0.5%)
    circuit = build_rotated_surface_code_circuit(distance=3, noise_rate=0.005)
    batch = sample_syndromes(circuit, num_shots=5000)

    decoder = PyMatchingDecoder(circuit)
    result = decoder.decode_batch(batch.syndromes, batch.observables)

    # Verificações básicas de sanidade
    assert result.predictions.shape == (5000, 1)
    assert 0.0 <= result.logical_error_rate <= 1.0
    assert result.elapsed_time_seconds > 0.0
    assert result.time_per_shot_microseconds > 0.0

    # Sob ruído p=0.005 e d=3, a taxa de erro lógico esperada fica tipicamente abaixo de 10%
    assert result.logical_error_rate < 0.10


def test_decoder_rejects_incompatible_batch_shapes():
    circuit = build_rotated_surface_code_circuit(distance=3, noise_rate=0.005)
    decoder = PyMatchingDecoder(circuit)
    syndromes = np.zeros((2, circuit.num_detectors), dtype=bool)
    observables = np.zeros((2, circuit.num_observables), dtype=bool)

    with pytest.raises(ValueError, match="detectores"):
        decoder.decode_batch(syndromes[:, :-1], observables)

    with pytest.raises(ValueError, match="actual_observables"):
        decoder.decode_batch(syndromes, observables[:1])


def test_decoder_rejects_empty_or_non_boolean_batches():
    circuit = build_rotated_surface_code_circuit(distance=3, noise_rate=0.005)
    decoder = PyMatchingDecoder(circuit)
    observables = np.zeros((1, circuit.num_observables), dtype=bool)

    with pytest.raises(ValueError, match="ao menos um shot"):
        decoder.decode_batch(
            np.zeros((0, circuit.num_detectors), dtype=bool),
            np.zeros((0, circuit.num_observables), dtype=bool),
        )

    with pytest.raises(TypeError, match="booleanas"):
        decoder.decode_batch(
            np.zeros((1, circuit.num_detectors), dtype=np.uint8), observables
        )


def test_identity_hybrid_pipeline_matches_baseline():
    circuit = build_rotated_surface_code_circuit(distance=3, noise_rate=0.005)
    batch = sample_syndromes(circuit, num_shots=256, seed=1234)
    baseline = PyMatchingDecoder(circuit).decode_batch(
        batch.syndromes, batch.observables
    )
    hybrid = HybridDecoder(circuit, IdentityPredecoder(circuit.num_observables))
    result = hybrid.decode_batch(batch.syndromes, batch.observables)

    assert np.array_equal(result.predictions, baseline.predictions)
    assert result.logical_errors == baseline.logical_errors
    assert result.end_to_end_seconds >= result.mwpm_seconds
