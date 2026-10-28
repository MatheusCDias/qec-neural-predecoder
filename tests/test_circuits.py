"""Testes de consistência para a geração de circuitos no Stim."""

from src.quantum.circuits import (
    build_rotated_surface_code_circuit,
    sample_syndromes,
)


def test_detector_counts_for_target_distances():
    """Valida se o número total de detectores bate com a dedução d * (d^2 - 1)."""
    expected_detectors = {
        3: 24,  # 3 * (9 - 1) = 24
        5: 120,  # 5 * (25 - 1) = 120
        7: 336,  # 7 * (49 - 1) = 336
    }

    for d, expected_bits in expected_detectors.items():
        circuit = build_rotated_surface_code_circuit(distance=d, noise_rate=0.005)
        assert circuit.num_detectors == expected_bits, (
            f"Falha em d={d}: esperado {expected_bits}, obtido {circuit.num_detectors}"
        )


def test_sampling_shapes():
    """Verifica se o amostrador vetorial retorna as matrizes nas dimensões corretas."""
    circuit = build_rotated_surface_code_circuit(distance=3, noise_rate=0.005)
    batch = sample_syndromes(circuit, num_shots=1000)

    assert batch.syndromes.shape == (1000, 24)
    assert batch.observables.shape == (1000, 1)


def test_sampling_is_reproducible_with_seed():
    circuit = build_rotated_surface_code_circuit(distance=3, noise_rate=0.005)
    first = sample_syndromes(circuit, num_shots=256, seed=20261028)
    second = sample_syndromes(circuit, num_shots=256, seed=20261028)

    assert (first.syndromes == second.syndromes).all()
    assert (first.observables == second.observables).all()
