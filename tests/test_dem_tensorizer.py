import numpy as np
import pytest

from src.models.lattice import DetectorLattice
from src.quantum.circuits import build_rotated_surface_code_circuit
from src.quantum.dem import DEMMechanisms


@pytest.mark.parametrize("distance", [3, 5])
def test_detector_lattice_tensor_shape_and_occupancy(distance):
    circuit = build_rotated_surface_code_circuit(distance=distance, noise_rate=0.005)
    lattice = DetectorLattice.from_circuit(circuit, distance=distance)
    sampler = circuit.compile_detector_sampler(seed=1000 + distance)
    syndromes, _ = sampler.sample(shots=32, separate_observables=True)

    tensor = lattice.tensorize(syndromes)
    expected_shape = (32, 5, distance + 1, distance + 1, distance + 1)
    assert tensor.shape == expected_shape
    assert lattice.occupancy.sum() == circuit.num_detectors
    assert np.all(tensor[:, 1].sum(axis=(1, 2, 3)) == circuit.num_detectors)
    assert np.all(tensor[:, 0, ~lattice.occupancy] == 0)

    for detector, (it, iy, ix) in enumerate(lattice.voxel_indices):
        np.testing.assert_array_equal(tensor[:, 0, it, iy, ix], syndromes[:, detector])


@pytest.mark.parametrize("distance", [3, 5])
def test_dem_mechanism_masks_recompose_sampled_syndromes_and_observables(distance):
    circuit = build_rotated_surface_code_circuit(distance=distance, noise_rate=0.005)
    dem = circuit.detector_error_model(decompose_errors=True)
    mechanisms = DEMMechanisms.from_dem(dem)
    detector_samples, observable_samples, error_samples = dem.compile_sampler(
        seed=2000 + distance
    ).sample(shots=128, return_errors=True)

    assert mechanisms.num_mechanisms == dem.num_errors
    assert error_samples.shape == (128, dem.num_errors)
    reconstructed_detectors, reconstructed_observables = mechanisms.reconstruct(
        error_samples
    )
    np.testing.assert_array_equal(reconstructed_detectors, detector_samples)
    np.testing.assert_array_equal(reconstructed_observables, observable_samples)

    coordinates = DetectorLattice.from_circuit(circuit, distance).coordinates
    local_indices = np.flatnonzero(mechanisms.local_mask(coordinates, max_span=1))
    local_activations = error_samples[:, local_indices]
    local_detectors, local_observables = mechanisms.reconstruct(
        local_activations, mechanism_indices=local_indices
    )

    # Independent XOR accumulation checks the local GF(2) projection itself.
    expected_detectors = np.zeros_like(local_detectors)
    expected_observables = np.zeros_like(local_observables)
    for column, mechanism in enumerate(local_indices):
        active = local_activations[:, column, None]
        expected_detectors ^= active & mechanisms.detector_masks[mechanism]
        expected_observables ^= active & mechanisms.observable_masks[mechanism]
    np.testing.assert_array_equal(local_detectors, expected_detectors)
    np.testing.assert_array_equal(local_observables, expected_observables)


def test_dem_mechanism_extraction_flattens_repeated_blocks_for_distance_7():
    circuit = build_rotated_surface_code_circuit(distance=7, noise_rate=0.005)
    dem = circuit.detector_error_model(decompose_errors=True)
    mechanisms = DEMMechanisms.from_dem(dem)
    detector_samples, observable_samples, error_samples = dem.compile_sampler(
        seed=2007
    ).sample(shots=16, return_errors=True)

    assert mechanisms.num_mechanisms == dem.num_errors
    reconstructed = mechanisms.reconstruct(error_samples)
    np.testing.assert_array_equal(reconstructed[0], detector_samples)
    np.testing.assert_array_equal(reconstructed[1], observable_samples)
