import numpy as np
import torch

from src.models.cnn_predecoder import MicroCNN, count_trainable_parameters
from src.models.lattice import DetectorLattice
from src.models.train_predecoder import FocalBCELoss
from src.quantum.circuits import build_rotated_surface_code_circuit
from src.quantum.dem import DEMMechanisms


def test_microcnn_forward_shape_parameter_budget_and_focal_loss_gradient():
    circuit = build_rotated_surface_code_circuit(distance=3, noise_rate=0.005)
    lattice = DetectorLattice.from_circuit(circuit, distance=3)
    mechanisms = DEMMechanisms.from_dem(
        circuit.detector_error_model(decompose_errors=True)
    )
    model = MicroCNN.from_lattice_and_dem(lattice, mechanisms)
    assert count_trainable_parameters(model) == 13_101
    assert count_trainable_parameters(model) < 25_000

    inputs = torch.from_numpy(
        lattice.tensorize(np.zeros((4, circuit.num_detectors), dtype=np.bool_))
    )
    logits = model(inputs)
    assert logits.shape == (4, len(model.local_mechanism_indices))

    targets = torch.zeros_like(logits)
    targets[:, 0] = 1
    loss = FocalBCELoss()(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder[0].weight.grad is not None
