import torch

from alzheimer_mri.model import build_model


def test_model_output_shape():
    model = build_model(num_classes=4)
    output = model(torch.zeros(2, 1, 96, 96))
    assert output.shape == (2, 4)
