import numpy as np
from PIL import Image

from alzheimer_mri.audit import entropy, perceptual_hash


def test_entropy_of_constant_image_is_zero():
    assert entropy(np.zeros((8, 8), dtype=np.uint8)) == 0.0


def test_perceptual_hash_is_stable_and_64_bit():
    image = Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16))
    first = perceptual_hash(image)
    second = perceptual_hash(image.copy())
    assert first == second
    assert len(first) == 16
