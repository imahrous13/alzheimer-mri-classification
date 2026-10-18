import numpy as np
from PIL import Image

from alzheimer_mri.audit import discover, entropy, normalize_dataset_name, perceptual_hash


def test_entropy_of_constant_image_is_zero():
    assert entropy(np.zeros((8, 8), dtype=np.uint8)) == 0.0


def test_perceptual_hash_is_stable_and_64_bit():
    image = Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16))
    first = perceptual_hash(image)
    second = perceptual_hash(image.copy())
    assert first == second
    assert len(first) == 16


def test_dataset_names_are_canonicalized():
    assert normalize_dataset_name("OriginalDataset") == "original"
    assert normalize_dataset_name("AugmentedAlzheimerDataset") == "augmented"
    assert normalize_dataset_name("original") == "original"
    assert normalize_dataset_name("augmented") == "augmented"


def test_discover_maps_archive_folder_names_to_canonical_names(tmp_path):
    for dataset, label in [
        ("OriginalDataset", "MildDemented"),
        ("AugmentedAlzheimerDataset", "MildDemented"),
    ]:
        directory = tmp_path / dataset / label
        directory.mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(directory / "a.jpg")
    discovered = discover(tmp_path)
    assert {record[1] for record in discovered} == {"original", "augmented"}
