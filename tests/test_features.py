import numpy as np
from PIL import Image

from alzheimer_mri.baseline import image_features


def test_hog_feature_length_matches_documented_descriptor(tmp_path):
    path = tmp_path / "scan.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (128, 128), dtype=np.uint8)).save(path)
    features = image_features(path, image_size=64)
    # 7x7 HOG blocks * (2*2 cells * 9 orientations) = 1764, plus a 16-bin histogram.
    assert features.shape == (1780,)
    assert np.isfinite(features).all()


def test_features_are_deterministic(tmp_path):
    path = tmp_path / "scan.png"
    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(0, 255, (176, 208), dtype=np.uint8)).save(path)
    first = image_features(path, image_size=64)
    second = image_features(path, image_size=64)
    np.testing.assert_array_equal(first, second)
