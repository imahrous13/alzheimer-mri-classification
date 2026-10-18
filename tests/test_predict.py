import numpy as np
from PIL import Image

from alzheimer_mri.predict import discover_images


def test_discover_images_accepts_single_file(tmp_path):
    path = tmp_path / "scan.jpg"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(path)
    assert discover_images(path) == [path]


def test_discover_images_recurses_and_filters_extensions(tmp_path):
    (tmp_path / "nested").mkdir()
    kept = tmp_path / "nested" / "a.PNG"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(kept)
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")
    found = discover_images(tmp_path)
    assert kept in found
    assert all(item.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"} for item in found)
