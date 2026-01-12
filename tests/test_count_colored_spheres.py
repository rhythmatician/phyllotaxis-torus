from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from count_colored_spheres import count_colored_spheres

test_data_path = Path(__file__).parent / "data/png"


@pytest.mark.parametrize(
    "img_path, expected_count",
    [
        (test_data_path / "0.png", 0),
        (test_data_path / "1.png", 1),
        (test_data_path / "5.png", 5),
        (test_data_path / "6.png", 5),  # One sphere is barely in frame
    ],
)
def test_count_colored_spheres(img_path, expected_count):
    # Load the image
    img = np.array(Image.open(img_path)) / 255.0

    result = count_colored_spheres(img)
    assert result["count"] == expected_count
