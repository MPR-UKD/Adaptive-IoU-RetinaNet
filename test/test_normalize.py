import numpy as np
import pytest
from src.normalization import Normalizer


def test_normalization():
    normalizer = Normalizer()
    # Test input data
    input_sample = {"dcm": np.array([[1, 2], [3, 4]]), "annot": {"x": 1, "y": 2}}

    # Expected output data
    expected_output = {
        "dcm": np.array([[-1.34164079, -0.4472136], [0.4472136, 1.34164079]]),
        "annot": {"x": 1, "y": 2},
    }

    # Normalize input data
    output = normalizer(input_sample)

    # Check if output is correct
    assert np.allclose(output["dcm"], expected_output["dcm"], rtol=1e-3, atol=1e-3)
    assert output["annot"] == expected_output["annot"]


if __name__ == "__main__":
    pytest.main()
