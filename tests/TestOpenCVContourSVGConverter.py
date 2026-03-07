import cv2
import pytest
from pathlib import Path
import numpy as np
from OpenCVContourSVGConverter import OpenCVContourSVGConverter

def test_basic():
    # Define square corner points
    square = np.array([
        [100, 100],
        [200, 100],
        [200, 200],
        [100, 200]
    ], dtype=np.int32)

    # Convert to OpenCV contour format (N,1,2)
    square_contour = square.reshape((-1, 1, 2))

    converter = OpenCVContourSVGConverter()

    svg, width, height = converter.convert([square_contour])

    print(svg)

    assert svg is not None
    assert width == 100
    assert height == 100

    assert True