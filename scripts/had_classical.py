"""Classical HAD detectors with settings from the TIFF calibration run."""

import numpy as np
from threadpoolctl import threadpool_limits

from algorithms.classical_had import aed


PARAMETERS = {
    "aed": {"area_threshold": 64, "sigma_range": 0.5},
    "lrx": {"inner": 5, "outer": 9, "delta": 1e-5},
    "rx": {},
}


def _local_rx(cube, inner, outer, delta):
    # Match LRXfunc.m's global scaling, wrapped borders and uncentered Gram.
    image = np.array(cube, dtype=np.float64, copy=True)
    image -= image.min()
    image /= image.max() or 1.0
    height, width, _ = image.shape
    radius = outer // 2
    row, col = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    ring = np.maximum(abs(row), abs(col)) > inner // 2
    dy, dx = row[ring], col[ring]
    scores = np.empty(height * width)
    for start in range(0, scores.size, 64):
        index = np.arange(start, min(start + 64, scores.size))
        y, x = np.divmod(index, width)
        neighbors = image[(y[:, None] + dy) % height, (x[:, None] + dx) % width]
        residual = image[y, x] - neighbors.mean(axis=1)
        # Woodbury identity reduces the solve to the number of ring samples.
        gram = neighbors @ neighbors.transpose(0, 2, 1)
        diagonal = np.arange(dy.size)
        gram[:, diagonal, diagonal] += delta
        projected = neighbors @ residual[:, :, None]
        solved = np.linalg.solve(gram, projected)
        scores[index] = (np.einsum("ij,ij->i", residual, residual)
                         - (projected * solved).sum(axis=(1, 2))) / delta
    scores -= scores.min()
    return (scores / (scores.max() or 1.0)).reshape(height, width)


def predict(method, cube):
    """Return a CPU H×W anomaly map, using the same settings on each scene."""
    with threadpool_limits(limits=1, user_api="blas"):
        if method == "aed":
            return aed(cube, **PARAMETERS[method])
        if method == "lrx":
            return _local_rx(cube, **PARAMETERS[method])
        if method == "rx":
            pixels = cube.reshape(-1, cube.shape[-1]).astype(np.float64)
            pixels -= pixels.mean(axis=0)
            precision = np.linalg.pinv(np.atleast_2d(np.cov(pixels, rowvar=False)), hermitian=True)
            return np.einsum("ij,ij->i", pixels @ precision, pixels).reshape(cube.shape[:2])
    raise ValueError(f"Unknown classical method: {method}")
