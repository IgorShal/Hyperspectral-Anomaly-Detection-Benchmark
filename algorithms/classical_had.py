"""Scene-based attribute and edge-preserving detection (AED)."""

import math

import numpy as np
from scipy.ndimage import label, maximum_filter
from skimage.filters import threshold_otsu
from skimage.morphology import area_closing, area_opening


def _scale(values, axis=None):
    shifted = values - values.min(axis=axis, keepdims=True)
    span = shifted.max(axis=axis, keepdims=True)
    return shifted / np.where(span > 0, span, 1)


def _recursive_sweep(image, weights):
    for column in range(1, image.shape[1]):
        image[:, column] += weights[:, column - 1] * (image[:, column - 1] - image[:, column])
    for column in range(image.shape[1] - 2, -1, -1):
        image[:, column] += weights[:, column] * (image[:, column + 1] - image[:, column])


def aed(cube, area_threshold=25, sigma_range=0.5):
    """Area attributes and three joint domain-transform filtering passes."""
    pixels = cube.reshape(-1, cube.shape[-1]).astype(np.float64)
    pixels -= pixels.mean(axis=0)
    _, basis = np.linalg.eigh(pixels.T @ pixels)
    guide = _scale(pixels @ basis[:, -3:][:, ::-1], axis=0).reshape(*cube.shape[:2], -1)
    score = np.zeros(cube.shape[:2])
    quantized = np.rint(guide * 1000).astype(np.int16)
    profiles = np.stack([
        area_closing(band, area_threshold=area_threshold, connectivity=1)
        - area_opening(band, area_threshold=area_threshold, connectivity=1)
        for band in quantized.transpose(2, 0, 1)
    ])
    for profile in _scale(profiles):
        dilated = maximum_filter(profile, size=2, origin=-1)
        components, _ = label(dilated > threshold_otsu(dilated), structure=np.ones((3, 3)))
        counts = np.bincount(components.ravel())
        keep = (components != 0) & (counts[components] <= max(1, score.size // 100))
        score += np.where(keep, profile, 0.0)
    score = _scale(score)
    guide_weight = 5 / sigma_range
    horizontal = 1 + guide_weight * np.abs(np.diff(guide, axis=1)).sum(axis=2)
    vertical = 1 + guide_weight * np.abs(np.diff(guide, axis=0)).sum(axis=2)
    for stage in (2, 1, 0):
        sigma = 5 * math.sqrt(3) * 2**stage / math.sqrt(4**3 - 1)
        _recursive_sweep(score, np.exp(-math.sqrt(2) * horizontal / sigma))
        _recursive_sweep(score.T, np.exp(-math.sqrt(2) * vertical.T / sigma))
    return score
