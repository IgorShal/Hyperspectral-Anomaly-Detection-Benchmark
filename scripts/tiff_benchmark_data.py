"""Prepare the supplied TIFF scene and ground-truth pairs for the benchmark."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.io import loadmat, savemat
from tifffile import TiffFile

from scripts.comparison_helpers import atomic


AVIRIS_NG_BANDS = np.r_[15:109, 118:145, 158:187, 227:274, 328:407]


def load_prepared_scene(folder, scene):
    prepared = loadmat(folder / f"{scene}.mat", variable_names=("data", "map"))
    return prepared["data"], prepared["map"]


def read_tiff(path):
    with TiffFile(path) as handle:
        series = handle.series[0]
        return series.asarray(), series.axes


def prepare_tiff_data(source, destination, requested=None):
    paths = sorted(path for path in source.iterdir() if path.suffix.lower() in {".tif", ".tiff"})
    images = {}
    masks = {}
    for path in paths:
        stem = path.stem
        registry = masks if stem.lower().endswith("_gt") else images
        key = stem[:-3] if registry is masks else stem
        if key in registry:
            raise ValueError(f"Duplicate TIFF scene or mask: {key}")
        registry[key] = path
    if not images:
        raise ValueError(f"No TIFF scenes found in {source}")
    if set(images) != set(masks):
        raise ValueError(f"Incomplete TIFF pairs: {sorted(set(images) ^ set(masks))}")

    scenes = {}
    for key in images:
        prefix, separator, name = key.partition("_")
        scene = name if separator and prefix.isdigit() and name else key
        if scene in scenes:
            raise ValueError(f"Duplicate scene name: {scene}")
        scenes[scene] = key
    selected = list(dict.fromkeys(requested)) if requested else list(scenes)
    unknown = sorted(set(selected) - scenes.keys())
    if unknown:
        raise ValueError(f"Unknown scenes: {', '.join(unknown)}. Available: {', '.join(scenes)}")

    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination.parent / "scenes.json"
    records = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    for scene in selected:
        key = scenes[scene]
        path = images[key]
        started = perf_counter()
        cube, axes = read_tiff(path)
        mask, mask_axes = read_tiff(masks[key])
        if cube.ndim != 3 or len(axes) != 3 or "Y" not in axes or "X" not in axes:
            raise ValueError(f"{path.name}: unsupported TIFF axes {axes}")
        spatial = (axes.index("Y"), axes.index("X"))
        spectral = next(i for i in range(3) if i not in spatial)
        cube = cube.transpose(*spatial, spectral)
        if mask_axes != "YX" or mask.shape != cube.shape[:2]:
            raise ValueError(f"{path.name}: mask {mask.shape} does not align with {cube.shape}")
        mask = (mask > 0).astype(np.uint8)
        positive = int(mask.sum())
        if not 0 < positive < mask.size:
            raise ValueError(f"{path.name}: ROC AUC requires both classes")
        channels = cube.shape[-1]
        if scene.lower().startswith("hydice"):
            indices = np.arange(channels)
        elif channels == 425:
            indices = AVIRIS_NG_BANDS[:186]
        elif channels >= 186:
            indices = np.arange(186)
        else:
            raise ValueError(f"{path.name}: fewer than 186 AVIRIS channels")
        cube = np.ascontiguousarray(cube[..., indices], dtype=np.float32)
        if not np.isfinite(cube).all():
            raise ValueError(f"{path.name}: non-finite values in retained bands")
        prep_s = perf_counter() - started
        target = destination / f"{scene}.mat"
        temporary = target.with_suffix(".tmp")
        savemat(temporary, {"data": cube, "map": mask}, appendmat=False)
        temporary.replace(target)
        records[scene] = {"image": path.name, "mask": masks[key].name, "shape": cube.shape,
                          "input_channels": channels, "bands_1_based": (indices + 1).tolist(),
                          "anomaly_pixels": positive, "prep_s": prep_s}
    atomic(manifest, json.dumps(records, indent=2))
    return selected
