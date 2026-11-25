"""Run the pinned author implementations on the benchmark's TIFF scenes."""

import importlib.util
import random
import sys
from contextlib import contextmanager, nullcontext
from functools import cache
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.comparison_helpers import score_auc, validation_stage


AUTHOR_CODE = Path(__file__).resolve().parents[1] / "algorithms"


@cache
def _transformer_sources(method):
    folder = AUTHOR_CODE / ("sm_had" if method == "smhad" else "gt_had/dnnmethods/GT-HAD")
    if not (folder / "net.py").is_file():
        raise FileNotFoundError(f"Author source missing at {folder}. See the pinned source revisions in README.md.")
    modules = {}
    for name in ("utils", "block", "net", "data"):
        spec = importlib.util.spec_from_file_location(f"_author_{method}_{name}", folder / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        if name == "data":
            # The upstream DatasetHsi uses absolute imports for its siblings.
            previous = {key: sys.modules.get(key) for key in ("utils", "block")}
            try:
                sys.modules.update({key: modules[key] for key in previous})
                spec.loader.exec_module(module)
            finally:
                for key, original in previous.items():
                    if original is None:
                        del sys.modules[key]
                    else:
                        sys.modules[key] = original
        else:
            spec.loader.exec_module(module)
        modules[name] = module
    return modules


def _fold_patches(blocks, padding, height, width, stride):
    # Folding on the padded grid handles asymmetric padding on either axis.
    left, right, top, bottom = padding
    size = (height + top + bottom, width + left + right)
    window = blocks.shape[-1]
    columns = blocks.flatten(1).T.unsqueeze(0)
    folded = F.fold(columns, size, window, stride=stride)
    counts = F.fold(columns.new_ones((1, window * window, len(blocks))), size, window, stride=stride)
    return (folded / counts)[:, :, top:top + height, left:left + width]


def _transformer_run(method, cube, seed, epochs, curve, validation):
    sources = _transformer_sources(method)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    def prepare_image():
        normalized = cube.transpose(2, 0, 1).astype(np.float32)
        normalized -= normalized.min()
        normalized /= normalized.max()
        return torch.from_numpy(normalized).cuda()[None]

    image = prepare_image()
    _, bands, height, width = image.shape
    window, stride, embed_dim = (15, 5, 96) if method == "smhad" else (9, 3, 64)
    dataset = sources["data"].DatasetHsi(image, wsize=window, wstride=stride)
    matcher = sources["block"].Block_search(image, wsize=window, wstride=stride)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    matches = image.new_zeros(len(dataset))
    indexes = torch.arange(len(dataset), device=image.device)
    inner_patch = 3
    grid = window // inner_patch
    model = sources["net"].Net(in_chans=bands, embed_dim=embed_dim, patch_size=inner_patch,
                                patch_stride=grid, mlp_ratio=2.0).cuda()
    if method == "smhad":
        # MVAB's published code counts tokens correctly only for a 3×3 grid.
        # A 15×15 window contains 5×5 of the paper's 3×3 subpatches.
        attention = model.attn_layer.mvab
        attention.N = grid * grid
        attention.mask = torch.diag(attention.mask.new_full((attention.N,), -100.0))
    optimizer = torch.optim.Adam(sources["utils"].get_params(model), lr=1e-3)

    @torch.inference_mode()
    def predict_from(patches, patch_indexes, gate_values):
        residuals = []
        for inputs, ids in zip(patches.block_input.split(64), patch_indexes.split(64)):
            output = model(inputs, block_idx=ids, match_vec=gate_values)
            residuals.append(F.avg_pool3d((inputs - output).square(), (5, 3, 3),
                                         stride=1, padding=(2, 1, 1)))
        folded = _fold_patches(torch.cat(residuals), patches.padding, height, width, stride)
        return sources["utils"].img2mask(folded)

    for epoch in range(1, epochs + 1):
        search = epoch % 25 == 0 and epoch < epochs
        if search:
            reconstructed = torch.empty_like(dataset.block_input)
        loss_sum = 0.0
        for batch in loader:
            ids = batch["index"].cuda()
            optimizer.zero_grad()
            output = model(batch["block_input"], block_idx=ids, match_vec=matches)
            if search:
                reconstructed[ids] = output.detach()
            loss = F.mse_loss(output, batch["block_gt"])
            loss.backward()
            optimizer.step()
            if curve is not None:
                loss_sum += loss.detach() * len(ids)
        if search:
            folded = _fold_patches(reconstructed, dataset.padding, height, width, stride)
            matcher(folded, matches.zero_(), indexes)
            del reconstructed, folded
        if curve is not None:
            with validation_stage(validation):
                model.eval()
                curve(loss_sum.item() / len(dataset), predict_from(dataset, indexes, matches))
                model.train()
    model.eval()

    @torch.inference_mode()
    def predict(*, recompute_cmm=False):
        scene_input = prepare_image()
        patches = sources["data"].DatasetHsi(scene_input, wsize=window, wstride=stride)
        if not recompute_cmm:
            return predict_from(patches, indexes, matches)

        scene_indexes = torch.arange(len(patches), device=scene_input.device)
        gates = scene_input.new_zeros(len(patches))
        reconstructed = torch.cat([
            model(inputs, block_idx=ids, match_vec=gates)
            for inputs, ids in zip(patches.block_input.split(64), scene_indexes.split(64))
        ])
        folded = _fold_patches(reconstructed, patches.padding, height, width, stride)
        del reconstructed
        search = sources["block"].Block_search(scene_input, wsize=window, wstride=stride)
        gates = search(folded, gates, scene_indexes)
        return predict_from(patches, scene_indexes, gates)

    return predict


def transformer_predictor(method, cube, mask, scene, *, epochs, loss_history=None, collect_curve=False,
                          validation=None):
    sources = _transformer_sources(method)
    seeds = (0, 100, 200) if method == "smhad" else (sources["utils"].seed_dict.get(scene, 0),)
    details = {"epochs": epochs, "best_epoch": epochs, "seed_auc": {}}
    auc_curve = []

    def record(loss, scores):
        loss_history.append(loss)
        auc_curve.append(score_auc(scores, mask))

    best_score = -float("inf")
    for seed in seeds[:1] if collect_curve else seeds:
        print(f"[{method}] {scene}: seed={seed}, epochs={epochs}", flush=True)
        prediction = _transformer_run(method, cube, seed, epochs, record if collect_curve else None, validation)
        with validation_stage(validation):
            auc = score_auc(prediction(), mask)
        details["seed_auc"][str(seed)] = auc
        if auc > best_score:
            predict, best_score = prediction, auc
            details["seed"] = seed
    details["auc_last_epoch"] = best_score
    if collect_curve:
        details["epoch_curve"] = auc_curve
    return predict, details


@cache
def _dms_sources():
    from importlib import import_module

    folder = AUTHOR_CODE / "dms2f_had"
    if not (folder / "src/train.py").is_file():
        raise FileNotFoundError(f"Author source missing at {folder}. See the pinned source revisions in README.md.")
    # The author's snapshot imports two unpublished modules. The tracker is
    # only invoked by decoder ablations, which this benchmark does not run.
    paths = ModuleType("src.project_paths")
    paths.REPO_ROOT = folder
    paths.DATA_DIR = folder / "Data"
    paths.EVAL_DATASET_DIR = paths.DATA_DIR
    paths.MODELS_DIR = folder / "Models"
    paths.RESULTS_DIR = folder / "Results"
    tracker = ModuleType("src.auc_tracker_abalation")

    def unused_tracker(*args, **kwargs):
        raise RuntimeError("The author did not publish the ablation tracker")

    tracker.save_auc_result = unused_tracker
    sys.path.insert(0, str(folder))
    sys.modules[paths.__name__] = paths
    sys.modules[tracker.__name__] = tracker
    try:
        return tuple(import_module(f"src.{name}") for name in ("data_loader", "model", "train"))
    finally:
        sys.path.remove(str(folder))
        del sys.modules[paths.__name__], sys.modules[tracker.__name__]


class _DmsTrainingTorch:
    """Time the pinned trainer's no-grad validation and saves without editing it."""

    def __init__(self, validation):
        self.validation = validation

    def __getattr__(self, name):
        return getattr(torch, name)

    @contextmanager
    def no_grad(self):
        with validation_stage(self.validation), torch.no_grad():
            yield

    def save(self, *args, **kwargs):
        with validation_stage(self.validation, check=False):
            return torch.save(*args, **kwargs)


def dms2f_predictor(cube, mask, scene, *, epochs, loss_history=None, collect_curve=False,
                    select_best=True, validation=None):
    data, architecture, training = _dms_sources()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    dataset = data.HSIDataset({"data": cube, "map": (mask > 0).astype(np.uint8)}, scene,
                              block_size=16, stride=8)
    model = architecture.AnomalyDetectionModel(cube.shape[-1], mode="full", dim=64, depth=1,
                                               spec_num=16, spec_rate=0.5, spa_token=16).cuda()
    aucs = []
    best_map = last_map = None
    loss_sum = 0.0

    def track_loss(module, inputs, outputs):
        nonlocal loss_sum
        if torch.is_grad_enabled():
            source, = inputs
            reconstruction = outputs[0].detach()
            loss = F.mse_loss(reconstruction, source) + 0.1 * F.l1_loss(reconstruction, source)
            loss_sum += loss.item() * len(source)

    def track_auc(labels, scores):
        nonlocal loss_sum, best_map, last_map
        auc = score_auc(scores, labels)
        last_map = scores.reshape(mask.shape).copy()
        if not aucs or auc > max(aucs):
            best_map = last_map
        aucs.append(auc)
        if collect_curve:
            loss_history.append(loss_sum / len(dataset))
            loss_sum = 0.0
        return auc

    loss_hook = model.register_forward_hook(track_loss) if collect_curve else nullcontext()
    cudnn = torch.backends.cudnn
    save_residuals = training.save_residuals

    def save_validation_map(*args):
        with validation_stage(validation, check=False):
            return save_residuals(*args)

    with TemporaryDirectory(prefix="dms2f-") as temp, loss_hook, cudnn.flags(enabled=cudnn.enabled, benchmark=True):
        temp = Path(temp)
        with patch.multiple(training, MODELS_DIR=temp, RESULTS_DIR=temp, roc_auc_score=track_auc,
                            torch=_DmsTrainingTorch(validation), save_residuals=save_validation_map):
            training.train_model(model, dataset, scene, epochs=epochs, batch_sz=32, lr=5e-4, wd=1e-4)
        if len(aucs) != epochs:
            raise RuntimeError(f"DMS2F-HAD evaluated {len(aucs)} of {epochs} epochs")
        best_epoch = max(range(epochs), key=aucs.__getitem__) + 1 if select_best else epochs
        checkpoint = "best_model_auc.pt" if select_best and max(aucs) > 0 else "last_model.pt"
        with validation_stage(validation, check=False):
            model.load_state_dict(torch.load(temp / scene / checkpoint, map_location="cuda"))

    @torch.no_grad()
    def predict():
        # The official block validation keeps the network in training mode.
        inputs = data.HSIDataset({"data": cube}, scene, block_size=16, stride=8)
        patches = torch.cat([model(batch.cuda())[0].cpu() for batch in inputs.blocks.split(32)])
        reconstructed = training.block_fold(patches, (inputs.H, inputs.W), inputs.block_size,
                                            inputs.stride, inputs.positions)
        original = torch.from_numpy(inputs.image).permute(2, 0, 1)
        return np.linalg.norm(reconstructed - original, axis=0)

    return predict, {"epochs": epochs, "best_epoch": best_epoch, "auc_last_epoch": aucs[-1],
                     "reference_scores": best_map if select_best else last_map,
                     "epoch_curve": aucs, "expected_optimizer_steps": ceil(len(dataset) / 32) * epochs}
