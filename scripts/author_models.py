"""Per-scene training adapters for separately installed Auto-AD and TDD."""

import importlib
import math
import random
import sys
from functools import cache
from types import ModuleType

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from scripts.author_had import AUTHOR_CODE
from scripts.comparison_helpers import score_auc, validation_stage


@cache
def _model_module(folder, filename):
    source = AUTHOR_CODE / folder / "models" / f"{filename}.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing {source}. See the author checkouts in README.md.")
    package = ModuleType(f"_benchmark_{folder}_models")
    package.__path__ = [str(source.parent)]
    sys.modules[package.__name__] = package
    return importlib.import_module(f"{package.__name__}.{filename}")


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autoad_predictor(cube, mask, _, *, epochs, loss_history, collect_curve, validation):
    skip = _model_module("auto_ad", "skip").skip
    _seed(0)
    bands = cube.shape[-1]
    # Preserve the original experiment's scaling and CPU allocation order.
    image = cube.transpose(2, 0, 1).astype(np.float64)
    scale = float(image.max()) or 1.0
    target = torch.from_numpy(((image - image.min()) / scale).astype(np.float32))[None].cuda()
    model = skip(bands, bands, num_channels_down=[128] * 5, num_channels_up=[128] * 5,
                 num_channels_skip=[128] * 5, pad="reflection", need_bias=True).cuda()
    noise_cpu = torch.rand(target.shape) * 0.1
    base = noise_cpu.cuda()
    noise = torch.empty_like(base)
    pixel_weight = target.new_ones(target.shape[-2:])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    @torch.inference_mode()
    def predict():
        return (model(noise_cpu.to(target.device)) - target).square().sum(dim=1)[0].cpu().numpy()

    auc_curve = []
    for step in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        error = model(base + 0.1 * noise.normal_()) - target
        if step > 1 and (step - 1) % 100 == 0:
            residual = error.detach().square().sum(dim=1)[0]
            span = residual.max() - residual.min()
            pixel_weight = (residual.max() - residual) / span.clamp_min(torch.finfo(residual.dtype).tiny)
        loss = (error * pixel_weight).square().mean()
        loss.backward()
        optimizer.step()
        if collect_curve:
            loss_history.append(loss.item())
            with validation_stage(validation):
                auc_curve.append(score_auc(predict(), mask))

    return predict, {"epochs": epochs, "best_epoch": epochs, "seed": 0, "epoch_curve": auc_curve}


class _TddPatches(Dataset):
    def __init__(self, image, width):
        self.image = image
        self.width = width
        self.spectra = image.flatten(1)
        stride = max(1, width // 2)
        axes = []
        for size in image.shape[-2:]:
            last = size - width
            starts = list(range(0, last + 1, stride))
            if starts[-1] != last:
                starts.append(last)
            axes.append(starts)
        self.starts = [(y, x) for y in axes[0] for x in axes[1]]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        y, x = self.starts[index]
        width = self.width
        patch = self.image[:, y:y + width, x:x + width].clone()
        mask = torch.zeros((1, width, width))
        height = random.randint(1, max(1, width // 6))
        span = random.randint(1, max(1, width // 6))
        top, left = random.randrange(width - height + 1), random.randrange(width - span + 1)
        samples = torch.randint(self.spectra.shape[1], (height * span,))
        patch[:, top:top + height, left:left + span] = self.spectra[:, samples].reshape(-1, height, span)
        mask[:, top:top + height, left:left + span] = 1
        return tuple(F.interpolate(tensor[None], size=(96, 96), mode="nearest")[0] for tensor in (patch, mask))


def tdd_predictor(cube, mask, _, *, epochs, loss_history, collect_curve, validation):
    model_type = _model_module("tdd_upstream", "TDD_seg_model").TDD_seg_model
    _seed(0)
    height, width, bands = cube.shape
    image = torch.from_numpy(cube.transpose(2, 0, 1).copy())
    image = (image - image.min()) / (image.max() - image.min()).clamp_min(torch.finfo(image.dtype).tiny)
    patch_width = min(24, height, width)
    inset = patch_width // 4
    tile = patch_width - 2 * inset
    patches = _TddPatches(image, patch_width)
    loader = DataLoader(patches, batch_size=1, shuffle=True)
    model = model_type(n_channels=bands).cuda()
    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": 3e-4},
        {"params": model.decoder.parameters(), "lr": 1e-3},
    ], weight_decay=5e-4)
    logits = []

    def capture(_layer, _inputs, output):
        logits.append(output)

    @torch.inference_mode()
    def predict():
        rows, cols = math.ceil(height / tile), math.ceil(width / tile)
        padding = (inset, cols * tile - width + inset, inset, rows * tile - height + inset)
        padded = F.pad(image[None], padding, mode="replicate")
        result = image.new_empty((rows * tile, cols * tile))
        for y in range(0, rows * tile, tile):
            for x in range(0, cols * tile, tile):
                patch = padded[:, :, y:y + patch_width, x:x + patch_width]
                resized = F.interpolate(patch.cuda(), size=(96, 96), mode="nearest")
                prediction, _ = model(resized)
                scores = F.interpolate(prediction[-1], size=(patch_width, patch_width), mode="nearest")
                result[y:y + tile, x:x + tile] = scores[0, 0, inset:inset + tile, inset:inset + tile].cpu()
        return result[:height, :width].numpy()

    auc_curve = []
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        hooks = [(cell.conv3 if index < 5 else cell.conv2).register_forward_hook(capture)
                 for index, cell in enumerate(model.decoder)]
        try:
            for patch, label in loader:
                optimizer.zero_grad(set_to_none=True)
                label = label.cuda()
                model(patch.cuda())
                loss = label.new_zeros(())
                for prediction, factor in zip(logits, (0.5, 0.5, 0.5, 0.8, 0.8, 1.0), strict=True):
                    target = F.interpolate(label, size=prediction.shape[-2:], mode="nearest")
                    loss = loss + factor * F.binary_cross_entropy_with_logits(prediction, target,
                                                                               weight=0.1 + 0.9 * target)
                logits.clear()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
                optimizer.step()
                if collect_curve:
                    epoch_loss += loss.item()
        finally:
            for hook in hooks:
                hook.remove()
        model.eval()
        if collect_curve:
            loss_history.append(epoch_loss / len(loader))
            with validation_stage(validation):
                auc_curve.append(score_auc(predict(), mask))

    return predict, {"epochs": epochs, "best_epoch": epochs, "seed": 0, "epoch_curve": auc_curve}
