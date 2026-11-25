"""Compare nine hyperspectral anomaly detectors and save full-resolution maps."""
import argparse
import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(ROOT))

from scripts.comparison_helpers import ValidationTiming, atomic, enable_deterministic_cuda, measure_model, score_auc
from scripts.tiff_benchmark_data import load_prepared_scene, prepare_tiff_data

METHODS = ("autoad", "gthad", "dms2f", "rx", "aed", "lrx", "smhad", "gmhad", "tdd")
EPOCH_COUNTS = {"autoad": 1001, "gthad": 150, "dms2f": 100, "smhad": 100, "tdd": 30}
CMM_METHODS = {"gthad", "smhad"}
AUTHOR_OUTPUTS = {"autoad": (".mat", "detection"), "gmhad": (".mat", "det"), "tdd": (".tif", None)}
AUTHOR_ENTRIES = {
    "autoad": "auto_ad/models/skip.py",
    "gthad": "gt_had/dnnmethods/GT-HAD/net.py",
    "dms2f": "dms2f_had/src/train.py",
    "smhad": "sm_had/net.py",
    "gmhad": "gm_had/main.py",
    "tdd": "tdd_upstream/models/TDD_seg_model.py",
}
TIMING_PROTOCOL = 3


def save_ground_truth(mask, path):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    Image.fromarray((mask > 0).astype("uint8") * 255).save(temporary, format="PNG")
    temporary.replace(path)


def save_anomaly_map(scores, path):
    import numpy as np
    from matplotlib import colormaps
    from PIL import Image

    low, high = np.percentile(scores, [1, 99])
    scaled = np.clip((scores - low) / max(high - low, 1e-12), 0, 1)
    temporary = path.with_suffix(".tmp")
    Image.fromarray(colormaps["jet"](scaled, bytes=True)).save(temporary, format="PNG")
    temporary.replace(path)


def import_author_scores(scores_dir, method, scene, mask, output):
    import numpy as np

    native_extension, variable = AUTHOR_OUTPUTS[method]
    source = scores_dir / scene / method / f"scores{native_extension}"
    saved_array = source.with_suffix(".npy")
    if saved_array.is_file():
        scores = np.load(saved_array, allow_pickle=False)
    elif not source.is_file():
        raise FileNotFoundError(f"Missing {saved_array} or {source}. See the author setup in README.md.")
    elif variable:
        from scipy.io import loadmat

        scores = loadmat(source, variable_names=(variable,))[variable]
    else:
        from tifffile import imread

        scores = imread(source)
    return {"auc": save_external_map(scores, method, scene, mask, output), "result_source": "author_scores"}


def save_external_map(scores, method, scene, mask, output):
    import numpy as np

    if scores.shape != mask.shape or scores.dtype.kind not in "iuf" or not np.isfinite(scores).all():
        raise ValueError(f"{scene}/{method}: expected a real, finite score map of shape {mask.shape}, got {scores.shape}")
    auc = score_auc(scores, mask)
    np.save(output / "scores.npy", scores)
    save_anomaly_map(scores, output / "map.png")
    return auc


def run_gmhad(scene, mask, prepared, output, *, deterministic_cuda):
    from scipy.io import loadmat

    source = ROOT / "algorithms/gm_had"
    command = [sys.executable, "-m", "scripts.gmhad_entry", str(source), str(prepared), str(output)]
    if deterministic_cuda:
        command.append("--deterministic-cuda")
    started = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - started
    scores = loadmat(output / f"{scene}.mat", variable_names=("det",))["det"]
    auc = save_external_map(scores, "gmhad", scene, mask, output)
    return {"auc": auc, "auc_last_epoch": auc, "result_source": "author_run",
            "author_elapsed_s": elapsed, "seed": 3, "epoch_limit": 60}


def evaluate(method, scene, folder, cube, mask, *, final_epoch=None, record_curve=False):
    import numpy as np

    if method not in EPOCH_COUNTS:
        from scripts.had_classical import predict

        scores, elapsed = measure_model(lambda: predict(method, cube), lambda: None)
        started = time.perf_counter()
        auc = score_auc(scores, mask)
        validation_s = time.perf_counter() - started
        np.save(folder / "scores.npy", scores)
        save_anomaly_map(scores, folder / "map.png")
        return {"infer_s": elapsed, "auc": auc, "validation_s": validation_s, "auc_checks": 1}

    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook

    if not torch.cuda.is_available():
        raise RuntimeError(f"{method} requires a CUDA-enabled PyTorch installation and NVIDIA GPU")

    def synchronize():
        torch.cuda.synchronize()

    def begin_training():
        synchronize()
        return time.perf_counter()

    validation = ValidationTiming(synchronize)
    metadata = {}
    loss_history = [] if final_epoch is None else None
    steps = 0

    def count(*args):
        nonlocal steps
        steps += 1

    def measure(predict, labels, best_epoch=None, reference_scores=None, timing_predict=None):
        nonlocal scores
        if record_curve:
            return metadata
        if final_epoch is not None:
            metadata["auc"] = score_auc(predict() if reference_scores is None else reference_scores, labels)
            return metadata
        synchronize()
        if best_epoch is not None:
            metadata.update(train_s=time.perf_counter() - start - validation.seconds, best_epoch=best_epoch)
        scores, duration = measure_model(timing_predict or predict, synchronize)
        with validation.track():
            if reference_scores is not None:
                scores = reference_scores
            elif timing_predict is not None:
                scores = predict()
            auc = score_auc(scores, labels)
        metadata.update(infer_s=duration, auc=auc, validation_s=validation.seconds, auc_checks=validation.checks)
        return metadata

    scores = None
    adapter = None
    if method in {"autoad", "tdd"}:
        from scripts.author_models import autoad_predictor, tdd_predictor

        adapter = autoad_predictor if method == "autoad" else tdd_predictor
    with register_optimizer_step_post_hook(count):
        start = begin_training()
        opts = {"epochs": final_epoch or EPOCH_COUNTS[method], "loss_history": loss_history,
                "collect_curve": record_curve, "validation": validation}
        if adapter is not None:
            predict, details = adapter(cube, mask, scene, **opts)
        elif method == "dms2f":
            from scripts.author_had import dms2f_predictor

            predict, details = dms2f_predictor(cube, mask, scene, select_best=final_epoch is None, **opts)
            expected_steps = details.pop("expected_optimizer_steps")
            if steps != expected_steps:
                raise RuntimeError(f"DMS2F-HAD completed {steps} of {expected_steps} optimizer steps")
        else:
            from scripts.author_had import transformer_predictor

            predict, details = transformer_predictor(method, cube, mask, scene, **opts)
        if method in CMM_METHODS:
            measure(predict, mask, details["best_epoch"], timing_predict=lambda: predict(recompute_cmm=True))
            details["inference_recomputes_cmm"] = True
        else:
            measure(predict, mask, details["best_epoch"], reference_scores=details.pop("reference_scores", None))
        if adapter is not None and not record_curve:
            details["auc_last_epoch"] = metadata["auc"]
        metadata.update(details)
    if record_curve:
        return {"train_loss": loss_history, "roc_auc": metadata["epoch_curve"], "seed": metadata.get("seed", 0)}
    if final_epoch is not None:
        return metadata
    if loss_history and "epoch_curve" in metadata:
        metadata["learning_curve"] = {"train_loss": loss_history, "roc_auc": metadata.pop("epoch_curve"),
                                      "seed": metadata.get("seed", 0)}
    else:
        metadata.pop("epoch_curve", None)
    if steps:
        metadata.update(optimizer_steps=steps, amortized_iter_s=metadata["train_s"] / steps)
    np.save(folder / "scores.npy", scores)
    save_anomaly_map(scores, folder / "map.png")
    return metadata


def save_learning_curve(scene, method, curve, folder, epochs):
    from math import isfinite
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    losses, auc = curve["train_loss"], curve["roc_auc"]
    if (len(losses) != epochs or len(auc) != epochs
            or not all(isfinite(v) and v >= 0 for v in losses)
            or not all(isfinite(v) and 0 <= v <= 1 for v in auc)):
        raise ValueError(f"{scene}/{method}: incomplete or invalid learning curve")

    best_epoch = max(range(epochs), key=auc.__getitem__) + 1
    unit = "step" if method == "autoad" else "epoch"
    names = {"autoad": "Auto-AD", "gthad": "GT-HAD", "dms2f": "DMS2F-HAD",
             "smhad": "SM-HAD", "gmhad": "GM-HAD", "tdd": "TDD"}
    fig = Figure(figsize=(9, 6), layout="constrained")
    FigureCanvasAgg(fig)
    loss_axis, auc_axis = fig.subplots(2, 1, sharex=True)
    fig.suptitle(f"{scene} · {names[method]}", fontweight="bold")
    x = range(1, epochs + 1)
    loss_axis.plot(x, losses, color="#245c91", linewidth=1.25)
    loss_axis.set_ylabel("Training loss")
    if min(losses) > 0 and max(losses) / min(losses) >= 100:
        loss_axis.set_yscale("log")
    auc_axis.plot(x, auc, color="#237358", linewidth=1.25)
    auc_axis.axvline(best_epoch, color="#353535", linestyle="--", linewidth=1)
    auc_axis.scatter([best_epoch], [auc[best_epoch - 1]], color="#222222", zorder=3,
                     label=f"Peak {auc[best_epoch - 1]:.4f} at {unit} {best_epoch}")
    auc_axis.scatter([epochs], [auc[-1]], marker="s", facecolors="none", edgecolors="#237358", zorder=4,
                     label=f"Final {unit} {auc[-1]:.4f}")
    auc_axis.set(xlabel="Optimizer step" if unit == "step" else "Epoch", ylabel="ROC AUC")
    auc_axis.legend(loc="best", frameon=False)
    for axis in (loss_axis, auc_axis):
        axis.grid(True, color="#e5e7eb", linewidth=0.7)
    pending = folder / "learning_curve.tmp"
    fig.savefig(pending, format="png", dpi=150)
    pending.replace(folder / "learning_curve.png")
    atomic(folder / "learning_curve.json", json.dumps(
        {"scene": scene, "method": method, "seed": 0, "epochs": epochs,
         "best_epoch": best_epoch, **curve}, allow_nan=False))


def collect_learning_curves(scenes, methods, output, rerun, *, deterministic_cuda):
    import torch

    torch.set_num_threads(2)
    failures = 0
    for scene in scenes:
        cube, mask = load_prepared_scene(output / "data", scene)
        for method in methods:
            folder = output / "site" / scene / method
            path = folder / "learning_curve.json"
            epochs = EPOCH_COUNTS[method]
            if not rerun and path.exists() and (folder / "learning_curve.png").exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if (saved.get("epoch_limit", saved["epochs"]) == epochs
                        and saved.get("deterministic_cuda", False) == deterministic_cuda):
                    continue
            print(scene, method, f"learning curve ({epochs})", flush=True)
            try:
                folder.mkdir(parents=True, exist_ok=True)
                curve = evaluate(method, scene, folder, cube, mask, record_curve=True)
                curve["deterministic_cuda"] = deterministic_cuda
                save_learning_curve(scene, method, curve, folder, epochs)
            except Exception:
                failures += 1
                traceback.print_exc()
    if failures:
        raise SystemExit(f"{failures} learning curves failed")


def publish_results(public):
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(public.glob("*/*/metrics.json"))
            if path.parent.name in METHODS]
    atomic(public / "results.json", json.dumps(rows, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--prepare-only", action="store_true", help="Export prepared MAT scenes for external detectors.")
    parser.add_argument("--author-scores-dir", type=Path, help="Directory containing Auto-AD, GM-HAD, or TDD score maps.")
    parser.add_argument("--rerun", action="store_true", help="Refresh completed results for the selected methods.")
    parser.add_argument("--deterministic-cuda", action="store_true",
                        help="Require deterministic PyTorch operations and disable CUDA TF32.")
    parser.add_argument("--learning-curves", action="store_true", help="Save training histories and plots.")
    parser.add_argument("--datasets", nargs="+", help="Run the selected scenes.")
    default_data = ROOT / "HAD_Data"
    parser.add_argument("--data-dir", type=Path, default=default_data if default_data.is_dir() else ROOT / "data/HAD_Data",
                        help="Directory containing TIFF scenes and their *_gt TIFF masks.")
    parser.add_argument("--output-dir", type=Path,
                        help="Directory for prepared scenes, scores, and the generated dashboard.")
    args = parser.parse_args()
    args.methods = args.methods or (tuple(EPOCH_COUNTS) if args.learning_curves else METHODS)
    imported = set(args.methods) & AUTHOR_OUTPUTS.keys() if args.author_scores_dir else set()
    if args.learning_curves and (imported or "gmhad" in args.methods):
        parser.error("Learning curves require an integrated training adapter.")
    if not args.prepare_only:
        missing = [str(ROOT / "algorithms" / AUTHOR_ENTRIES[method])
                   for method in args.methods if method in AUTHOR_ENTRIES and method not in imported
                   and not (ROOT / "algorithms" / AUTHOR_ENTRIES[method]).is_file()]
        if missing:
            parser.error("Install the author checkouts from README.md. Missing:\n" + "\n".join(missing))
    default_output = "had_comparison_deterministic" if args.deterministic_cuda else "had_comparison_paper"
    output = (args.output_dir or ROOT / "results" / default_output).resolve()
    data_dir = output / "data"
    public = output / "site"
    if args.learning_curves:
        methods = [method for method in args.methods if method in EPOCH_COUNTS]
        if not methods:
            parser.error("learning curves require a trained method")
    if not args.data_dir.is_dir():
        parser.error(f"TIFF directory does not exist: {args.data_dir}")
    try:
        scenes = prepare_tiff_data(args.data_dir, data_dir, args.datasets)
    except ValueError as exc:
        parser.error(str(exc))
    if args.prepare_only:
        return
    if args.deterministic_cuda and (set(args.methods) & AUTHOR_ENTRIES.keys()) - imported:
        enable_deterministic_cuda()
    if args.learning_curves:
        collect_learning_curves(scenes, methods, output, args.rerun, deterministic_cuda=args.deterministic_cuda)
        return
    public.mkdir(exist_ok=True)
    scene_metadata = json.loads((output / "scenes.json").read_text(encoding="utf-8"))
    atomic(public / "index.html", (ROOT / "docs/index.html").read_text(encoding="utf-8"))
    publish_results(public)
    failures = 0
    for scene in scenes:
        cube, mask = load_prepared_scene(data_dir, scene)
        save_ground_truth(mask, public / scene / "ground_truth.png")
        for method in args.methods:
            folder = public / scene / method
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "metrics.json"
            previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            stale_budget = method in EPOCH_COUNTS and previous.get("epochs") != EPOCH_COUNTS[method]
            deterministic_cuda = args.deterministic_cuda and method in AUTHOR_ENTRIES and method not in imported
            stale_cuda = previous.get("deterministic_cuda", False) != deterministic_cuda
            if (method not in imported and previous.get("result_source") != "author_scores"
                    and "auc" in previous and "error" not in previous
                    and not (stale_budget or stale_cuda or args.rerun)
                    and (method not in CMM_METHODS or previous.get("inference_recomputes_cmm") is True)
                    and previous.get("timing_protocol") == TIMING_PROTOCOL):
                if method not in EPOCH_COUNTS or previous.get("auc_last_epoch") is not None:
                    continue
                final_epoch = previous.get("epochs", previous.get("training_steps", EPOCH_COUNTS[method]))
                print(scene, method, "last epoch", final_epoch, flush=True)
                try:
                    previous["auc_last_epoch"] = (
                        previous["auc"] if previous["best_epoch"] == final_epoch
                        else evaluate(method, scene, folder, cube, mask, final_epoch=final_epoch)["auc"]
                    )
                    previous.pop("last_epoch_error", None)
                except Exception as exc:
                    failures += 1
                    traceback.print_exc()
                    previous["last_epoch_error"] = str(exc)
                atomic(path, json.dumps(previous))
                publish_results(public)
                continue
            print(scene, method, flush=True)
            try:
                with tempfile.TemporaryDirectory(prefix="fit-", dir=folder) as pending:
                    pending = Path(pending)
                    if method in imported:
                        metadata = import_author_scores(args.author_scores_dir, method, scene, mask, pending)
                    elif method == "gmhad":
                        metadata = run_gmhad(scene, mask, data_dir / f"{scene}.mat", pending,
                                             deterministic_cuda=deterministic_cuda)
                    else:
                        metadata = evaluate(method, scene, pending, cube, mask)
                    if method in AUTHOR_ENTRIES and method not in imported:
                        metadata["deterministic_cuda"] = deterministic_cuda
                    artifacts = ["scores.npy", "map.png"]
                    curve = metadata.pop("learning_curve", None)
                    if curve is not None:
                        curve["deterministic_cuda"] = deterministic_cuda
                        epochs = metadata.get("epochs", metadata.get("training_steps", EPOCH_COUNTS[method]))
                        save_learning_curve(scene, method, curve, pending, epochs)
                        artifacts.extend(["learning_curve.png", "learning_curve.json"])
                    for artifact in artifacts:
                        (pending / artifact).replace(folder / artifact)
            except Exception as exc:
                failures += 1
                traceback.print_exc()
                metadata = {**previous, "error": str(exc)}
            atomic(path, json.dumps({**metadata, "scene": scene, "method": method,
                                     "scene_prep_s": scene_metadata[scene]["prep_s"],
                                     "timing_protocol": TIMING_PROTOCOL}))
            publish_results(public)
    if failures:
        raise SystemExit(f"{failures} comparisons failed. See the dashboard or metrics.json for details.")


if __name__ == "__main__":
    main()
