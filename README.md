# Hyperspectral Anomaly Detection Benchmark

## Abstract

We compare nine hyperspectral anomaly detectors on 14 TIFF scenes using pixel-level ROC AUC and wall-clock runtime. AED achieves the highest mean ROC AUC (0.9754), followed by GT-HAD (0.9646) and SM-HAD (0.9491). GM-HAD reaches 0.8761 under the current 60-epoch protocol and 0.9470 in the historical 200-epoch benchmark. The runtime protocol distinguishes shared TIFF preparation, training, ROC AUC evaluation, and inference.

[Interactive report](http://127.0.0.1:8767/) · [Detailed results](results/had_comparison_paper/report.md) · [Excel workbook](results/had_comparison_paper/site/HAD_comparison.xlsx) · [Raw metrics](results/had_comparison_paper/site/results.json)

## Experimental methods

The tables below report the original nine-detector experiments. The current runner launches all nine methods with separate author checkouts. Auto-AD and TDD use per-scene adapters around the author models, and GM-HAD uses the author training program. These runs have their own protocol and results.

### Scenes and evaluation

The scene pairs consist of hyperspectral TIFF cubes and binary anomaly masks. Each method receives the same cube and spectral band selection within a scene. Preprocessing retains all bands for Hydice_Urban, selects 186 bands from 425-band AVIRIS-NG cubes, and keeps the first 186 bands of the other scenes. Positive mask pixels mark anomalies. AED and LRX were calibrated on HAD12, HAD13, San_Diego, and Hydice_Urban, which are included in the benchmark. [Scene manifest](results/had_comparison_paper/scenes.json) · [Preprocessing code](scripts/tiff_benchmark_data.py)

ROC AUC is calculated across pixels within each scene, and the reported means weight scenes equally. The current comparison uses the last completed epoch with seed 0, seed 3 for GM-HAD, and one detector run for each classical method. Models are fitted and assessed on the same scene.

Saved maps for Auto-AD, DMS2F-HAD, and TDD use an epoch selected by ROC AUC against the evaluated scene's mask. SM-HAD saves the map with the highest final-epoch AUC over seeds 0, 100, and 200. GT-HAD and GM-HAD save the final-epoch map. In the JSON, `auc` evaluates the saved map. The main comparison uses `seed_auc["0"]` for SM-HAD and `auc_last_epoch` for the other learned methods.

### Timing protocol

Runtime was measured under WSL2 on a system with an AMD Ryzen 5 5600X CPU and an NVIDIA RTX 3060 GPU. Methods ran sequentially. CUDA synchronization brackets the training and validation measurements where applicable and ensures inference work completes before a timed repetition ends.

| Measurement | Definition |
| --- | --- |
| Shared TIFF preparation | A separate per-scene wall-clock sample covers reading the TIFF cube and mask, aligning axes, binarizing the mask, selecting spectral bands, and building the CPU array. Intermediate MAT serialization and loading are outside the method timers. |
| Training | Wall-clock duration from the prepared CPU cube through method-specific input preparation, model and optimizer setup, device transfers, and optimizer updates, with the separately measured ROC AUC evaluation time deducted. SM-HAD training time covers all three seeds. |
| Mean optimizer-step time | Per scene, training time divided by the number of optimizer updates. SM-HAD counts updates across all three seeds. The reported mean averages the per-scene ratios. RX, LRX, and AED have no training or optimizer-step time. |
| ROC AUC evaluation | Separate wall-clock measurement for validation passes, transfers for evaluation, metric calculation, and checkpoint handling where applicable. The check count includes the final evaluation of the map. |
| Inference | Starts from the prepared scene cube in CPU memory with the model ready to predict. Includes method-specific input preparation, transfers, prediction, postprocessing, and returning the full score map to CPU memory. Each per-scene value is the median of five runs after one untimed warm-up. Model loading, shared TIFF preparation, ROC AUC computation, and file export are outside this latency. |

GT-HAD and SM-HAD include a preliminary reconstruction, fresh content matching (CMM), and a final reconstruction with frozen weights in every timed inference. The CMM-inclusive timings come from repeat fits with the original settings and selected seeds. The ROC AUC and saved maps come from the original runs and use the correspondences obtained during training.

GM-HAD rebuilds PCA, superpixels, regional features, and the graph inside the inference timer. Auto-AD reuses the fixed noise tensor prepared during its fit. The timer includes moving this tensor to the GPU and computing the reconstruction residual. DMS2F-HAD follows the upstream block validation in training mode with gradients disabled, so random masking is active. [Timing and evaluation code](scripts/had_comparison.py) · [Repeated inference timer](scripts/comparison_helpers.py) · [Method adapters](scripts/author_had.py)

### Training settings

The rerun uses 14 local TIFF scenes with shared spectral band preprocessing in `scripts/tiff_benchmark_data.py`. The models are trained separately on each scene using an RTX 3060 in WSL2 with PyTorch 2.7.1+cu128 and `mamba-ssm` 2.2.6.post3.

| Method | Epochs | Optimizer | Other settings and sources |
| --- | ---: | --- | --- |
| DMS2F-HAD | 100 | Adam, LR `5e-4`, weight decay `1e-4`, batch size 32 | Section 4.3 of the paper specifies a 16×16 window, stride 8, embedding dimension 64, and spectral group size 16. Batch size 32, depth 1, `spec_rate=0.5`, `spa_token=16`, `MSE + 0.1 × L1`, and seed 0 come from the authors' code. The ablation script uses 150 epochs and `spec_num=12`. [WACV paper](https://openaccess.thecvf.com/content/WACV2026/papers/Pant_DMS2F-HAD_A_Dual-branch_Mamba-based_Spatial-Spectral_Fusion_Network_for_Hyperspectral_Anomaly_WACV_2026_paper.pdf), [authors' training script](https://github.com/Ayushma00/DMS2F-HAD/blob/8546823701ef021da2a150e0bf9e9965863bbabc/main.py) |
| GT-HAD | 150 | Adam, LR `1e-3`, batch size 64 | The authors' `main.py` specifies a 9×9 window, stride 3, embedding dimension 64, MLP expansion factor 2, MSE, and CMM every 25 epochs before the final epoch. Local scenes without a matching upstream seed use seed 0. [IEEE paper](https://doi.org/10.1109/TNNLS.2024.3355166), [authors' training script](https://github.com/jeline0110/GT-HAD/blob/c118ee0d606610aae12f3fab1ef18579169f4809/dnnmethods/GT-HAD/main.py#L53-L78) |
| SM-HAD | 100 per seed | Adam, LR `1e-3`, batch size 64 | Section IV-C specifies a 15×15 window, embedding dimension 96, state dimension 64, and soft-shrinkage `0.01`. The outer stride of 5 comes from the `DatasetHsi` default. Section III-C describes 3×3 inner patches with stride 3. MLP expansion factor 2, MSE, background matching every 25 epochs, and seeds 0, 100, and 200 come from the public code. [IEEE paper](https://doi.org/10.1109/TGRS.2026.3676658), [authors' training script](https://github.com/Tanishq251/SM-HAD/blob/5fbbb4c836253a331d1528160e3d10d30f8bcda5/main.py#L61-L71) |
| GM-HAD | Up to 60 | AdamW with PCGrad, LR `5e-4`, weight decay `5e-4`, betas `(0.9, 0.99)` | Training stops after 10 epochs without a decrease greater than `1e-6` in the summed loss. The published run initializes seed 3 once before the scene loop. Hidden dimension 256, Mamba state dimension 32, three PCA components, and 1% region masking with at least one masked region. [IEEE paper](https://doi.org/10.1109/JSTARS.2026.3682454), [authors' script](https://github.com/TaoziLakeDeepSleep/GM-HAD/blob/1671f755a3b2a7b8798d1961f00d6a33b39e44ab/main.py) |
| Auto-AD | 1,001 optimizer steps | Adam, LR `1e-2` | Five levels with 128 channels per level, input noise standard deviation `0.1`, weighted MSE, mask updates every 100 steps, and seed 0. [author repository](https://github.com/RSIDEA-WHU2020/Auto-AD) |
| TDD | 30 | Adam, encoder LR `3e-4`, decoder LR `1e-3`, weight decay `5e-4`, batch size 1 | Local per-scene fitting with pseudo-anomalies, 24×24 patches, stride 12, 96×96 network input, and seed 0. [author repository](https://github.com/Jingtao-Li-CVer/TDD) |
| RX | — | — | Full-scene statistics and Mahalanobis distance with a covariance pseudoinverse. [Reed and Yu](https://doi.org/10.1109/29.60107), [local implementation](scripts/had_classical.py) |
| LRX | — | — | 5×5 inner window, 9×9 outer window, and regularization of `1e-5`. [Local implementation](scripts/had_classical.py) |
| AED | — | — | Area threshold 64 and range parameter `0.5`. [Source paper](https://ieeexplore.ieee.org/document/7994698/), [local implementation](algorithms/classical_had.py) |

## Results

### Current benchmark

Table 1. Final-epoch or single-run pixel-level ROC AUC on all 14 scenes. Bold marks the highest unrounded value in each row.

| Scene | AED | GT-HAD | SM-HAD | GM-HAD | RX | LRX | DMS2F-HAD | Auto-AD | TDD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HAD12 | **0.9842** | 0.9616 | 0.9340 | 0.8896 | 0.7753 | 0.8626 | 0.6652 | 0.8048 | 0.4152 |
| HAD13 | 0.9971 | **0.9989** | 0.9988 | 0.9740 | 0.9854 | 0.9832 | 0.9793 | 0.8066 | 0.9208 |
| HAD14 | 0.9276 | 0.6855 | 0.6877 | **0.9350** | 0.8704 | 0.9185 | 0.5734 | 0.9119 | 0.7239 |
| HAD21 | **0.9655** | 0.9590 | 0.9520 | 0.8774 | 0.9268 | 0.9646 | 0.7823 | 0.8306 | 0.4744 |
| HAD27 | 0.9841 | **0.9934** | 0.9932 | 0.9874 | 0.9779 | 0.9602 | 0.9546 | 0.8865 | 0.8125 |
| HAD29 | 0.9338 | 0.9892 | **0.9895** | 0.7244 | 0.8191 | 0.8732 | 0.9689 | 0.7525 | 0.4491 |
| HAD106 | 0.9955 | 0.9788 | 0.9764 | 0.6586 | **0.9966** | 0.9965 | 0.9727 | 0.8710 | 0.9890 |
| HAD113 | 0.9980 | 0.9998 | 0.9998 | 0.9503 | 0.9997 | **0.9999** | 0.9994 | 0.8790 | 0.8000 |
| HAD125 | 0.9642 | 0.9970 | **0.9984** | 0.9907 | 0.9894 | 0.9748 | 0.9776 | 0.8522 | 0.0496 |
| San_Diego | **0.9869** | 0.9842 | 0.9837 | 0.9164 | 0.9433 | 0.8952 | 0.7892 | 0.9374 | 0.7898 |
| Urban | 0.9957 | **0.9970** | 0.9962 | 0.9908 | 0.9912 | 0.9967 | 0.9132 | 0.9266 | 0.3149 |
| Airport | 0.9846 | **0.9913** | 0.9675 | 0.3932 | 0.9456 | 0.7936 | 0.9071 | 0.9461 | 0.6924 |
| Hydice_Urban | 0.9452 | 0.9693 | 0.8150 | 0.9937 | 0.9857 | **0.9970** | 0.9354 | 0.9844 | 0.8631 |
| Beach | 0.9935 | **0.9996** | 0.9950 | 0.9844 | 0.9812 | 0.9699 | 0.9661 | 0.9857 | 0.9313 |
| Mean | **0.9754** | 0.9646 | 0.9491 | 0.8761 | 0.9420 | 0.9419 | 0.8846 | 0.8840 | 0.6590 |

AED leads the scene-average ROC AUC. The detailed report and workbook provide the runtime breakdown and per-scene measurements. [Detailed results](results/had_comparison_paper/report.md) · [Excel workbook](results/had_comparison_paper/site/HAD_comparison.xlsx)

#### Inference latency

Table 2. Wall-clock inference latency in milliseconds for all nine methods. The mean gives equal weight to each of the 14 scenes. Bold marks the shortest unrounded latency in each row.

| Scene | AED | GT-HAD | SM-HAD | GM-HAD | RX | LRX | DMS2F-HAD | Auto-AD | TDD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HAD12 | 34.33 | 254.58 | 133.98 | 110.81 | 19.53 | 301.21 | 43.32 | **6.61** | 827.68 |
| HAD13 | 34.52 | 259.65 | 136.44 | 115.54 | 18.39 | 292.86 | 37.18 | **6.70** | 856.82 |
| HAD14 | 56.91 | 1320.31 | 557.38 | 167.97 | 35.18 | 459.10 | 90.30 | **19.78** | 1828.37 |
| HAD21 | 56.14 | 1347.17 | 569.21 | 143.55 | 32.98 | 434.60 | 95.76 | **14.35** | 1829.66 |
| HAD27 | 38.99 | 411.91 | 216.75 | 119.21 | 18.98 | 245.54 | 44.81 | **7.91** | 1112.57 |
| HAD29 | 34.48 | 259.27 | 136.78 | 109.90 | 19.31 | 289.33 | 42.57 | **6.96** | 818.20 |
| HAD106 | 34.11 | 254.83 | 134.60 | 105.02 | 19.09 | 291.00 | 36.37 | **6.58** | 824.09 |
| HAD113 | 34.06 | 258.84 | 136.70 | 119.32 | 17.00 | 294.59 | 35.36 | **6.29** | 813.56 |
| HAD125 | 34.59 | 259.09 | 137.03 | 115.23 | 17.45 | 294.90 | 41.56 | **6.27** | 820.15 |
| San_Diego | 56.25 | 1345.07 | 567.97 | 163.74 | 33.10 | 435.71 | 90.27 | **13.66** | 1848.68 |
| Urban | 57.39 | 1344.84 | 557.03 | 177.96 | 34.28 | 444.79 | 95.00 | **13.47** | 1846.34 |
| Airport | 56.54 | 1345.20 | 567.81 | 138.18 | 33.60 | 440.49 | 78.97 | **12.97** | 1834.14 |
| Hydice_Urban | 47.57 | 816.02 | 369.86 | 115.37 | 25.16 | 401.51 | 58.25 | **11.87** | 1426.03 |
| Beach | 122.53 | 6227.39 | 2555.56 | 327.21 | 72.46 | 1029.05 | 202.66 | **39.01** | 3846.74 |
| Mean | 49.89 | 1121.73 | 484.08 | 144.93 | 28.32 | 403.91 | 70.88 | **12.32** | 1466.65 |

Each scene value is the median of five timed runs after one untimed warm-up. The mean is the sum of the 14 scene medians divided by 14. The calculation uses the original values in seconds from the `Замеры` (Measurements) sheet of `HAD_comparison (3).xlsx`, multiplied by 1,000 and rounded for display.

Auto-AD has the shortest latency on all 14 scenes and averages 12.32 ms per scene. RX averages 28.32 ms per scene, the shortest among the classical methods.

### Historical benchmark with extended training budgets

Table 3. Historical and current training budgets and mean ROC AUC for all nine methods. The 200-epoch configuration applies to SM-HAD, DMS2F-HAD, and GM-HAD. Historical values come from the `ROC AUC` sheet of the 200-epoch benchmark workbook.

| Method | Historical budget | Historical mean ROC AUC | Current budget | Current mean ROC AUC |
| --- | --- | ---: | --- | ---: |
| AED | No training | 0.9754 | No training | 0.9754 |
| GT-HAD | 150 epochs | 0.9597 | 150 epochs | 0.9646 |
| SM-HAD | 200 epochs | 0.9864 | 100 epochs per seed | 0.9491 |
| GM-HAD | 200 epochs | 0.9470 | 60 epochs | 0.8761 |
| RX | No training | 0.9420 | No training | 0.9420 |
| LRX | No training | 0.9419 | No training | 0.9419 |
| DMS2F-HAD | 200 epochs | 0.9384 | 100 epochs | 0.8846 |
| Auto-AD | 1,001 optimizer steps | 0.8768 | 1,001 optimizer steps | 0.8840 |
| TDD | 30 epochs | 0.3542 | 30 epochs | 0.6590 |

Table 4. Historical final-epoch ROC AUC from the same workbook. RX, LRX, and AED use the single-run values. Means are computed from the original cell values.

| Scene | AED | GT-HAD | SM-HAD | GM-HAD | RX | LRX | DMS2F-HAD | Auto-AD | TDD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HAD12 | **0.9842** | 0.9527 | 0.9517 | 0.8533 | 0.7753 | 0.8626 | 0.9166 | 0.6924 | 0.1185 |
| HAD13 | 0.9971 | **0.9989** | 0.9988 | 0.9907 | 0.9854 | 0.9832 | 0.9960 | 0.7941 | 0.0023 |
| HAD14 | 0.9276 | 0.6628 | 0.9291 | **0.9535** | 0.8704 | 0.9185 | 0.8735 | 0.8485 | 0.3924 |
| HAD21 | 0.9655 | 0.9590 | **0.9853** | 0.9659 | 0.9268 | 0.9646 | 0.9459 | 0.8855 | 0.4453 |
| HAD27 | 0.9841 | 0.9934 | 0.9905 | 0.9871 | 0.9779 | 0.9602 | **0.9973** | 0.7909 | 0.0598 |
| HAD29 | 0.9338 | 0.9893 | **0.9932** | 0.8469 | 0.8191 | 0.8732 | 0.9003 | 0.8327 | 0.4671 |
| HAD106 | 0.9955 | 0.9784 | 0.9945 | 0.8774 | **0.9966** | 0.9965 | 0.9717 | 0.9150 | 0.0191 |
| HAD113 | 0.9980 | 0.9998 | 0.9998 | 0.9940 | 0.9997 | **0.9999** | 0.9997 | 0.9760 | 0.1925 |
| HAD125 | 0.9642 | 0.9970 | **0.9997** | 0.9932 | 0.9894 | 0.9748 | 0.9958 | 0.8240 | 0.9651 |
| San_Diego | 0.9869 | 0.9819 | **0.9956** | 0.8607 | 0.9433 | 0.8952 | 0.8147 | 0.9159 | 0.7379 |
| Urban | 0.9957 | **0.9971** | 0.9957 | 0.9927 | 0.9912 | 0.9967 | 0.9763 | 0.9445 | 0.1851 |
| Airport | 0.9846 | **0.9915** | 0.9841 | 0.9614 | 0.9456 | 0.7936 | 0.8637 | 0.9041 | 0.3597 |
| Hydice_Urban | 0.9452 | 0.9404 | 0.9926 | 0.9945 | 0.9857 | **0.9970** | 0.9429 | 0.9801 | 0.9595 |
| Beach | 0.9935 | 0.9942 | **0.9996** | 0.9867 | 0.9812 | 0.9699 | 0.9431 | 0.9721 | 0.0544 |
| Mean | 0.9754 | 0.9597 | **0.9864** | 0.9470 | 0.9420 | 0.9419 | 0.9384 | 0.8768 | 0.3542 |

GM-HAD reaches **0.9470** at 200 epochs in the historical workbook and has higher final-epoch AUC than the current run on 11 of 14 scenes. Its mean over checkpoints selected by the scene labels is 0.9762. The earlier 200-epoch run in this workspace achieved **0.9468** at the final epoch with seed 0. The current GM-HAD run reaches **0.8761** at 60 epochs with seed 3. Early stopping after ten epochs without a loss improvement was enabled, and all 14 scenes reached the limit.

Historical final-epoch ROC AUC is also higher for SM-HAD (0.9864 at 200 epochs) and DMS2F-HAD (0.9384 at 200 epochs). These are comparisons between benchmark runs. GM-HAD changes both its seed and training budget, and the historical SM-HAD and DMS2F-HAD runs use earlier local implementations. The differences therefore cannot be attributed to the training budget alone. The historical workbook also predates the current inference timing protocol.

## Reproducibility

### Serve the report

Start the server from the repository root. Viewing the report requires only Python and its standard library.

```bash
python -m http.server 8767 --bind 127.0.0.1 --directory results/had_comparison_paper/site
```

The report shows ROC AUC, mean scores, per-scene training and inference times, average training step time, training settings, and source links for all nine methods across 14 scenes. Select a scene name in either table to view its maps and detailed timings. Select an anomaly map to download its per-pixel scores as a NumPy array. The Excel button is available when a workbook matches the displayed results. [Open the report](http://127.0.0.1:8767/), [Per-scene timings](http://127.0.0.1:8767/#timing)

### Prepare the data

Place each TIFF cube and its ground-truth mask in the same directory. For example, `01_HAD12.tif` and `01_HAD12_gt.tiff` form the `HAD12` scene. Both `.tif` and `.tiff` extensions are supported. The cube must have two spatial axes, Y and X, and one spectral axis. The mask must have shape Y × X, with positive values marking anomalies.

Preprocessing keeps all bands for scenes whose names start with `Hydice`, selects 186 bands from 425-band AVIRIS-NG cubes, and takes the first 186 bands for all other scenes. The scene manifest lists the source filenames, dimensions, and band indices. Rerunning the benchmark requires the original TIFF files. [Scene manifest](results/had_comparison_paper/scenes.json)

### Run the benchmark

RX, LRX, and AED can run on a CPU with Python 3.10 or later.

```bash
python -m pip install -r requirements.txt
python -m scripts.had_comparison --datasets HAD12 --methods rx lrx aed
```

The script searches `HAD_Data` by default and falls back to `data/HAD_Data`. Use `--data-dir` to specify another TIFF directory.

To run all nine methods, use Linux or WSL2 with an NVIDIA GPU. The commands below use Python 3.12, PyTorch 2.7.1, and CUDA 12.8. Building CUDA extensions from source requires the CUDA Toolkit with `nvcc`. Use a CUDA Toolkit that matches the PyTorch CUDA build, with its `bin` directory on `PATH`. Clone the six author repositories at the revisions listed below. [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/), [Mamba release](https://github.com/state-spaces/mamba/releases/tag/v2.2.6.post3)

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-gpu.txt packaging setuptools wheel ninja
python -m pip install --no-build-isolation causal-conv1d==1.5.4 mamba-ssm==2.2.6.post3

# One method on one scene
python -m scripts.had_comparison --datasets HAD12 --methods gthad

# All nine methods on all available scenes
python -m scripts.had_comparison
```

Add `--datasets HAD12` to run all nine methods on one scene. Use `--methods` to select individual detectors and `--output-dir` for a separate run. Subsequent runs skip completed results. Add `--rerun` to recompute them. The `--learning-curves` option applies to Auto-AD, GT-HAD, DMS2F-HAD, SM-HAD, and TDD.

The default report directory is `results/had_comparison_paper/site`. Each anomaly map is saved alongside `scores.npy` and `metrics.json`.

## Code and provenance

Clone the author repositories into the paths below and check out the listed revisions. Git ignores all six checkout paths.

| Method | Source used by the benchmark | Upstream revision |
| --- | --- | --- |
| DMS2F-HAD | `algorithms/dms2f_had` | [`8546823`](https://github.com/Ayushma00/DMS2F-HAD/tree/8546823701ef021da2a150e0bf9e9965863bbabc) |
| GT-HAD | `algorithms/gt_had` | [`c118ee0`](https://github.com/jeline0110/GT-HAD/tree/c118ee0d606610aae12f3fab1ef18579169f4809) |
| SM-HAD | `algorithms/sm_had` | [`5fbbb4c`](https://github.com/Tanishq251/SM-HAD/tree/5fbbb4c836253a331d1528160e3d10d30f8bcda5) |
| Auto-AD | `algorithms/auto_ad` | [`149d552`](https://github.com/RSIDEA-WHU2020/Auto-AD/tree/149d5526db595bf058ca5a7aae808af3ad7abf54) |
| GM-HAD | `algorithms/gm_had` | [`1671f75`](https://github.com/TaoziLakeDeepSleep/GM-HAD/tree/1671f755a3b2a7b8798d1961f00d6a33b39e44ab) |
| TDD | `algorithms/tdd_upstream` | [`8d1f1ec`](https://github.com/Jingtao-Li-CVer/TDD/tree/8d1f1ec99349a9e659383328e00ba8c8f2208cea) |

Run the following commands once from the benchmark root.

```bash
git clone https://github.com/RSIDEA-WHU2020/Auto-AD.git algorithms/auto_ad
git -C algorithms/auto_ad checkout --detach 149d5526db595bf058ca5a7aae808af3ad7abf54
git clone https://github.com/jeline0110/GT-HAD.git algorithms/gt_had
git -C algorithms/gt_had checkout --detach c118ee0d606610aae12f3fab1ef18579169f4809
git clone https://github.com/Ayushma00/DMS2F-HAD.git algorithms/dms2f_had
git -C algorithms/dms2f_had checkout --detach 8546823701ef021da2a150e0bf9e9965863bbabc
git clone https://github.com/Tanishq251/SM-HAD.git algorithms/sm_had
git -C algorithms/sm_had checkout --detach 5fbbb4c836253a331d1528160e3d10d30f8bcda5
git clone https://github.com/TaoziLakeDeepSleep/GM-HAD.git algorithms/gm_had
git -C algorithms/gm_had checkout --detach 1671f755a3b2a7b8798d1961f00d6a33b39e44ab
git clone https://github.com/Jingtao-Li-CVer/TDD.git algorithms/tdd_upstream
git -C algorithms/tdd_upstream checkout --detach 8d1f1ec99349a9e659383328e00ba8c8f2208cea
```

Then run `python -m scripts.had_comparison` to process all available scenes, or append `--datasets HAD12` to check a single scene.

- Auto-AD runs 1,001 optimizer steps on the authors' skip network and evaluates its final reconstruction.
- TDD trains the authors' model on scene-local pseudo-anomaly patches for 30 epochs and evaluates the final model. This adapter uses its own patch sampling and tiling.
- GM-HAD invokes the authors' program on each prepared MAT scene. Its `author_elapsed_s` includes subprocess startup and may include first-time CUDA extension compilation. The common training and inference timing columns are blank for this method.

Run all nine detectors on HAD12 with deterministic CUDA settings:

```bash
python -m scripts.had_comparison --datasets HAD12 --deterministic-cuda
```

Omit `--datasets HAD12` to run all scenes. The flag is off by default. With the flag, the runner requires deterministic PyTorch operations, disables TF32 and cuDNN benchmarking, and uses the same settings in the GM-HAD subprocess. In this mode, GM-HAD computes segment medians by sorting and selecting the lower middle value. The default output directory is `results/had_comparison_deterministic`. [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.7/notes/randomness.html)

In two runs on HAD12 with the flag, five methods produced identical score maps. DMS2F-HAD, SM-HAD, and GM-HAD showed small differences in final ROC AUC. Their upstream CUDA extensions use custom kernels outside the scope of PyTorch's determinism checks. TDD's final ROC AUC was 0.5090 and 0.7261 in the two runs. [Mamba CUDA backward kernel](https://github.com/state-spaces/mamba/blob/v2.2.6.post3/csrc/selective_scan/selective_scan_bwd_kernel.cuh)

In the default mode, Auto-AD and TDD use fixed seeds, but some CUDA operations can still produce different gradients between runs. Auto-AD uses reflection padding, and TDD uses bilinear resizing. The GPU, PyTorch and CUDA versions, and TF32 settings can affect final ROC AUC.

The optional `--author-scores-dir` still accepts existing maps for Auto-AD, GM-HAD, and TDD. Each map goes in `<scene>/<method>/scores.npy`. The importer also accepts the native `scores.mat` keys `detection` and `det` for Auto-AD and GM-HAD respectively, and `scores.tif` for TDD.

#### Integrated adapters

The benchmark integrates DMS2F-HAD, GT-HAD, and SM-HAD through local adapters.

- DMS2F-HAD calls the upstream `train_model`. The adapter supplies the path settings and the unused ablation tracker import missing from that revision. The selected epoch's map and the final epoch's AUC come from the upstream validation routine. In that routine, the network stays in training mode and applies random masking.
- GT-HAD and SM-HAD use the authors' `Net`, `DatasetHsi`, and background matching code. The adapter reproduces the training loop for the local scenes and assembles patches on the padded grid before cropping. This keeps the anomaly map aligned with the ground truth when padding is asymmetric. The timed inference rebuilds the matches, and the ROC AUC evaluation uses the matches obtained during training. [Adapter](scripts/author_had.py)
- With a 15×15 SM-HAD window, the published model's inner layer receives a 5×5 grid. The adapter computes 25 tokens and creates a diagonal mask of the corresponding size. The public `main.py` runs a configuration with a 9×9 window and embedding dimension 64.

| Path | Contents |
| --- | --- |
| `scripts/had_comparison.py` | Method execution, timing, and export of metrics and maps |
| `scripts/tiff_benchmark_data.py` | TIFF loading, spectral band selection, and scene preprocessing |
| `algorithms/classical_had.py` | Benchmark implementation of AED |
| `utils/epoch_selection.py` | Scene checkpoint selection |
| `docs/index.html` | Comparison report template |

The baseline comparison code is adapted from the repository accompanying the survey by Pant et al. [Survey preprint](https://arxiv.org/abs/2507.05730) · [Survey repository](https://github.com/Ayushma00/Hyperspectral-Anomaly-Detection-Survey)

### Implementations and papers

| Method | Published paper | Code repository |
| --- | --- | --- |
| RX | Reed and Yu, [Adaptive Multiple-Band CFAR Detection of an Optical Pattern with Unknown Spectral Distribution](https://doi.org/10.1109/29.60107) | [Spectral Python implementation](https://github.com/spectralpython/spectral) · [Benchmark code](scripts/had_classical.py) |
| LRX | Molero et al., [Analysis and Optimizations of Global and Local Versions of the RX Algorithm for Anomaly Detection in Hyperspectral Data](https://doi.org/10.1109/JSTARS.2013.2238609) | [MATLAB implementation](https://github.com/lzw-lzw/Hyperspectral-anomaly-detection-methods/tree/master/LRX) · [Benchmark code](scripts/had_classical.py) |
| AED | Kang et al., [Hyperspectral Anomaly Detection With Attribute and Edge-Preserving Filters](https://doi.org/10.1109/TGRS.2017.2710145) | [Third-party AED implementation](https://github.com/yousefan/AED-algorithm) · [Benchmark code](algorithms/classical_had.py) |
| Auto-AD | Wang et al., [Auto-AD: Autonomous Hyperspectral Anomaly Detection Network Based on Fully Convolutional Autoencoder](https://doi.org/10.1109/TGRS.2021.3057721) | [RSIDEA-WHU2020/Auto-AD](https://github.com/RSIDEA-WHU2020/Auto-AD) |
| GT-HAD | [Gated Transformer for Hyperspectral Anomaly Detection](https://doi.org/10.1109/TNNLS.2024.3355166) | [jeline0110/GT-HAD](https://github.com/jeline0110/GT-HAD) |
| DMS2F-HAD | Pant et al., [DMS2F-HAD: A Dual-branch Mamba-based Spatial–Spectral Fusion Network for Hyperspectral Anomaly Detection](https://doi.org/10.1109/WACV61042.2026.00545) · [CVF open-access PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Pant_DMS2F-HAD_A_Dual-branch_Mamba-based_Spatial-Spectral_Fusion_Network_for_Hyperspectral_Anomaly_WACV_2026_paper.pdf) | [Ayushma00/DMS2F-HAD](https://github.com/Ayushma00/DMS2F-HAD) |
| SM-HAD | Rachamalla et al., [SM-HAD: Spectrum Mamba for Hyperspectral Anomaly Detection](https://doi.org/10.1109/TGRS.2026.3676658) | [Tanishq251/SM-HAD](https://github.com/Tanishq251/SM-HAD) |
| GM-HAD | Tang et al., [Graph Mamba for Fast Hyperspectral Anomaly Detection](https://doi.org/10.1109/JSTARS.2026.3682454) | [TaoziLakeDeepSleep/GM-HAD](https://github.com/TaoziLakeDeepSleep/GM-HAD) |
| TDD | Li et al., [One-Step Detection Paradigm for Hyperspectral Anomaly Detection via Spectral Deviation Relationship Learning](https://doi.org/10.1109/TGRS.2024.3392189) | [Jingtao-Li-CVer/TDD](https://github.com/Jingtao-Li-CVer/TDD) |

### Licenses

The benchmark code is available under the MIT license. Separately downloaded implementations are subject to their respective authors' terms. Auto-AD is provided for academic use. [Benchmark license](LICENSE) ? [Auto-AD terms](https://github.com/RSIDEA-WHU2020/Auto-AD/blob/main/LICENSE)

TIFF scenes and ground-truth masks are local inputs. Git ignores these files.
