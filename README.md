# matmatch2real

`matmatch2real` is a research-oriented codebase for teacher-generated pseudo labels, soft distillation, and hard-distill experiments built on the official Ultralytics training and validation interfaces.

The repository currently covers three main workflows:

1. Teacher inference and pseudo-label export
2. Soft distillation training with teacher supervision
3. Hard distill and few-shot experiments with Ultralytics YOLO

This README is written for two use cases:

1. Continuing local experiments
2. Publishing the project on GitHub with enough context for others to understand the structure and entry points

## License

This repository is released under the MIT license. See [LICENSE](LICENSE).

## Citation

Citation metadata is available in [CITATION.cff](CITATION.cff).

If you use this repository in academic work, cite it as:

```bibtex
@software{zhang2026matmatch2real,
  author = {Zhang, Skyler},
  title = {matmatch2real},
  year = {2026},
  version = {0.1.0},
  license = {MIT},
  note = {Research code for teacher-generated pseudo labels, soft distillation, and hard-distill experiments}
}
```

## 1. Repository Layout

```text
matmatch2real/
  configs/
    distill/           Soft-distill configs
    hard_distill/      Hard-distill configs
    teacher/           Teacher configs
    yolo/              Standalone YOLO configs
  debug/
    hard_distill/      Hard-distill debug and profiling scripts
    teacher/           Teacher-side debug and visualization scripts
  scripts/
    export_pseudolabels.py
    eval_pseudolabels.py
    train_soft_distill.py
    run_hard_distill.py
    build_fewshot_split.py
    build_yolo_labels.py
    evaluate_hard_distill.py
  src/matmatch2real/
    core/
    data/
    evaluation/
    hard_distill/
    soft_distill/
    student/
    teacher/
    utils/
  checkpoints/         Local weight files
  external/            Upstream external repositories
  lib/                 Local prototype files, CSVs, and related assets
  runs/                Ultralytics run outputs
  outputs/             Soft-distill outputs
```

## 2. Main Capabilities

- Teacher-side pseudo-label generation based on SAM2
- Teacher post-processing, reliability scoring, and DINOv3 classification
- COCO-style pseudo-label export and evaluation
- Soft-distill training entry point
- Two-stage hard-distill workflow
- Few-shot sampling from ground-truth training data
- Sampling by total image count for hard-distill experiments
- Hard-distill training, evaluation, and inference profiling
- Debug scripts for teacher instances, DINOv3 feature maps, and YOLO inference cost analysis

## 3. Installation From Scratch

The repository is primarily used in a Windows + PowerShell setup.

The pinned dependency files in this repository mirror the recent local `sam2` experiment environment:

- Python `3.12`
- `torch==2.6.0+cu126`
- `torchvision==0.21.0+cu126`
- `ultralytics==8.4.22`
- `numpy==2.1.2`
- `Pillow==11.0.0`
- `PyYAML==6.0.2`
- `pycocotools==2.0.10`

Files:

- [environment.yml](environment.yml): minimal conda environment definition
- [requirements.txt](requirements.txt): base Python dependencies
- [requirements-cu126.txt](requirements-cu126.txt): tested CUDA 12.6 runtime stack

### 3.1 Create the Python Environment

Using Conda:

```powershell
conda env create -f environment.yml
conda activate matmatch2real
pip install -r requirements-cu126.txt
```

Using `venv`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-cu126.txt
```

If your machine is not CUDA 12.6-based, keep the versions in `requirements.txt` and install a matching `torch` / `torchvision` pair from the official PyTorch selector before installing `ultralytics`.

### 3.2 Download Upstream Source Repositories

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\download_upstream_sources.ps1
```

Bash:

```bash
bash ./download_upstream_sources.sh
```

This clones:

- `external/sam2`
- `external/dinov3`

### 3.3 Download Upstream Weights

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\download_upstream_weights.ps1
```

Bash:

```bash
bash ./download_upstream_weights.sh
```

This downloads the checkpoints expected by the default configs:

- `checkpoints/upstream/sam2/sam2.1_hiera_tiny.pt`
- `checkpoints/upstream/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`

### 3.4 Verify the Installation

Run a lightweight import check:

```powershell
python -c "import torch, torchvision, ultralytics, pycocotools, numpy, yaml; print(torch.__version__)"
```

Then verify the main entry points are available:

```powershell
python scripts\build_yolo_labels.py --help
python scripts\run_hard_distill.py --help
python scripts\export_pseudolabels.py --help
```

Important notes:

- `external/`, `checkpoints/`, `runs/`, `outputs/`, and large local assets are not intended to be uploaded to GitHub
- anyone cloning the repo needs to prepare dependencies, checkpoints, and datasets locally
- optional integrations such as TensorBoard and SwanLab are not required for the base install path above

## 4. Dataset Layout

The default hard-distill dataset root is:

```text
../data/cdw_classify/dataset_seg
```

The expected dataset layout is:

```text
dataset_seg/
  images/
    train/
    val/
    test/
    pseudo/
  labels/
    train/
    val/
    test/
    pseudo/
  annotations/
    instances_train.json
    instances_val.json
    instances_test.json
  pseudolabels/
    pseudolabels.json
    pseudolabels_results.json
```

Notes:

- `labels/pseudo` is used for stage-1 pseudo pretraining
- `labels/train` is used for few-shot or image-count sampling
- `annotations/*.json` and `pseudolabels/*.json` are local experiment assets and are intentionally not tracked by Git
- a fuller data-preparation guide is available in [docs/dataset.md](docs/dataset.md)

## 5. Teacher Pseudo-Label Workflow

### 5.1 Export Pseudo Labels

Main entry point:

```bash
python scripts/export_pseudolabels.py --config configs/teacher/teacher_default.yaml
```

Common usage:

```bash
python scripts/export_pseudolabels.py ^
  --config configs/teacher/teacher_default.yaml ^
  --coco-gt ../data/cdw_classify/dataset_seg/annotations/instances_test.json ^
  --images-root ../data/cdw_classify/dataset_seg/images/test ^
  --output ../data/cdw_classify/dataset_seg/pseudolabels/pseudolabels.json ^
  --output-results ../data/cdw_classify/dataset_seg/pseudolabels/pseudolabels_results.json
```

Outputs:

- A COCO-like pseudo-label file
- A COCO results file

### 5.2 Evaluate Pseudo Labels

Main entry point:

```bash
python scripts/eval_pseudolabels.py --pred ../data/cdw_classify/dataset_seg/pseudolabels/pseudolabels_results.json --gt ../data/cdw_classify/dataset_seg/annotations/instances_test.json --iou-type segm --class-agnostic false
```

This script now reports:

- Overall COCO metrics
- Per-class `AP`, `AP50`, `AP75`, and `AR`

To save a JSON summary:

```bash
python scripts/eval_pseudolabels.py --pred ... --gt ... --iou-type segm --class-agnostic false --out-json outputs/pseudolabel_eval_summary.json
```

## 6. Soft Distillation

The soft-distill training entry point is:

```bash
python scripts/train_soft_distill.py --config configs/distill/distill_default.yaml
```

This workflow does the following:

1. Load the distillation config
2. Build the training and evaluation datasets
3. Build the teacher and student models
4. Run the distillation training loop
5. Save logs, checkpoints, and a config snapshot

The output directory is created by the training script.

Typical experiment directions:

- Distillation from offline pseudo labels
- Enabling or disabling different soft-distill loss branches
- Measuring whether teacher soft supervision improves the student

## 7. Hard Distillation

The hard-distill workflow in this repository is explicitly split into two separate stages.

### 7.1 Stage 1: Pseudo Pretraining

Use only pseudo labels to pretrain the student:

```bash
python scripts/run_hard_distill.py --cfg configs/hard_distill/pseudo_pretrain.yaml
```

Typical output directory:

```text
runs/hard_distill/pseudo_pretrain
```

Typical checkpoint to reuse later:

```text
runs/hard_distill/pseudo_pretrain/weights/best.pt
```

### 7.2 Stage 2: Few-Shot Fine-Tuning

First sample a subset from ground-truth `train`, then continue training with the official Ultralytics interface.

Sample by `K` images per class:

```bash
python scripts/build_fewshot_split.py --shot 5 --seed 42
```

Sample by total number of images:

```bash
python scripts/build_fewshot_split.py --image-count 117 --seed 42
```

Generated outputs include:

- `configs/hard_distill/generated/*.yaml`
- `configs/hard_distill/generated/*_summary.json`
- `configs/hard_distill/generated/manifests/*.txt`

Then continue training. Example for few-shot:

```bash
python scripts/run_hard_distill.py --cfg configs/hard_distill/default.yaml --data configs/hard_distill/generated/shot_5_seed_42.yaml --model runs/hard_distill/pseudo_pretrain/weights/best.pt --name shot_5_seed_42
```

Example for image-count sampling:

```bash
python scripts/run_hard_distill.py --cfg configs/hard_distill/default.yaml --data configs/hard_distill/generated/images_117_seed_42.yaml --model runs/hard_distill/pseudo_pretrain/weights/best.pt --name image_shot_117_seed_42
```

### 7.3 Full Ground-Truth Training

If you want to train directly on the full ground-truth dataset instead of using few-shot sampling:

```bash
python scripts/run_hard_distill.py --cfg configs/hard_distill/default.yaml --data configs/hard_distill/cdw_dataset_seg.yaml --name full_gt_train
```

If you want to initialize from the pseudo-pretrained checkpoint:

```bash
python scripts/run_hard_distill.py --cfg configs/hard_distill/default.yaml --data configs/hard_distill/cdw_dataset_seg.yaml --model runs/hard_distill/pseudo_pretrain/weights/best.pt --name full_gt_finetune
```

## 8. Hard-Distill Evaluation

Evaluation entry point:

```bash
python scripts/evaluate_hard_distill.py --weights runs/hard_distill/shot_5_seed_42/weights/best.pt --split test --device 0
```

You can also pass the dataset yaml explicitly:

```bash
python scripts/evaluate_hard_distill.py --weights runs/hard_distill/shot_5_seed_42/weights/best.pt --data configs/hard_distill/generated/shot_5_seed_42.yaml --split test --device 0
```

This script:

- Calls the official `YOLO.val()` interface
- Reports box and mask metrics
- Writes `evaluation_summary.json` and `evaluation_summary.txt` to the validation output directory

## 9. Main Scripts

### 9.1 `scripts/`

- `export_pseudolabels.py`
  Export teacher-generated pseudo labels
- `eval_pseudolabels.py`
  Evaluate pseudo labels and report per-class metrics
- `train_soft_distill.py`
  Soft-distillation training entry point
- `run_hard_distill.py`
  Single-stage hard-distill training entry point built on official Ultralytics training
- `build_fewshot_split.py`
  Generate training subsets by few-shot or by total image count
- `build_yolo_labels.py`
  Convert COCO annotations and pseudo labels into Ultralytics `labels/`
- `evaluate_hard_distill.py`
  Evaluate hard-distill checkpoints
- `visualize_pseudolabels.py`
  Visualize pseudo labels and optionally select a specific `image_id`

### 9.2 `debug/teacher/`

- `visualize_dinov3_first_instance_feature_map.py`
  Given GT and an `image_id`, export the first instance patch-token feature map from the teacher-side DINOv3 extractor
- `profile_teacher_inference.py`
  Measure teacher inference FPS, latency, and GPU memory usage
- `profile_postprocess.py`
  Compare post-processing runtime paths
- `classify_gt_boxes_with_prototype.py`
  Debug GT-instance classification with the prototype classifier
- `classify_gt_boxes_with_knn.py`
  Debug GT-instance classification with the kNN classifier

### 9.3 `debug/hard_distill/`

- `profile_yolo_inference.py`
  Measure YOLO inference FPS, latency, and GPU memory usage for the hard-distill workflow

## 10. Debug Examples

### 10.1 Visualize a Specific Pseudo-Label Image

```bash
python scripts/visualize_pseudolabels.py --image-id 42
```

### 10.2 Visualize a Teacher DINOv3 Feature Map

```bash
python debug/teacher/visualize_dinov3_first_instance_feature_map.py --image-id 42 --gt-json ../data/cdw_classify/dataset_seg/annotations/instances_test.json --images-root ../data/cdw_classify/dataset_seg/images/test
```

### 10.3 Profile Teacher Inference

```bash
python debug/teacher/profile_teacher_inference.py --config configs/teacher/teacher_default.yaml --limit 100 --warmup 1 --repeat 3
```

### 10.4 Profile Hard-Distill YOLO Inference

```bash
python debug/hard_distill/profile_yolo_inference.py --cfg configs/hard_distill/default.yaml --weights runs/hard_distill/shot_5_seed_42/weights/best.pt --data configs/hard_distill/generated/shot_5_seed_42.yaml --split test --warmup 1 --repeat 3
```

## 11. GitHub Upload Notes

The repository is configured not to upload the following:

- All `*.pt`
- All `*.pth`
- All `*.json`
- `configs/hard_distill/generated/`
- `debug/output/`
- `debug/outputs/`
- `lib/`
- `external/`
- `outputs/`
- `runs/`

So the GitHub version of the repository is intended to contain:

- Source code
- Config templates
- Debug scripts
- Training and evaluation entry points

It is not intended to contain:

- Datasets
- Result files
- Model weights
- Runtime-generated experiment directories

## 12. Known Notes

- Many default paths assume datasets live outside the repository under `../data/...`
- The teacher pipeline depends on upstream repositories and local checkpoints under `external/` and `checkpoints/`
- `configs/hard_distill/generated/` is an experiment artifact directory and should not be maintained manually
- To reproduce experiments, prepare the dataset, checkpoints, and upstream dependencies first, then run the commands in this README
