# Dataset Preparation

This repository assumes that datasets live outside the Git repository, by default under:

```text
../data/cdw_classify/dataset_seg
```

The default hard-distill YAMLs and several scripts point to that layout. If you use a different location, update the relevant config values or pass CLI overrides.

## 1. Expected Directory Layout

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

Usage by workflow:

- `images/train` + `annotations/instances_train.json`: supervised training source.
- `images/val` + `annotations/instances_val.json`: validation source.
- `images/test` + `annotations/instances_test.json`: final evaluation source.
- `images/pseudo` + `pseudolabels/pseudolabels.json`: pseudo-labeled images for stage-1 hard-distill pretraining.
- `labels/*`: Ultralytics segmentation labels generated from COCO-style annotations and pseudo labels.

## 2. Required COCO Fields

Ground-truth annotation files are expected to follow standard COCO-style instance segmentation structure:

- `images[]`: each item should provide `id`, `file_name`, `height`, and `width`.
- `annotations[]`: each item should provide `image_id`, `category_id`, `bbox`, and `segmentation`.
- `categories[]`: each item should provide `id` and `name`.

Important repository-specific notes:

- `bbox` is expected in COCO `xywh` format.
- `segmentation` can be polygon lists or COCO RLE.
- foreground category ids should be positive integers because the YOLO label builder ignores ids `<= 0`
- relative `file_name` entries are resolved against the split image directory

## 3. Pseudo-Label Files

Teacher export writes two related JSON files:

- `pseudolabels/pseudolabels.json`: COCO-like annotation payload used by repository loaders.
- `pseudolabels/pseudolabels_results.json`: COCO results payload used for evaluation with `pycocotools`.

The pseudo-label loader reads:

- `images[]`
- `annotations[]`
- optional `meta`

Each pseudo annotation should contain at least:

- `image_id`
- `category_id` or `class_id`
- `bbox`
- `score`
- `reliability`
- `segmentation`

## 4. Build Ultralytics Labels

Convert COCO-style annotations and pseudo labels into Ultralytics segmentation label text files with:

```powershell
python scripts/build_yolo_labels.py --dataset-root ..\data\cdw_classify\dataset_seg --clear-existing
```

Useful options:

- `--labels-dir-name labels`: output directory name under the dataset root.
- `--min-points 3`: minimum polygon vertex count kept per instance.
- `--min-area 4.0`: minimum polygon area kept per instance.
- `--clear-existing`: remove old label files before rebuilding.

This command writes:

- `labels/train/*.txt`
- `labels/val/*.txt`
- `labels/test/*.txt`
- `labels/pseudo/*.txt`
- `labels/summary.json`

## 5. Hard-Distill YAML Expectations

The provided hard-distill dataset YAMLs expect:

- `path` to point at the dataset root
- `train`, `val`, and `test` to point at image folders relative to that root
- `names` to map class indices used by YOLO training

Example:

```yaml
path: ../../data/cdw_classify/dataset_seg
train: images/val
val: images/test
test: images/test
names:
  0: crushed_stone
  1: brick
  2: concrete
  3: ceramic
```

If you change COCO category ids, regenerate labels and confirm the class index mapping in `labels/summary.json`.

## 6. Files Not Tracked By Git

The GitHub version of the repository is intended to exclude large or local-only assets. In practice, do not commit:

- raw datasets under `../data/...`
- generated `labels/` under external dataset roots
- pseudo-label JSON outputs
- evaluation result JSON files
- checkpoints under `checkpoints/`
- upstream repositories under `external/`
- experiment outputs under `runs/` and `outputs/`
- classifier support assets under `lib/`

If you want to share a tiny example payload publicly, place it in a dedicated tracked docs or examples directory and adjust `.gitignore` accordingly.
