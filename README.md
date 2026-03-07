# distill_cdw

Scaffold for SAM2 teacher -> lightweight student distillation. The design mirrors the structure of `segment_cdw/` while keeping components modular and swappable.

Key ideas:
- Teacher: SAM2 generates class-agnostic masks; postprocess + classifier adds class + score.
- Student: lightweight model learns from pseudo labels with reliability-aware loss reweighting.
- Data: supports offline pseudo labels (cached) and optional online generation via teacher.
- Remote dataset: optional SwanLab sync to pull/push data caches.

## Quick start

```bash
python distill_cdw/scripts/train_distill.py \
  --config distill_cdw/configs/distill_default.yaml
```

The default config is JSON-compatible YAML. If PyYAML is not installed, the loader falls back to JSON parsing.

## Directory layout

```
distill_cdw/
  README.md
  configs/
    distill_default.yaml
  scripts/
    train_distill.py
    export_pseudolabels.py
    evaluate.py
    debug_run.py
  distill/
    core/
    data/
    teacher/
    student/
    evaluation/
    utils/
```

## Data format (pseudo labels)

The loader accepts a COCO-like JSON with additional fields:

```
{
  "images": [{"id": 1, "file_name": "xxx.jpg", "height": 512, "width": 512}],
  "annotations": [{
    "id": 1,
    "image_id": 1,
    "category_id": 3,
    "bbox": [x, y, w, h],
    "score": 0.93,
    "reliability": 0.85,
    "segmentation": {"mask": [[0,1,...]]}
  }],
  "categories": [{"id": 3, "name": "class_name"}],
  "meta": {"source": "sam2"}
}
```

Masks can be bitmap arrays or RLE (placeholder in this scaffold). See `distill_cdw/distill/data/pseudolabel_io.py` for the exact fields.

## Extending

- Teacher pipeline: `distill_cdw/distill/teacher/sam2_teacher.py` and `distill_cdw/distill/teacher/postprocess.py`.
- Student models: `distill_cdw/distill/student/models.py` and `distill_cdw/distill/student/heads.py`.
- Losses: `distill_cdw/distill/student/losses.py`.
- Evaluation: `distill_cdw/distill/evaluation/`.
- Registry: `distill_cdw/distill/core/registry.py`.

## Postprocess acceleration

You can switch between numpy and torch postprocess paths via config:

```
teacher:
  postprocess:
    runtime:
      use_torch: true
      device: cuda
      topk_pre: 0
      torch_nms: true
      topk_nms: 0
      log_stats: false
```

Tools:

- Profile numpy vs torch paths:
  - `python distill_cdw/scripts/profile_postprocess.py --config distill_cdw/configs/distill_default.yaml --image /path/to/img.jpg --device cuda`
- Verify equivalence between paths:
  - `python distill_cdw/scripts/verify_equivalence.py --config distill_cdw/configs/distill_default.yaml --image /path/to/img.jpg --device cuda`

## Segment CDW adapter

The teacher can call into `segment_cdw/` via a dynamic adapter. Set in config:

```
"teacher": {
  "segment_cdw": {
    "enabled": true,
    "module": "segment_cdw.run_eval",
    "callable": "run_inference"
  }
}
```

If the adapter cannot import the target, it logs a clear error and falls back to a no-op.

## Remote dataset (SwanLab)

Config block:

```
"data": {
  "remote": {
    "enabled": false,
    "provider": "swanlab",
    "remote_uri": "",
    "local_cache_dir": "data/cache/distill",
    "sync_mode": "down",
    "version": ""
  }
}
```

When enabled, the trainer calls `ensure_dataset_available` before building datasets. If SwanLab is not installed, it degrades gracefully with a clear message and uses local paths.

## Notes

- This is a scaffold with placeholder components. The main entry point runs with dummy data if real data is missing.
- Replace placeholders with real SAM2 inference, postprocessing, and classifier logic as needed.
