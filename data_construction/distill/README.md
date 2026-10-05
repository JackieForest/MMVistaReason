# Shared Qwen3.5 Distillation Rollout

This directory contains one shared implementation for the 9B, 27B, and 122B
stages of the distillation cascade. Select the stage by editing
`config/distill.yaml`; the three model paths and their matching TP/GPU presets
are documented together in that file.

```bash
python3 scripts/split.py --config config/distill.yaml --resume
bash scripts/launch.sh config/distill.yaml
python3 scripts/status.py --config config/distill.yaml
```

Each sample receives four independent rollout requests. Successful and failed
records are written to the directories configured by `success_dir_name` and
`failed_dir_name`.

The active configuration is the 9B stage. When switching to 27B or 122B, also
change the stage name, input/output paths, split count, concurrency, tensor
parallel size, and requested GPU count.
