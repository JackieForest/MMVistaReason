# Distillation Trajectory Verification

This pipeline uses Qwen3.5-27B as the common verifier for 9B, 27B, and 122B
distillation rollouts. Select the rollout stage by switching the input/output
paths and split count in `config/verify.yaml`. The checked-in configuration
currently points to the 9B rollout stage.

The verifier evaluates final-answer correctness first and reasoning consistency
second. Every candidate is sent to the verifier; tag-format diagnostics are
retained as metadata but are not used as a pre-filter. The verifier must emit a
single JSON object with `verdict`, `reasoning_consistent`, and `reason`.

```bash
bash scripts/launch.sh config/verify.yaml
python3 scripts/status.py --config config/verify.yaml
```
