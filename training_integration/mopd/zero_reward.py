#!/usr/bin/env python3
"""Zero task reward for pure multi-teacher on-policy distillation runs.

The PPO trainer still expects a reward tensor, but MOPD updates are driven by
the distillation loss when ``use_task_rewards=False``.
"""


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    return {
        "score": 0.0,
        "acc": 0.0,
        "task_reward": 0.0,
    }
