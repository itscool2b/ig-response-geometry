#!/usr/bin/env bash
# Month 7 verification pass, as executed on a RunPod RTX 5090 pod
# (template runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404, LunarG Vulkan SDK
# 1.4.313, torch 2.8 cu128). Re-runs the main-pass protocol with the current
# seed-tagged sidecar naming, then faithfulness and the standard sanity checks,
# then the m=128 seed-42 arm, writing every record under the m7_ prefix.
#
# The pass reproduces the original m=64 population shape deliberately:
# --max-policy-calls 4 (7 on PegInsertionSide) matches the per-task ManiSkill
# default step caps the original pass ran under, so the rows are comparable to
# the published tables. Prerequisites are the same as run_full_pass.sh, plus
# real T5-XXL instruction embeddings under data/lang_embeds/ (the recorded run
# encoded them on the pod with encode_task_lang.py). GPU required.
#
# Stages: v1 (full pass) | v2 (faithfulness) | v3 (sanity) | v4 (m=128 seed-42)
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${STAGE:-all}"
MODEL="${MODEL:-170m}"
run_stage() { [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }

peg_calls() { [[ "$1" == "PegInsertionSide-v1" ]] && echo 7 || echo 4; }

if run_stage v1; then
  for task in PickCube-v1 StackCube-v1 PegInsertionSide-v1 PickSingleYCB-v1; do
    calls=$(peg_calls "$task")
    for seed in 42 142; do
      out="data/m7_metrics_${task}_${MODEL}_seed${seed}.jsonl"
      echo "=== V1 ${task} seed=${seed} calls=${calls} -> ${out} ==="
      .venv/bin/python per_step_ig.py --task "$task" --model "$MODEL" \
        --episodes 50 --m 64 --seed-base "$seed" --max-policy-calls "$calls" \
        --out "$out" --resume --no-checkpoint
    done
  done
fi

if run_stage v2; then
  for task in PickCube-v1 StackCube-v1 PegInsertionSide-v1 PickSingleYCB-v1; do
    for seed in 42 142; do
      echo "=== V2 faithfulness ${task} seed=${seed} ==="
      .venv/bin/python faithfulness.py \
        --metrics "data/m7_metrics_${task}_${MODEL}_seed${seed}.jsonl" \
        --task "$task" --model "$MODEL" --no-checkpoint \
        --out "data/m7_metrics_faithfulness_${task}_${MODEL}_seed${seed}.jsonl"
    done
  done
fi

if run_stage v3; then
  for task in PickCube-v1 StackCube-v1 PegInsertionSide-v1 PickSingleYCB-v1; do
    for seed in 42 142; do
      metrics="data/m7_metrics_${task}_${MODEL}_seed${seed}.jsonl"
      echo "=== V3 sanity C1 ${task} seed=${seed} ==="
      .venv/bin/python sanity.py --phase C1 --metrics "$metrics" --task "$task" \
        --model "$MODEL" --m 16 --limit 50 --no-checkpoint \
        --out "data/m7_metrics_sanity_C1_${task}_${MODEL}_seed${seed}.jsonl"
      echo "=== V3 sanity C2 ${task} seed=${seed} ==="
      .venv/bin/python sanity.py --phase C2 --metrics "$metrics" --task "$task" \
        --model "$MODEL" --m 16 --limit 50 --shuffle-seed 777 --no-checkpoint \
        --out "data/m7_metrics_sanity_C2_${task}_${MODEL}_seed${seed}.jsonl"
    done
  done
fi

if run_stage v4; then
  for task in PegInsertionSide-v1 PickSingleYCB-v1; do
    out="data/m7_metrics_${task}_${MODEL}_seed42_m128.jsonl"
    echo "=== V4 m=128 ${task} seed=42 ==="
    .venv/bin/python per_step_ig.py --task "$task" --model "$MODEL" \
      --episodes 8 --m 128 --seed-base 42 \
      --out "$out" --resume --no-checkpoint
    .venv/bin/python faithfulness.py --metrics "$out" --task "$task" \
      --model "$MODEL" --no-checkpoint \
      --out "data/m7_metrics_faithfulness_${task}_${MODEL}_seed42_m128.jsonl"
  done
fi

echo "verification pass done."
