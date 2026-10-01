"""Render identified current contexts into a fresh provenance-bearing directory.

The historical implementation is preserved under legacy/2026-09-30. Missing
legacy contexts are not reconstructed from matching filenames or image plots.
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import torch

from experiment_io import atomic_bytes, canonical_json, file_hash
from faithfulness import authenticated_source, load_sidecar
import overlays


def render_run(metrics, task, output, *, language_path=None, three_panel=False, episode_frames=4, limit=None):
    metrics, output = Path(metrics).resolve(), Path(output).resolve()
    rows, manifest = authenticated_source(metrics)
    configuration = manifest["configuration"]
    if re.fullmatch(r"[0-9a-f]{32}", manifest["run_id"]) is None:
        raise ValueError("Invalid output run identifier")
    if configuration["task"] != task:
        raise ValueError("Requested task differs from authenticated run")
    if configuration["observation_pipeline"] != "one_current_external_camera_five_background_slots":
        raise ValueError("Unknown observation geometry")
    if type(episode_frames) is not int or episode_frames < 2 or (limit is not None and limit < 1):
        raise ValueError("Require at least two episode frames and a positive optional limit")
    token_labels = None
    if language_path is not None:
        if file_hash(language_path) != configuration["language"]["task_embedding_sha256"]:
            raise ValueError("Language label file differs from recorded embedding identity")
        language = torch.load(language_path, weights_only=True, map_location="cpu")
        token_labels = language.get("tokens")
    rows = rows if limit is None else rows[:limit]
    if not rows:
        raise ValueError("No authenticated calls selected")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "run_id": manifest["run_id"],
        "configuration_sha256": manifest["configuration_sha256"],
        "configuration": configuration, "source_metrics_sha256": file_hash(metrics),
        "generator_sha256": file_hash(__file__), "renderer_sha256": file_hash(overlays.__file__),
        "selection": {"method": "all" if limit is None else "explicit_prefix", "limit": limit,
                      "episode_summary": "evenly_spaced_within_selected_calls", "frames": episode_frames},
        "vision_display": "absolute signed hidden-coordinate sum; nearest patch display on the saved 384x384 input extent",
        "interpretation": "Representation attribution illustration, not pixel attribution or behavioral validation.",
        "contexts": [], "artifacts": []}
    by_episode = defaultdict(list)

    def checkpoint():
        atomic_bytes(output / "manifest.json", canonical_json(record) + b"\n")

    checkpoint()
    try:
        for row in rows:
            sidecar, digest = load_sidecar(row, metrics, manifest)
            context_id = row["context_id"]
            if re.fullmatch(r"[0-9a-f]{64}", context_id) is None:
                raise ValueError("Invalid output context identifier")
            directory = output / manifest["run_id"] / context_id
            directory.mkdir(parents=True, exist_ok=False)
            context = {"run_id": manifest["run_id"], "context_id": context_id,
                "episode": row["episode"], "policy_call_idx": row["policy_call_idx"],
                "seed": row["seed"], "task": task, "model": row["model"],
                "target": configuration["target"], "m": configuration["m"],
                "solver_steps": configuration["solver_steps"], "sidecar_sha256": digest}
            overlays.render_overlay_only_png(sidecar, directory / "vision.png")
            overlays.render_tokens_only_figure(sidecar, token_labels, directory / "language.png",
                                               lang_attn_mask=sidecar["lang_attn_mask"])
            if three_panel:
                overlays.render_step_figure(sidecar, sidecar["lang_attn_mask"], token_labels,
                    directory / "three-panel.png", title=f"{task} | {row['model']} | seed {row['seed']} | episode {row['episode']} call {row['policy_call_idx']}")
            record["contexts"].append(context)
            for path in sorted(directory.glob("*.png")):
                record["artifacts"].append({"file": path.relative_to(output).as_posix(), "sha256": file_hash(path), "context_id": context_id})
            by_episode[row["episode"]].append(row)
            checkpoint()
        for episode, episode_rows in sorted(by_episode.items()):
            payloads = []
            for row in sorted(episode_rows, key=lambda value: value["policy_call_idx"]):
                payload, _ = load_sidecar(row, metrics, manifest)
                payload["policy_call_idx"] = row["policy_call_idx"]
                payloads.append(payload)
            path = output / manifest["run_id"] / f"episode-{episode}-summary.png"
            overlays.render_episode_summary(payloads, path, n_frames=episode_frames)
            record["artifacts"].append({"file": path.relative_to(output).as_posix(), "sha256": file_hash(path),
                                        "episode": episode, "context_ids": [r["context_id"] for r in episode_rows]})
        record["status"] = "complete"
        checkpoint()
    except Exception as error:
        record.update(status="failed", failure={"type": type(error).__name__, "reason": str(error)})
        checkpoint()
        raise
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True, help="Fresh directory; existing outputs are never overwritten")
    parser.add_argument("--lang-embeds")
    parser.add_argument("--three-panel", action="store_true")
    parser.add_argument("--episode-frames", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    record = render_run(args.metrics, args.task, args.out, language_path=args.lang_embeds,
        three_panel=args.three_panel, episode_frames=args.episode_frames, limit=args.limit)
    print(f"Rendered {len(record['contexts'])} authenticated contexts; no historical result was regenerated.")


if __name__ == "__main__":
    main()
