"""Toy RDT-shaped contexts and a stubbed replay layer for the revision GPU tests."""
import json
from types import SimpleNamespace

import torch

import revision_numerics as rn
from experiment_io import file_hash
from per_step_attribution import MANISKILL_INDICES

GRID = [0, 1, 5, 10, 20, 30, 50, 75, 100]
STRICT = dict(cublas_workspace_config=":4096:8", deterministic_algorithms=True, deterministic_warn_only=False,
              cudnn_benchmark=False, cudnn_deterministic=True, float32_matmul_precision="highest",
              cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False)
DEFAULT = dict(STRICT, cublas_workspace_config=None, deterministic_algorithms=False, cudnn_deterministic=False,
               cudnn_allow_tf32=True)


def toy_context(dtype=torch.bfloat16, seed=0, hidden=4, tokens=20):
    """Tensors shaped like a replayed RDT context, with a small differentiable policy."""
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(1, 4374, hidden, generator=g)
    img_bl = img.clone()
    img_bl[:, 2187:2916] = 0.1 * torch.randn(1, 729, hidden, generator=g)
    lang = torch.zeros(1, 1024, hidden)
    lang[:, :tokens] = torch.randn(1, tokens, hidden, generator=g)
    lang_bl = lang.clone()
    lang_bl[:, :tokens] = 0.05 * torch.randn(1, tokens, hidden, generator=g)
    mask = torch.zeros(1, 1024, dtype=torch.long)
    mask[0, :tokens] = 1
    state = torch.zeros(1, 1, 128)
    state[0, 0, MANISKILL_INDICES] = torch.randn(8, generator=g)
    action_mask = torch.zeros(1, 1, 128)
    action_mask[0, 0, MANISKILL_INDICES] = 1
    adaptor = torch.nn.Linear(256, hidden)
    with torch.no_grad():
        adaptor.weight.copy_(0.1 * torch.randn(hidden, 256, generator=g))
        adaptor.bias.zero_()
    adaptor = adaptor.to(dtype).requires_grad_(False)
    w_img, w_lang, w_state = (torch.randn(hidden, 8, generator=g) for _ in range(3))
    scale = torch.linspace(0.5, 1.5, 64).view(1, 64, 1)

    def sample(language, image, state_token):
        feature = (torch.tanh(image[:, 2187:2916].float() @ w_img).mean(1)
                   + torch.tanh(language[:, :tokens].float() @ w_lang).mean(1)
                   + torch.tanh(state_token.float() @ w_state).mean(1))
        action = torch.zeros(1, 64, 128)
        action[..., MANISKILL_INDICES] = torch.tanh(feature).unsqueeze(1) * scale
        return action.to(dtype)

    ctx = dict(img_adapted=img.to(dtype), img_adapted_bl=img_bl.to(dtype), lang_adapted=lang.to(dtype),
               lang_adapted_bl=lang_bl.to(dtype), lang_attn_mask=mask, state_input_actual=state.to(dtype),
               state_input_baseline=torch.zeros_like(state).to(dtype), action_mask=action_mask.to(dtype),
               runner=SimpleNamespace(state_adaptor=adaptor), seeded_conditional_sample=sample,
               initial_noise=torch.zeros(1, 64, 128), ctrl_freqs=torch.tensor([25.0]))
    with torch.no_grad():
        ctx["state_traj_actual"] = adaptor(torch.cat([ctx["state_input_actual"], ctx["action_mask"]], dim=2))
        ctx["ref_action"] = sample(ctx["lang_adapted"], ctx["img_adapted"], ctx["state_traj_actual"])
    return ctx


def toy_protocol():
    return dict(schema=rn.PROTOCOL_SCHEMA, decision_id="TEST", recorded_before_execution=True,
                banks=dict(P=dict(source_replay_settings=STRICT), M=dict(source_replay_settings=DEFAULT)),
                strict_settings=STRICT,
                numerics=dict(levels=3, comparator_rule="T8", historical_budgets=[4], gauss_legendre=[3, 4],
                              adaptive=dict(rtol=1e-3, max_intervals=3, snapshot_evaluations=[63]),
                              store_rules=dict(vision=["T8", "E4"], language="all", state="all"),
                              qualification_triple=["A4", "A3GL4", "A2T8"]),
                rankings=dict(ig_rules=["T8", "E4"], path_gradient_rules=["T8"], fp32_rules=["A2T8", "A3GL4", "A4"]),
                evaluation=dict(grid_percent=GRID, count_rule="max1_round",
                                denominator_min=dict(Q=1e-12, L2=1e-8, RMS=1e-8),
                                random_permutations=dict(vision=2, language=3), random_seed=7),
                verification=dict(levels=2, repeats=2))


def stub_replay_layer(monkeypatch, tmp_path, contexts=2, fail_on=None):
    """Replace the GPU replay layer with the toy context; returns (protocol path, bank id)."""
    import faithfulness
    import fp32_probe_cache
    import paired_comparison
    from scripts import validate_downstream_precision as precision
    entries, rows, sidecars = [], {}, {}
    for i in range(contexts):
        ctx = toy_context(seed=i)
        entry = dict(episode=i, policy_call_idx=3, status="available", context_id=f"ctx{i}", reset_seed=900 + i,
                     episode_id=f"ep{i}", row_sha256="r", sidecar_sha256="s")
        entries.append(entry)
        rows[(i, 3)] = dict(episode=i, policy_call_idx=3, context_id=f"ctx{i}", seed=900 + i, attr_sha256="s")
        sidecars[f"ctx{i}"] = ctx
    bank = dict(kind="paired_context_bank", selection=dict(rule="exact_collector_prespecified_call_indices"),
                contexts=entries)
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(bank))
    protocol = toy_protocol()
    protocol["banks"]["P"].update(metrics=str(tmp_path / "metrics.jsonl"), bank=str(bank_path),
                                  bank_sha256=file_hash(bank_path), task="PickCube-v1", model="170m", lang_dir="lang",
                                  modalities=["vision", "language", "state"], fp32_contexts="first_episodes:1",
                                  adaptive_contexts="first_episodes:1", full_node_contexts="none",
                                  full_node_modalities=["state"], verify_contexts="contexts:e0000-c003")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    runner = torch.nn.Linear(1, 1)

    def replay_context(args, row, sidecar, pipe, lang, **kwargs):
        if fail_on == row["context_id"]:
            raise ValueError("Replayed ref_action differs from source")
        return sidecars[row["context_id"]]

    def load_sidecar(row, metrics, manifest):
        ctx = sidecars[row["context_id"]]
        return {key: ctx[key] for key in ("initial_noise", "ref_action", "lang_attn_mask", "action_mask",
                                          "ctrl_freqs")}, "sidecar-" + row["context_id"]

    def fp32_context(source, runner32):
        index = next(k for k, v in sidecars.items() if torch.equal(v["ref_action"], source["ref_action"]))
        return toy_context(dtype=torch.float32, seed=int(index[3:])), dict(probe="toy")

    monkeypatch.setattr(faithfulness, "replay_inputs", lambda args, apply_limit=True: (list(rows.values()),
                        dict(run_id="run", configuration_sha256="cfg"), "digest"))
    monkeypatch.setattr(faithfulness, "replay_pipeline", lambda args, manifest: (dict(runner=runner), dict()))
    monkeypatch.setattr(faithfulness, "replay_context", replay_context)
    monkeypatch.setattr(faithfulness, "load_sidecar", load_sidecar)
    monkeypatch.setattr(paired_comparison, "make_bank", lambda metrics, selection: json.loads(bank_path.read_text()))
    monkeypatch.setattr(paired_comparison, "bank_rows", lambda bank, rows_, manifest, digest: rows)
    monkeypatch.setattr(paired_comparison, "make_fp32_probe_context", fp32_context)
    monkeypatch.setattr(precision, "convert_downstream_to_fp32", lambda runner32: dict(converted=True))
    monkeypatch.setattr(fp32_probe_cache, "apply_settings", lambda settings: dict(settings))
    monkeypatch.setattr(fp32_probe_cache, "runtime_settings", lambda: dict(STRICT))
    return protocol_path, protocol


def worker_args(protocol_path, out, phases="verify,bf16,fp32,scoring"):
    return rn.parse_args(["--protocol", str(protocol_path), "--bank", "P", "--phases", phases, "--out", str(out)])
