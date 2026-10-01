"""CPU controls for evaluator identity, selection and intervention contracts."""
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from displacement import displacement, parse_grid, select_calls
from experiment_io import file_hash
from faithfulness import EvaluationWriter, preflight_source_replays
from sanity import (pixel_shuffle_image, reinit_full_backbone,
                    safe_spearman, shuffled_language)


def language_fixture(tmp_path):
    # Noncontiguous real positions challenge the old prefix-only permutation.
    mask=torch.tensor([[True,False,True,True,False,True]])
    embeds=torch.arange(12,dtype=torch.float32).reshape(1,6,2)
    artifact=tmp_path/"source.pt"
    metadata=dict(embeds=embeds,attn_mask=mask,token_ids=[10,20,30,1],task="fixture",
                  identity=dict(model_revision="fixed-t5-revision",tokenizer_sha256="tokenizer"))
    torch.save(metadata,artifact)
    base=dict(lang_tokens=embeds,lang_attn_mask=mask,lang_tokens_baseline=torch.zeros_like(embeds),
              identity=dict(task_embedding_sha256=file_hash(artifact)))
    return artifact,base,metadata


def test_post_encoding_fixed_eos_permutes_only_explicit_content_positions(tmp_path):
    path,base,_=language_fixture(tmp_path)
    before=torch.random.get_rng_state().clone()
    changed,identity=shuffled_language(base,path,mode="post_encoding",seed=7,eos_policy="fixed")
    assert torch.equal(torch.random.get_rng_state(),before)
    assert identity["destination_indices"] == [0,2,3]
    assert sorted(identity["source_indices"]) == [0,2,3]
    assert torch.equal(changed["lang_tokens"][:,[1,4,5]],base["lang_tokens"][:,[1,4,5]])
    assert torch.equal(base["lang_tokens"],torch.arange(12,dtype=torch.float32).reshape(1,6,2))
    repeated,_=shuffled_language(base,path,mode="post_encoding",seed=7,eos_policy="fixed")
    assert torch.equal(changed["lang_tokens"],repeated["lang_tokens"])


def test_post_encoding_shuffle_eos_is_a_distinct_explicit_population(tmp_path):
    path,base,_=language_fixture(tmp_path)
    _,identity=shuffled_language(base,path,mode="post_encoding",seed=7,eos_policy="shuffle")
    assert identity["destination_indices"] == [0,2,3,5]
    assert 5 in identity["source_indices"]


def test_cache_presence_cannot_change_requested_c2_protocol(tmp_path):
    path,base,_=language_fixture(tmp_path)
    cached=tmp_path/"cached.pt"
    cached.write_bytes(b"not even valid torch data")
    with pytest.raises(ValueError,match="does not consume"):
        shuffled_language(base,path,mode="post_encoding",seed=7,eos_policy="fixed",shuffled_path=cached)
    with pytest.raises(ValueError,match="explicit shuffled"):
        shuffled_language(base,path,mode="pre_encoding",seed=7,eos_policy="fixed")


def test_pre_encoding_requires_matching_encoder_seed_and_eos(tmp_path):
    path,base,metadata=language_fixture(tmp_path)
    ids=metadata["token_ids"][:-1].copy()
    random.Random(7).shuffle(ids)
    shuffled={**metadata,"token_ids":ids+[1],"shuffle_seed":7,
              "shuffle_mode":"pre_encoding_content_tokens_eos_fixed"}
    target=tmp_path/"pre.pt"
    torch.save(shuffled,target)
    result,identity=shuffled_language(base,path,mode="pre_encoding",seed=7,eos_policy="fixed",shuffled_path=target)
    assert identity["intervention_file_sha256"] == file_hash(target)
    assert result["lang_tokens"].shape == base["lang_tokens"].shape
    for change in [{"shuffle_seed":8},{"identity":{"model_revision":"different"}},
                   {"token_ids":[99,20,30,1]}]:
        torch.save({**shuffled,**change},target)
        with pytest.raises(ValueError):
            shuffled_language(base,path,mode="pre_encoding",seed=7,eos_policy="fixed",shuffled_path=target)


def test_pixel_shuffle_preserves_colors_and_has_local_rng():
    image=np.arange(36,dtype=np.uint8).reshape(3,4,3)
    state=np.random.get_state()
    changed=np.asarray(pixel_shuffle_image(image,7))
    assert sorted(map(tuple,image.reshape(-1,3))) == sorted(map(tuple,changed.reshape(-1,3)))
    assert np.array_equal(np.random.get_state()[1],state[1])
    assert np.array_equal(changed,np.asarray(pixel_shuffle_image(image,7)))


def test_spearman_undefined_is_explicit_and_nonfinite_is_not_a_pass():
    assert safe_spearman([1,1],[2,3]) is None
    assert safe_spearman([1],[2]) is None
    assert safe_spearman([1e-30,2e-30,3e-30],[3,2,1]) == pytest.approx(-1)
    with pytest.raises(ValueError,match="Nonfinite"):
        safe_spearman([1,float("nan")],[2,3])


def test_full_randomization_reports_actual_scope_and_preserves_rng():
    model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.LayerNorm(4),torch.nn.Linear(4,2))
    runner=SimpleNamespace(model=model)
    before={name:value.clone() for name,value in model.named_parameters()}
    rng=torch.random.get_rng_state().clone()
    identity=reinit_full_backbone(runner,123)
    assert not identity["layerwise_cascade"]
    assert torch.equal(rng,torch.random.get_rng_state())
    for name,value in model.named_parameters():
        if value.ndim == 1:
            assert torch.equal(value,before[name])
        else:
            assert not torch.equal(value,before[name])
            assert name in identity["changed_tensors"]


def test_displacement_casts_operands_before_bf16_subtraction():
    original=torch.ones(1,1,128,dtype=torch.bfloat16)
    perturbed=torch.full_like(original,-0.00390625)
    measured=displacement(perturbed,original)
    expected=(perturbed.float()[...,[0,1,2,3,4,5,6,10]]-original.float()[...,[0,1,2,3,4,5,6,10]]).norm().item()
    old=(perturbed-original).float()[...,[0,1,2,3,4,5,6,10]].norm().item()
    assert measured["l2_active"] == expected
    assert measured["l2_active"] != old


def test_relative_displacement_does_not_hide_zero_denominator():
    original=torch.zeros(1,1,128)
    measured=displacement(torch.ones_like(original),original)
    assert measured["rel_active"] is None
    assert measured["relative_status"] == "undefined_reference_norm"
    assert measured["l2_active"] > 0
    json.dumps(measured,allow_nan=False)
    with pytest.raises(ValueError,match="Nonfinite"):
        displacement(original+float("inf"),original)


def test_round_robin_selection_separates_seed_episode_groups():
    rows=[dict(task="t",model="m",seed=s,episode=0,policy_call_idx=i) for s in (42,142) for i in range(3)]
    chosen=select_calls(rows,3,"round_robin_episodes")
    assert [(r["seed"],r["policy_call_idx"]) for r in chosen] == [(42,0),(142,0),(42,1)]
    assert select_calls(rows,2,"prefix") == rows[:2]


@pytest.mark.parametrize("text,solver",[("0,5",True),("2,1",True),("1,1",True),("1,5",False),("0,101",False)])
def test_invalid_study_grids_fail(text,solver):
    with pytest.raises(ValueError):
        parse_grid(text,solver=solver)


def writer_args(tmp_path):
    source=tmp_path/"source.jsonl"
    source.write_bytes(b"source bytes\n")
    return SimpleNamespace(metrics=str(source),out=str(tmp_path/"output.jsonl"),limit=None)


def test_evaluation_writer_requires_exact_count_and_fresh_output(tmp_path):
    args=writer_args(tmp_path)
    with EvaluationWriter(args,None,file_hash(args.metrics),"fixture",{},1) as writer:
        writer.set_context(dict(episode=0,seed=42,policy_call_idx=0),source_attr_sha256="sha")
        writer.write(dict(event="fixture",value=1.0))
    assert json.loads((tmp_path/"output.jsonl.completion.json").read_text())["completed_rows"] == 1
    assert json.loads((tmp_path/"output.jsonl").read_text())["identity_status"] == "legacy_unverified"
    with pytest.raises(FileExistsError):
        with EvaluationWriter(args,None,file_hash(args.metrics),"fixture",{},1):
            pass


@pytest.mark.parametrize("failure",["exception","missing_rows","changed_source"])
def test_failed_evaluation_retains_failure_record_and_has_no_completion(tmp_path,failure):
    args=writer_args(tmp_path)
    with pytest.raises(ValueError):
        with EvaluationWriter(args,None,file_hash(args.metrics),"fixture",{},1) as writer:
            writer.set_context(dict(episode=0,seed=42,policy_call_idx=0))
            if failure == "exception":
                raise ValueError("Injected nonfinite response")
            if failure == "changed_source":
                writer.write(dict(event="fixture",value=1))
                from pathlib import Path
                Path(args.metrics).write_bytes(b"changed")
    rows=[json.loads(line) for line in (tmp_path/"output.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "evaluation_failure"
    assert not (tmp_path/"output.jsonl.completion.json").exists()


def test_source_replay_preflight_rejects_reference_drift(monkeypatch,tmp_path):
    import faithfulness
    payload=dict(obs_image=torch.zeros(2,2,3,dtype=torch.uint8),proprio=torch.zeros(8),
                 initial_noise=torch.ones(1),ref_action=torch.ones(1),
                 lang_attn_mask=torch.ones(1,2,dtype=torch.bool),action_mask=torch.ones(1),ctrl_freqs=torch.ones(1))
    monkeypatch.setattr(faithfulness,"load_sidecar",lambda *a:(payload,"sidecar-hash"))
    monkeypatch.setattr(faithfulness,"prepare_ig_context",lambda *a,**kw:{**payload,"ref_action":torch.zeros(1)})
    pipe={key:None for key in ("runner","vision_model","bg_image_encoded","img_tokens_baseline","action_mask","ctrl_freqs")}
    language={key:None for key in ("lang_tokens","lang_attn_mask","lang_tokens_baseline")}
    args=SimpleNamespace(metrics="fixture",target="logpi")
    with pytest.raises(ValueError,match="ref_action"):
        preflight_source_replays(args,[dict(seed=42,context_id="context")],{},pipe,language)


def test_sanity_does_not_mutate_model_before_source_replay_passes(monkeypatch,tmp_path):
    import sanity
    args=writer_args(tmp_path)
    for key,value in dict(phase="C1",shuffle_seed=None,c2_mode=None,eos_policy=None,
                          shuffled_language=None,m=1,randomization_seed=123,c2_modalities="both",
                          task="fixture",model="170m",target="logpi",lang_dir=str(tmp_path)).items():
        setattr(args,key,value)
    manifest=dict(run_id="run",configuration_sha256="config",configuration=dict(m=1,quadrature="trapezoid"))
    monkeypatch.setattr(sanity,"parse_args",lambda:args)
    monkeypatch.setattr(sanity,"replay_inputs",lambda a:([dict(episode=0)],manifest,file_hash(args.metrics)))
    monkeypatch.setattr(sanity,"replay_pipeline",lambda *a:({"runner":None,"identity":{}},{"identity":{}}))
    def drift(*a,**kw):
        raise ValueError("Source reference drift")
    monkeypatch.setattr(sanity,"preflight_source_replays",drift)
    mutated=[]
    monkeypatch.setattr(sanity,"reinit_last_action_layer",lambda *a:mutated.append(True))
    with pytest.raises(ValueError,match="drift"):
        sanity.main()
    assert mutated == []
    assert json.loads((tmp_path/"output.jsonl").read_text())["event"] == "evaluation_failure"


def test_displacement_preflights_source_even_when_rank_depth_not_in_grid(monkeypatch,tmp_path):
    import displacement as module
    args=writer_args(tmp_path)
    for key,value in dict(solver_steps="2,20",del_grid="0,5",reference_norm_min=0.0,
                          no_signal_filter=False,signal_filter="all",modality="vision",
                          selection="prefix",task="fixture",model="170m",target="logpi").items():
        setattr(args,key,value)
    row=dict(run_id="run",seed=42,episode=0,policy_call_idx=0,solver_steps=5)
    manifest=dict(run_id="run",configuration_sha256="config",configuration=dict(pipeline={},solver_steps=5))
    monkeypatch.setattr(module,"parse_args",lambda:args)
    monkeypatch.setattr(module,"replay_inputs",lambda *a,**kw:([row],manifest,file_hash(args.metrics)))
    loaded=[]
    def pipeline(*a,**kw):
        loaded.append(kw.get("solver_steps","source"))
        return {},{}
    monkeypatch.setattr(module,"replay_pipeline",pipeline)
    def drift(*a,**kw):
        raise ValueError("Source reference drift")
    monkeypatch.setattr(module,"preflight_source_replays",drift)
    with pytest.raises(ValueError,match="drift"):
        module.main()
    assert loaded == ["source"]
