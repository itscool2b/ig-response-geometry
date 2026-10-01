"""E05: declared Linear-weight arrangements and trained-response ranking transfer.

The null preserves each changed tensor's exact learned value multiset. It is
conditional on all retained tensors, not native initialization or an untrained
policy. This module never changes the already completed base-study artifacts.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path
import json
import time
from types import SimpleNamespace

import torch
from torch import nn

from experiment_io import canonical_json, file_hash, object_hash, strict_json, tensor_hash
from integrated_gradients import NonFiniteAttributionError
import paired_comparison as paired

NULL_KIND = "within_linear_weight_entry_permutation_v1"
TARGETS = ("Q", "L2")
MODALITIES = ("vision", "language", "state")
TIE_RULE = "stable_descending_then_index"
SCOPE = dict(null_kind=NULL_KIND, reference="own_function_self_reference", response="trained_probe_RMS",
             tie_rule=TIE_RULE, parameter_seed_unit="episode", completion_population="separate_joint_complete_episodes")


def implementation_hashes():
    root=Path(__file__).parent
    names=("weight_arrangement_control.py","paired_comparison.py","fp32_probe_cache.py","integrated_gradients.py",
        "rdt_sampling.py","per_step_attribution.py","checkpoint_contract.py","pipeline.py","faithfulness.py",
        "scripts/validate_downstream_precision.py","scripts/run_weight_arrangement_control.py")
    return {name:file_hash(root/name) for name in names}


def class_name(value):
    return type(value).__module__ + "." + type(value).__qualname__


def _members(runner):
    modules = dict(runner.named_modules(remove_duplicate=False))
    if len({id(value) for value in modules.values()}) != len(modules):
        raise ValueError("Shared module aliases are unsupported")
    if "model" not in modules:
        raise ValueError("The null requires a runner.model backbone")
    tensors = {}
    seen_objects, seen_storage = {}, {}
    for kind, items in (("parameter", runner.named_parameters(remove_duplicate=False)),
                        ("buffer", runner.named_buffers(remove_duplicate=False))):
        for name, value in items:
            if name in tensors or id(value) in seen_objects:
                raise ValueError("Shared tensor aliases are unsupported: " + name)
            if value.layout != torch.strided or not value.is_contiguous() or value.is_complex():
                raise ValueError("Unsupported tensor representation: " + name)
            if kind == "parameter" and type(value) is not nn.Parameter:
                raise ValueError("Special parameter subclasses are unsupported: " + name)
            seen_objects[id(value)] = name
            if value.device.type != "meta" and value.numel():
                storage = (str(value.device), value.untyped_storage().data_ptr())
                if storage in seen_storage:
                    raise ValueError("Shared tensor storage is unsupported: " + name)
                seen_storage[storage] = name
            tensors[name] = (kind, value)
    return modules, tensors


def describe_structure(runner):
    """Comparable structural lock, independent of device, values and training flags."""
    modules, tensors = _members(runner)
    for name, module in modules.items():
        if isinstance(module, nn.Linear) and type(module) is not nn.Linear:
            raise ValueError("Special Linear wrappers are unsupported: " + name)
        if hasattr(module, "parametrizations") or hasattr(module, "weight_orig"):
            raise ValueError("Parametrized or reparameterized modules are unsupported: " + name)
    descriptions = {}
    for name, (kind, value) in sorted(tensors.items()):
        owner, _, local = name.rpartition(".")
        module = modules[owner]
        permute = kind == "parameter" and owner.startswith("model.") and type(module) is nn.Linear and local == "weight"
        if owner == "model" and type(module) is nn.Linear and local == "weight":
            permute = True
        descriptions[name] = dict(kind=kind, owner_type=class_name(module), shape=list(value.shape),
            intervention="permute_flat_entries" if permute else "retain_exact")
    if not any(item["intervention"] == "permute_flat_entries" for item in descriptions.values()):
        raise ValueError("No backbone Linear weights match the declared null")
    return dict(schema_version=1, null_kind=NULL_KIND,
        modules={name:class_name(value) for name,value in sorted(modules.items())}, tensors=descriptions)


def structure_from_native_registry(registry, model):
    """Adapt the reviewed CPU structural shell, explicitly qualifying its root."""
    if (registry.get("kind") != "native_CPU_architecture_no_checkpoint_values_no_inference" or
            registry.get("upstream_commit") != "cd79363a1387e8f81c7724d070ef7e45fd23150f" or
            registry.get("cuda_initialized") is not False):
        raise ValueError("Unknown native structure qualification")
    candidates = [item for item in registry["models"] if item["model"] == model]
    if len(candidates) != 1:
        raise ValueError("Native model registry is missing or repeated")
    item = candidates[0]
    modules = {entry["name"]:entry["owner_class"] for entry in item["modules"]}
    if len(modules) != len(item["modules"]) or modules.get("") != "torch.nn.modules.module.Module":
        raise ValueError("Native registry does not describe the reviewed structural shell")
    modules[""] = "models.rdt_runner.RDTRunner"
    tensors = {}
    for kind, entries in (("parameter",item["parameters"]),("buffer",item["buffers"])):
        for entry in entries:
            name = entry["name"]
            if name in tensors or entry["intervention"] not in {"entry_permutation","retained_exact"}:
                raise ValueError("Repeated or unclassified native tensor")
            tensors[name] = dict(kind=kind, owner_type=entry["owner_class"],shape=entry["shape"],
                intervention="permute_flat_entries" if entry["intervention"] == "entry_permutation" else "retain_exact")
    permuted = sorted(name for name,value in tensors.items() if value["intervention"] == "permute_flat_entries")
    if permuted != item["permuted_names"] or len(permuted) != item["permuted_parameters"]:
        raise ValueError("Native registry permutation membership disagrees")
    return dict(schema_version=1,null_kind=NULL_KIND,modules=dict(sorted(modules.items())),tensors=dict(sorted(tensors.items())))


def multiset_hash(value):
    """SHA-256 of sorted FP32 bit patterns, including the distinction of signed zero."""
    if value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise ValueError("Permutation multisets require finite FP32 values")
    bits = value.detach().cpu().contiguous().view(torch.int32).flatten()
    return tensor_hash(torch.sort(bits).values)


def tensor_stream(draw, name):
    required = {"namespace", "master_seed", "stratum_id", "episode_id", "draw_index"}
    if set(draw) != required or any(not isinstance(draw[k], str) or not draw[k].strip()
                                   for k in ("namespace", "stratum_id", "episode_id")):
        raise ValueError("A complete independent episode/draw seed namespace is required")
    if any(type(draw[k]) is not int or draw[k] < 0 for k in ("master_seed", "draw_index")):
        raise ValueError("Draw seed and index must be nonnegative integers")
    identity = dict(null_kind=NULL_KIND, **draw, parameter_name=name)
    digest = object_hash(identity)
    return dict(identity=identity, identity_sha256=digest, seed=int(digest[:16],16),
                generator="torch.Generator(device=cpu)/torch.randperm", torch_version=torch.__version__)


class ArrangementSession:
    """Snapshot all runner tensors; verify each draw and unconditional restoration."""
    def __init__(self, runner, expected_structure):
        self.runner = runner
        self.structure = describe_structure(runner)
        if self.structure != expected_structure:
            raise ValueError("Runner module/parameter registry differs from the locked structure")
        _, members = _members(runner)
        self.tensors = {name:value for name,(_,value) in members.items()}
        for name,value in self.tensors.items():
            if value.device.type == "meta" or (value.is_floating_point() and value.dtype != torch.float32):
                raise ValueError("E05 requires a fully materialized FP32 runner: " + name)
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite trained checkpoint tensor: " + name)
        self.original = {name:value.detach().cpu().clone() for name,value in self.tensors.items()}
        self.hashes = {name:tensor_hash(value) for name,value in self.original.items()}
        self.permuted = sorted(name for name,item in self.structure["tensors"].items()
                               if item["intervention"] == "permute_flat_entries")
        self.multisets = {name:multiset_hash(self.original[name]) for name in self.permuted}
        self.registry = dict(structure=self.structure, structure_sha256=object_hash(self.structure),
            tensors={name:dict(**self.structure["tensors"][name], dtype=str(value.dtype),
                trained_sha256=self.hashes[name],
                trained_multiset_sha256=self.multisets.get(name)) for name,value in sorted(self.original.items())})
        self.active = False
        self.used_seed_identities = {}

    def verify(self, expected):
        if describe_structure(self.runner) != self.structure:
            raise ValueError("Runner structure changed during the null intervention")
        _, current = _members(self.runner)
        if any(current[name][1] is not value for name,value in self.tensors.items()):
            raise ValueError("Runner replaced a registered tensor object")
        for name,value in self.tensors.items():
            if tensor_hash(value) != expected[name]:
                raise ValueError("Runner tensor changed outside its declared assignment: " + name)

    @contextmanager
    def draw(self, identity):
        if self.active:
            raise ValueError("Nested parameter interventions are unsupported")
        self.verify(self.hashes)
        self.active = True
        expected = dict(self.hashes)
        report = dict(null_kind=NULL_KIND, draw=deepcopy(identity), registry_sha256=object_hash(self.registry),
            tensors={}, preserved_sha256={name:digest for name,digest in self.hashes.items() if name not in self.permuted})
        try:
            for name in self.permuted:
                stream = tensor_stream(identity, name)
                seed = stream["seed"]
                previous = self.used_seed_identities.get(seed)
                if previous is not None and previous != stream["identity_sha256"]:
                    raise ValueError("Distinct parameter streams collide in their CPU seed")
                self.used_seed_identities[seed] = stream["identity_sha256"]
                generator = torch.Generator(device="cpu").manual_seed(seed)
                original = self.original[name]
                permutation = torch.randperm(original.numel(), generator=generator)
                child = original.flatten()[permutation].reshape(original.shape)
                child_multiset = multiset_hash(child)
                if child_multiset != self.multisets[name]:
                    raise ValueError("Entry permutation changed a learned tensor multiset")
                with torch.no_grad():
                    self.tensors[name].copy_(child)
                expected[name] = tensor_hash(child)
                report["tensors"][name] = dict(parent_sha256=self.hashes[name], child_sha256=expected[name],
                    parent_multiset_sha256=self.multisets[name], child_multiset_sha256=child_multiset,
                    permutation_sha256=tensor_hash(permutation), stream=stream,
                    changed=expected[name] != self.hashes[name])
                del permutation, child
            self.verify(expected)
            report["parameter_draw_sha256"] = object_hash(report)
            yield report
            self.verify(expected)
        finally:
            with torch.no_grad():
                for name,value in self.tensors.items():
                    value.copy_(self.original[name])
            self.active = False
            self.verify(self.hashes)


def own_reference_context(trained_context):
    """Use the current runner's function but preserve the trained cached inputs/noise."""
    ctx = dict(trained_context)
    with torch.no_grad():
        reference = ctx["seeded_conditional_sample"](ctx["lang_adapted"],ctx["img_adapted"],ctx["state_traj_actual"])
    ctx["ref_action"] = reference.detach()
    if reference.dtype != torch.float32 or reference.shape != trained_context["ref_action"].shape:
        raise ValueError("Randomized reference violates the trained probe dtype/shape contract")
    if not torch.isfinite(reference).all():
        raise paired.attach_probe_failure(FloatingPointError("Nonfinite randomized own-function reference"),ctx,"randomized_self_reference")
    return ctx


def own_rankings(ctx, modality, settings, *, context_id):
    """Return own-reference maps while the declared parameter draw is active."""
    actual, baseline, indices, axis, forward = paired.modality_inputs(ctx,modality)
    with torch.no_grad():
        replay = forward(actual)
    if replay.dtype != torch.float32 or replay.shape != ctx["ref_action"].shape:
        raise ValueError("Randomized actual endpoint dtype/shape differs")
    if not torch.equal(replay,ctx["ref_action"]):
        error = FloatingPointError("Randomized own-reference endpoint did not repeat exactly")
        error.raw_tensors = {"reference":ctx["ref_action"],"actual_replay":replay}
        raise error
    rankings, coordinates = {}, {"randomized_reference":ctx["ref_action"].detach().cpu()}
    for target in TARGETS:
        label = "permuted_" + target + "_IG"
        try:
            result, gradient = paired.integrate_with_path_gradient(
                lambda value:paired.common_scores(forward(value),ctx["ref_action"])[target],
                actual,baseline,settings[target],context=dict(context_id=context_id,modality=modality,target=target,null_kind=NULL_KIND))
            rankings[label] = paired.ranked(paired.grouped_scores(result.attributions,indices,axis),indices)
            rankings[label].update(diagnostics=result.diagnostics(),
                attribution_sha256=tensor_hash(result.attributions), path_gradient_sha256=tensor_hash(gradient),
                own_reference_sha256=tensor_hash(ctx["ref_action"]))
            coordinates[target+"_attribution"] = result.attributions.detach().cpu()
            coordinates[target+"_path_gradient"] = gradient.detach().cpu()
        except (NonFiniteAttributionError,FloatingPointError) as error:
            rankings[label] = dict(status="numerical_failure",failure_kind="randomized_integration",
                                  reason=str(error),diagnostics=getattr(error,"diagnostics",{}))
    return rankings, coordinates, dict(input_sha256=tensor_hash(actual),baseline_sha256=tensor_hash(baseline),
        input_shape=list(actual.shape),eligible_indices=indices,axis=axis,
        randomized_reference_sha256=tensor_hash(ctx["ref_action"]),noise_sha256=tensor_hash(ctx["initial_noise"]))


def transfer_on_trained(ctx, modality, randomized_rankings, base_row, grid, denominator_min):
    """The caller must restore the trained state before invoking this function."""
    actual,baseline,indices,axis,forward = paired.modality_inputs(ctx,modality)
    base = base_row["results"]
    if (ctx["ref_action"][...,paired.MANISKILL_INDICES].cpu().tolist() != base["active_reference_action"] or
            tensor_hash(actual) != base_row["input_sha256"] or tensor_hash(baseline) != base_row["baseline_sha256"] or
            tensor_hash(ctx["initial_noise"]) != base_row["noise_sha256"]):
        raise ValueError("Trained transfer context differs from the completed base probe")
    with torch.no_grad():
        replay = forward(actual)
    if not torch.equal(replay,ctx["ref_action"]):
        raise ValueError("Restored trained actual action differs from its completed base identity")
    rankings = {"trained_"+target+"_IG":deepcopy(base["rankings"][target+"_IG"]) for target in TARGETS}
    rankings.update(randomized_rankings)
    results = paired.evaluate_rankings(forward,ctx["ref_action"],actual,baseline,indices,axis,rankings,grid,denominator_min)
    verify_trained_anchors(results,base)
    return results


def verify_trained_anchors(results, base):
    if results["active_reference_sha256"] != base["active_reference_sha256"] or results["realized_fractions"] != base["realized_fractions"]:
        raise ValueError("Trained transfer reference or grid differs from base")
    for target in TARGETS:
        left,right = results["rankings"]["trained_"+target+"_IG"],base["rankings"][target+"_IG"]
        if left["status"] != right["status"]:
            raise ValueError("Trained ranking outcome differs from base")
        if left["status"] == "defined":
            if left["order"] != right["order"] or left["curves"] != right["curves"]:
                raise ValueError("Trained ranking or response changed across transfer draws")
    for key in set(results["response_table"]) & set(base["response_table"]):
        left,right = results["response_table"][key],base["response_table"][key]
        if left != right:
            raise ValueError("A reused trained intervention response changed")


def _digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def validate_registry(registry):
    structure=registry["structure"]
    if registry["structure_sha256"] != object_hash(structure) or structure.get("null_kind") != NULL_KIND:
        raise ValueError("Parameter registry structure hash or null differs")
    if set(registry["tensors"]) != set(structure["tensors"]):
        raise ValueError("Parameter registry tensor membership differs")
    for name,spec in structure["tensors"].items():
        item=registry["tensors"][name]
        if any(item.get(k)!=v for k,v in spec.items()) or not _digest(item["trained_sha256"]):
            raise ValueError("Parameter registry trained identity differs")
        owner,_,local=name.rpartition(".")
        if structure["modules"].get(owner)!=spec["owner_type"] or spec["kind"] not in {"parameter","buffer"}:
            raise ValueError("Unclassified parameter registry owner")
        permute=(owner=="model" or owner.startswith("model.")) and local=="weight" and spec["kind"]=="parameter" and spec["owner_type"]=="torch.nn.modules.linear.Linear"
        if spec["intervention"] != ("permute_flat_entries" if permute else "retain_exact"):
            raise ValueError("Registry changes the declared Linear-weight null")
        if permute and (item["dtype"]!="torch.float32" or not _digest(item["trained_multiset_sha256"])):
            raise ValueError("Changed weights need exact FP32 multiset identities")
    return registry


def validate_draw_record(record,registry,identity):
    validate_registry(registry)
    if (record["null_kind"]!=NULL_KIND or record["draw"]!=identity or
            record["registry_sha256"]!=object_hash(registry) or
            record["parameter_draw_sha256"]!=object_hash({k:v for k,v in record.items() if k!="parameter_draw_sha256"})):
        raise ValueError("Parameter draw identity differs")
    changed={name for name,value in registry["tensors"].items() if value["intervention"]=="permute_flat_entries"}
    retained={name:value["trained_sha256"] for name,value in registry["tensors"].items() if name not in changed}
    if set(record["tensors"])!=changed or record["preserved_sha256"]!=retained:
        raise ValueError("Changed or retained draw tensor roster differs")
    seeds=set()
    for name,value in record["tensors"].items():
        parent=registry["tensors"][name]
        stream=tensor_stream(identity,name)
        if any(value["stream"].get(k)!=v for k,v in stream.items() if k!="torch_version"):
            raise ValueError("Parameter draw CPU stream differs")
        if value["stream"]["seed"] in seeds:
            raise ValueError("Parameter stream seed collision")
        seeds.add(value["stream"]["seed"])
        if (value["parent_sha256"]!=parent["trained_sha256"] or
                value["parent_multiset_sha256"]!=parent["trained_multiset_sha256"] or
                value["child_multiset_sha256"]!=parent["trained_multiset_sha256"] or
                not _digest(value["child_sha256"]) or not _digest(value["permutation_sha256"]) or
                value["changed"]!=(value["parent_sha256"]!=value["child_sha256"])):
            raise ValueError("Parameter draw does not preserve its declared trained multiset")


def _auxiliary(path,descriptor,completion,*,tensors=False):
    name=descriptor["file"]
    if not isinstance(name,str) or Path(name).name!=name or not name.startswith(path.name+"."):
        raise ValueError("E05 auxiliary must be a local registered artifact")
    suffix=name[len(path.name)+1:]
    actual=path.with_name(name)
    if completion["auxiliary_sha256"].get(suffix)!=descriptor["sha256"] or file_hash(actual)!=descriptor["sha256"]:
        raise ValueError("E05 auxiliary hash differs")
    return torch.load(actual,map_location="cpu",weights_only=True) if tensors else strict_json(actual)


def _verify_coordinates(row,raw,protocol,base):
    identity=row["coordinate_identity"]
    reference=raw["randomized_reference"]
    if reference.dtype!=torch.float32 or reference.shape!=(1,64,128) or not torch.isfinite(reference).all():
        raise ValueError("Invalid randomized own-reference coordinate artifact")
    if tensor_hash(reference)!=identity["randomized_reference_sha256"] or row["randomized_reference_sha256"]!=identity["randomized_reference_sha256"]:
        raise ValueError("Randomized own-reference hash differs")
    if (identity["axis"]!=(2 if row["modality"]=="state" else 1) or
            identity["eligible_indices"]!=base["results"]["eligible_indices"] or
            identity["input_sha256"]!=base["input_sha256"] or identity["baseline_sha256"]!=base["baseline_sha256"] or
            identity["noise_sha256"]!=base["noise_sha256"]):
        raise ValueError("Randomized maps used another input, baseline or noise")
    expected={"randomized_reference"}
    for target in TARGETS:
        ranking=row["results"]["rankings"]["permuted_"+target+"_IG"]
        if ranking["status"]=="numerical_failure":
            continue
        if ranking["status"]!="defined":
            raise ValueError("Unknown randomized map outcome")
        names=[target+"_attribution",target+"_path_gradient"]
        expected.update(names)
        attr,gradient=(raw[name] for name in names)
        dtype=getattr(torch,protocol["numerics"][row["modality"]][target]["arithmetic_dtype"])
        if any(value.dtype!=dtype or list(value.shape)!=identity["input_shape"] or not torch.isfinite(value).all()
               for value in (attr,gradient)):
            raise ValueError("Invalid randomized attribution coordinate array")
        if tensor_hash(attr)!=ranking["attribution_sha256"] or tensor_hash(gradient)!=ranking["path_gradient_sha256"]:
            raise ValueError("Randomized attribution coordinate hash differs")
        expected_ranking=paired.ranked(paired.grouped_scores(attr,identity["eligible_indices"],identity["axis"]),identity["eligible_indices"])
        if any(ranking.get(k)!=v for k,v in expected_ranking.items()) or ranking["own_reference_sha256"]!=tensor_hash(reference):
            raise ValueError("Randomized ranking does not follow its own-reference coordinates")
    if set(raw)!=expected:
        raise ValueError("Unexpected or missing coordinate artifact membership")


def load_completed_control(path, *, base_path):
    """Authenticate full bank x modality x draw membership and the actual base artifact."""
    path,base_path=Path(path),Path(base_path)
    base_rows,base_manifest,base_completion=paired.load_completed_study(base_path)
    completion=strict_json(str(path)+".completion.json")
    manifest=strict_json(str(path)+".manifest.json")
    if (completion["status"]!="complete" or completion["output_sha256"]!=file_hash(path) or
            completion["manifest_sha256"]!=file_hash(str(path)+".manifest.json") or
            completion["evaluation_id"]!=manifest["evaluation_id"] or manifest["study"]!="weight_arrangement_control"):
        raise ValueError("Incomplete or changed E05 artifact")
    config=manifest["configuration"]; protocol=config["protocol"]; bank=config["bank"]
    for name in ("protocol_sha256","bank_sha256"):
        if completion[name]!=config[name]:
            raise ValueError("E05 protocol or bank hash binding differs")
    if (any(protocol.get(k)!=v for k,v in SCOPE.items()) or type(protocol.get("parameter_draws")) is not int or
            protocol["parameter_draws"]<2 or protocol["stage"]!=base_manifest["configuration"]["protocol"]["stage"]):
        raise ValueError("E05 inferential stage or null contract differs")
    bindings=dict(base_metrics_sha256=file_hash(base_path),base_manifest_sha256=file_hash(str(base_path)+".manifest.json"),
        base_completion_sha256=file_hash(str(base_path)+".completion.json"),base_protocol_sha256=base_manifest["configuration"]["protocol_sha256"])
    if any(config.get(k)!=v for k,v in bindings.items()) or bank!=base_manifest["configuration"]["bank"]:
        raise ValueError("E05 does not bind the completed base study")
    if any(manifest[name]!=base_manifest[name] for name in ("source_metrics_sha256","source_run_id","source_configuration_sha256")):
        raise ValueError("E05 and base source identities differ")
    for suffix,digest in completion["auxiliary_sha256"].items():
        if Path(suffix).name!=suffix or file_hash(path.with_name(path.name+"."+suffix))!=digest:
            raise ValueError("Registered E05 auxiliary changed")
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result: raise ValueError("Duplicate E05 JSON key")
            result[key]=value
        return result
    rows=[json.loads(line,object_pairs_hook=unique) for line in path.read_text(encoding="utf-8").splitlines()]
    canonical_json(rows)
    expected={(entry["episode_id"],entry["policy_call_idx"],modality,index):entry["context_id"]
              for entry in bank["contexts"] for modality in MODALITIES for index in range(protocol["parameter_draws"])}
    if len(rows)!=len(expected) or any(value!=len(rows) for value in (manifest["expected_rows"],completion["expected_rows"],completion["completed_rows"])):
        raise ValueError("E05 missing planned context/modality/draw outcomes")
    base_lookup={(r["episode_id"],r["policy_call_idx"],r["modality"]):r for r in base_rows}
    registry=validate_registry(config["parameter_registry"])
    observed,draw_ids,reference_ids,used_seeds={}, {}, {}, {}
    for row in rows:
        key=row["episode_id"],row["policy_call_idx"],row["modality"],row["draw_index"]
        if key in observed or row["evaluation_id"]!=manifest["evaluation_id"]:
            raise ValueError("Repeated or foreign E05 result")
        observed[key]=row["source_context_id"]
        base=base_lookup[key[:3]]
        if row["base_row_sha256"]!=object_hash(base):
            raise ValueError("E05 row no longer binds its base outcome")
        if any(row.get(name)!=base.get(name) for name in ("episode","seed","source_context_id","source_attr_sha256")):
            raise ValueError("E05 row source membership differs from base")
        draw_key=row["episode_id"],row["draw_index"]
        record=_auxiliary(path,row["parameter_draw_artifact"],completion)
        identity=dict(namespace=protocol["parameter_seed_namespace"],master_seed=protocol["parameter_master_seed"],
            stratum_id=protocol["stratum_id"],episode_id=row["episode_id"],draw_index=row["draw_index"])
        validate_draw_record(record,registry,identity)
        if row["parameter_draw_sha256"]!=record["parameter_draw_sha256"]:
            raise ValueError("E05 row parameter draw differs")
        if draw_key in draw_ids and draw_ids[draw_key]!=row["parameter_draw_sha256"]:
            raise ValueError("Parameter draw changed across calls or modalities")
        draw_ids[draw_key]=row["parameter_draw_sha256"]
        restoration_name="restoration-"+object_hash(identity)+".json"
        restoration=_auxiliary(path,dict(file=path.name+"."+restoration_name,
            sha256=completion["auxiliary_sha256"].get(restoration_name)),completion)
        if (restoration.get("parameter_draw_sha256")!=record["parameter_draw_sha256"] or
                restoration.get("registry_sha256")!=object_hash(registry) or
                restoration.get("all_trained_tensor_hashes_verified") is not True or
                restoration.get("pre_intervention_reference_replay_verified") is not True):
            raise ValueError("Original trained-state restoration/replay is not authenticated")
        for item in record["tensors"].values():
            seed,digest=item["stream"]["seed"],item["stream"]["identity_sha256"]
            if seed in used_seeds and used_seeds[seed]!=digest:
                raise ValueError("E05 independent parameter stream seeds collide")
            used_seeds[seed]=digest
        if row["status"]=="evaluated":
            if base["status"]!="evaluated": raise ValueError("Unavailable base outcome was silently replaced")
            if set(row["results"]["rankings"])!={prefix+target+"_IG" for prefix in ("trained_","permuted_") for target in TARGETS}:
                raise ValueError("Unexpected E05 ranking membership")
            paired.authenticate_action_results(row["results"])
            verify_trained_anchors(row["results"],base["results"])
            raw=_auxiliary(path,row["coordinate_artifact"],completion,tensors=True)
            _verify_coordinates(row,raw,protocol,base)
            reference_key=row["episode_id"],row["policy_call_idx"],row["draw_index"]
            reference=row["randomized_reference_sha256"]
            if reference_key in reference_ids and reference_ids[reference_key]!=reference:
                raise ValueError("One randomized context has multiple own-function references")
            reference_ids[reference_key]=reference
        elif row["status"]!="numerical_failure" or not row.get("failure_kind"):
            raise ValueError("Unknown or unclassified E05 failure")
        if row.get("failure_artifact") is not None:
            _auxiliary(path,row["failure_artifact"],completion,tensors=True)
    if observed!=expected or len(set(draw_ids.values()))!=len(draw_ids):
        raise ValueError("E05 population or independent draw membership differs")
    return rows,manifest,completion


def validate_run_protocol(protocol, gate, base_manifest, native_digest):
    import fp32_probe_cache
    base=base_manifest["configuration"]
    if any(protocol.get(k)!=v for k,v in SCOPE.items()):
        raise ValueError("E05 null, reference, response, tie or population contract differs")
    if protocol.get("stage") not in {"variance_only_pilot","confirmatory_locked"} or protocol["stage"]!=base["protocol"]["stage"]:
        raise ValueError("E05 stage must match the completed base stage")
    if type(protocol.get("parameter_draws")) is not int or protocol["parameter_draws"]<2:
        raise ValueError("An explicit variance-planned parameter_draws >= 2 is required")
    if (not _digest(protocol.get("global_design_sha256")) or
            protocol["global_design_sha256"]!=base["protocol"].get("global_design_sha256")):
        raise ValueError("E05 and base must bind the same prospective global design")
    tensor_stream(dict(namespace=protocol["parameter_seed_namespace"],master_seed=protocol["parameter_master_seed"],
        stratum_id=protocol["stratum_id"],episode_id="validation-placeholder",draw_index=0),"validation-placeholder")
    if protocol.get("failure_policy")!="retain_every_planned_draw_no_replacement":
        raise ValueError("All planned draws require a nonreplacement failure policy")
    if protocol.get("forward_precision")!=paired.FP32_PROBE or base["protocol"].get("forward_precision")!=paired.FP32_PROBE:
        raise ValueError("E05 requires the exact offline FP32 probe")
    if protocol["runtime_settings"]!=fp32_probe_cache.strict_candidate_settings() or protocol["runtime_settings"]!=base["protocol"].get("runtime_settings"):
        raise ValueError("E05 runtime differs from the trained probe")
    if (protocol["native_registry_sha256"]!=native_digest or gate.get("native_registry_sha256")!=native_digest or
            gate.get("status")!="approved_for_weight_arrangement_transfer" or gate.get("independent_holdout_complete") is not True or
            gate.get("null_scope")!=SCOPE or gate.get("weight_arrangement_control_sha256")!=file_hash(__file__)):
        raise ValueError("A separate randomized-function numerical gate is required")
    if gate.get("source_sha256")!=implementation_hashes() or gate.get("torch_version")!=torch.__version__:
        raise ValueError("Randomized numerical gate does not bind the running helpers and permutation implementation")
    if not all(isinstance(gate.get(k),list) and gate[k] and all(_digest(x) for x in gate[k])
               for k in ("selection_report_sha256","heldout_report_sha256")):
        raise ValueError("Randomized numerical selection and independent heldout evidence must be bound")
    if set(gate["selection_report_sha256"]) & set(gate["heldout_report_sha256"]):
        raise ValueError("Selection reports cannot also be independent heldout reports")
    if protocol["runtime_settings"]!=gate.get("runtime_settings") or protocol["numerics"]!=gate.get("approved_numerics"):
        raise ValueError("Randomized numerical settings differ from the separate gate")
    if set(protocol["numerics"])!=set(MODALITIES):
        raise ValueError("All three modalities must be planned")
    for spec in protocol["numerics"].values():
        if set(spec)!=set(TARGETS): raise ValueError("Both own-reference targets must be planned")
        for settings in spec.values():
            if (set(settings)!={"m","quadrature","arithmetic_dtype"} or type(settings["m"]) is not int or settings["m"]<1 or
                    settings["quadrature"]!="trapezoid" or settings["arithmetic_dtype"] not in {"float32","float64"}):
                raise ValueError("Explicit gate-approved IG arithmetic is required")
    paired.validate_gate_population(gate,base["bank"])


def run(args, startup):
    """Execute a separate, fully gated E05 shard; no implicit study sizing."""
    import fp32_probe_cache as cache_helper
    from faithfulness import authenticated_source, load_sidecar, replay_pipeline
    from scripts.validate_downstream_precision import convert_downstream_to_fp32,fp32_math_settings
    if not startup or not startup.get("torch_not_preimported"):
        raise RuntimeError("E05 production requires a fresh process before torch import")
    protocol=strict_json(args.protocol)
    if file_hash(args.protocol)!=args.protocol_sha256:
        raise ValueError("E05 protocol differs from its trusted hash")
    gate=strict_json(args.e05_gate); native=strict_json(args.native_registry)
    if file_hash(args.e05_gate)!=protocol["e05_gate_sha256"]:
        raise ValueError("E05 numerical gate hash differs")
    base_rows,base_manifest,_=paired.load_completed_study(args.base)
    base_bindings=dict(base_metrics_sha256=file_hash(args.base),base_manifest_sha256=file_hash(str(args.base)+".manifest.json"),
        base_completion_sha256=file_hash(str(args.base)+".completion.json"),base_protocol_sha256=base_manifest["configuration"]["protocol_sha256"])
    if any(protocol.get(k)!=v for k,v in base_bindings.items()):
        raise ValueError("E05 protocol names another completed base artifact")
    native_digest=file_hash(args.native_registry)
    validate_run_protocol(protocol,gate,base_manifest,native_digest)
    bank=base_manifest["configuration"]["bank"]
    source_rows,source_manifest=authenticated_source(args.metrics)
    digest=file_hash(args.metrics)
    if digest!=base_manifest["source_metrics_sha256"] or paired.make_bank(args.metrics,bank["selection"])!=bank:
        raise ValueError("E05 source or complete context bank differs from base")
    import paired_study_analysis as study
    study._selection(bank)
    selected=paired.bank_rows(bank,source_rows,source_manifest,digest)
    prepared_hash=base_manifest["configuration"]["prepared_completion_sha256"]
    if protocol["prepared_completion_sha256"]!=prepared_hash:
        raise ValueError("E05 requires the exact base prepared cache")
    preparation=cache_helper.verify_preparation(args.prepared,prepared_hash)
    applied=cache_helper.apply_settings(protocol["runtime_settings"])
    replay_args=SimpleNamespace(model=bank["model"],task=bank["task"],target="logpi",no_checkpoint=False,
        checkpoint_mode=None,checkpoint_path=args.checkpoint_path,model_revision=None,vision_revision=None,lang_dir=args.lang_dir)
    pipe,_=replay_pipeline(replay_args,source_manifest)
    conversion=convert_downstream_to_fp32(pipe["runner"])
    expected_structure=structure_from_native_registry(native,bank["model"])
    session=ArrangementSession(pipe["runner"],expected_structure)
    if gate.get("structure_sha256")!=object_hash(expected_structure):
        raise ValueError("Separate randomized gate used another parameter structure")
    base_lookup={(r["episode_id"],r["policy_call_idx"],r["modality"]):r for r in base_rows}
    episodes=defaultdict(list)
    for entry in bank["contexts"]: episodes[entry["episode_id"]].append(entry)
    source_hashes=implementation_hashes()
    inputs={Path(args.protocol):args.protocol_sha256,Path(args.e05_gate):protocol["e05_gate_sha256"],
        Path(args.native_registry):native_digest,Path(args.base):base_bindings["base_metrics_sha256"],
        Path(str(args.base)+".completion.json"):base_bindings["base_completion_sha256"],Path(args.metrics):digest}
    configuration=dict(protocol=protocol,protocol_sha256=args.protocol_sha256,bank=bank,
        bank_sha256=base_manifest["configuration"]["bank_sha256"],e05_gate=gate,e05_gate_sha256=protocol["e05_gate_sha256"],
        native_registry_sha256=native_digest,parameter_registry=session.registry,
        native_registry_root_qualification="Only the inspected torch.nn.Module shell root is mapped to pinned models.rdt_runner.RDTRunner.",
        source_sha256=source_hashes,prepared_completion_sha256=prepared_hash,**base_bindings)
    writer_args=SimpleNamespace(out=args.out,metrics=args.metrics,limit=None)
    device=next(pipe["runner"].parameters()).device
    def make_context(entry):
        row=selected[entry["episode"],entry["policy_call_idx"]]
        payload,_=load_sidecar(row,args.metrics,source_manifest)
        cache,record=cache_helper.load_prepared_context(args.prepared,preparation,entry,source_manifest,row,payload)
        source={key:value.to(device) for key,value in cache.items()}
        ctx,identity=paired.make_fp32_probe_context(source,pipe["runner"])
        identity["prepared_cache"]=record["cache_file"]
        return ctx,identity
    with fp32_math_settings(), paired.PairedEvaluationWriter(writer_args,source_manifest,digest,"weight_arrangement_control",configuration,
            len(bank["contexts"])*len(MODALITIES)*protocol["parameter_draws"]) as writer:
        writer.write_auxiliary("conversion.json",dict(conversion=conversion,runtime_settings=applied,startup=startup))
        for episode_id,entries in sorted(episodes.items()):
            entries=sorted(entries,key=lambda entry:entry["policy_call_idx"])
            for draw_index in range(protocol["parameter_draws"]):
                started=time.perf_counter()
                identity=dict(namespace=protocol["parameter_seed_namespace"],master_seed=protocol["parameter_master_seed"],
                    stratum_id=protocol["stratum_id"],episode_id=episode_id,draw_index=draw_index)
                # Require exact trained replay before the intervention, not only after restoration.
                for entry in entries:
                    references=[base_lookup[episode_id,entry["policy_call_idx"],m] for m in MODALITIES
                                if base_lookup[episode_id,entry["policy_call_idx"],m]["status"]=="evaluated"]
                    if references:
                        ctx,_=make_context(entry)
                        if any(tensor_hash(ctx["ref_action"])!=row["reference_sha256"] for row in references):
                            raise ValueError("Pre-intervention trained replay differs from completed base")
                        del ctx
                pending={}
                with session.draw(identity) as draw_record:
                    name="draw-"+object_hash(identity)+".json"
                    draw_artifact=dict(file=writer.path.name+"."+name,sha256=writer.write_auxiliary(name,draw_record))
                    for entry in entries:
                        available=any(base_lookup[episode_id,entry["policy_call_idx"],m]["status"]=="evaluated" for m in MODALITIES)
                        if not available:
                            continue
                        try:
                            ctx,_=make_context(entry)
                        except (NonFiniteAttributionError,FloatingPointError) as error:
                            raw=getattr(error,"raw_tensors",{})
                            artifact=writer.write_tensor_auxiliary("reference-failure-"+entry["context_id"]+f"-{draw_index}.pt",raw) if raw else None
                            for modality in MODALITIES:
                                pending[entry["context_id"],modality]=dict(status="numerical_failure",failure_kind="randomized_self_reference",
                                    reason=str(error),diagnostics=getattr(error,"diagnostics",{}),failure_artifact=artifact)
                            continue
                        for modality in MODALITIES:
                            start=time.perf_counter()
                            try:
                                rankings,raw,coordinates=own_rankings(ctx,modality,protocol["numerics"][modality],context_id=entry["context_id"])
                                artifact=writer.write_tensor_auxiliary("coordinates-"+entry["context_id"]+f"-{modality}-{draw_index}.pt",raw)
                                pending[entry["context_id"],modality]=dict(status="evaluated",randomized_rankings=rankings,
                                    coordinate_artifact=artifact,coordinate_identity=coordinates,
                                    randomized_reference_sha256=coordinates["randomized_reference_sha256"])
                            except (NonFiniteAttributionError,FloatingPointError) as error:
                                raw=getattr(error,"raw_tensors",{})
                                artifact=writer.write_tensor_auxiliary("modality-failure-"+entry["context_id"]+f"-{modality}-{draw_index}.pt",raw) if raw else None
                                pending[entry["context_id"],modality]=dict(status="numerical_failure",failure_kind="randomized_modality",
                                    reason=str(error),diagnostics=getattr(error,"diagnostics",{}),failure_artifact=artifact)
                            pending[entry["context_id"],modality]["ranking_wall_seconds"]=time.perf_counter()-start
                        del ctx
                # The context manager has restored and hashed every trained runner tensor.
                for entry in entries:
                    ctx=None
                    for modality in MODALITIES:
                        base=base_lookup[episode_id,entry["policy_call_idx"],modality]
                        writer.set_context(selected[entry["episode"],entry["policy_call_idx"]])
                        common=dict(event="weight_arrangement_control",episode_id=episode_id,modality=modality,draw_index=draw_index,
                            reset_seed=entry["reset_seed"],base_row_sha256=object_hash(base),parameter_draw_sha256=draw_record["parameter_draw_sha256"],
                            parameter_draw_artifact=draw_artifact)
                        if base["status"]!="evaluated":
                            writer.write(dict(**common,status="numerical_failure",failure_kind="base_outcome_unavailable",
                                base_failure_kind=base.get("failure_kind"),results=None))
                            continue
                        outcome=pending[entry["context_id"],modality]
                        if outcome["status"]=="evaluated":
                            if ctx is None:
                                ctx,_=make_context(entry)
                            start=time.perf_counter()
                            rankings=outcome.pop("randomized_rankings")
                            results=transfer_on_trained(ctx,modality,rankings,base,
                                base_manifest["configuration"]["protocol"]["grid_percent"],
                                base_manifest["configuration"]["protocol"]["denominator_min"])
                            outcome.update(results=results,trained_response_wall_seconds=time.perf_counter()-start)
                        else:
                            outcome["results"]=None
                        writer.write(dict(**common,**outcome))
                    del ctx
                writer.write_auxiliary("restoration-"+object_hash(identity)+".json",dict(parameter_draw_sha256=draw_record["parameter_draw_sha256"],
                    registry_sha256=object_hash(session.registry),all_trained_tensor_hashes_verified=True,
                    pre_intervention_reference_replay_verified=True,draw_wall_seconds=time.perf_counter()-started))
        session.verify(session.hashes)
        cache_helper.verify_preparation(args.prepared,prepared_hash)
        paired.load_completed_study(args.base)
        authenticated_source(args.metrics)
        if cache_helper.runtime_settings()!=applied:
            raise ValueError("E05 runtime changed during execution")
        for path,expected in {**inputs,**{Path(__file__).parent/p:h for p,h in source_hashes.items()}}.items():
            if file_hash(path)!=expected:
                raise ValueError("E05 input or forward helper changed during execution")
