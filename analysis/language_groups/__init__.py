"""How the saved curves counted language feature groups (CPU only).

Tokenizes the committed task instructions exactly as encode_task_lang.py does,
tabulates the selected group counts under the historical faithfulness.py rule
k = max(1, int(round(G*p/100))) (all G groups once k >= G) and the current
no-floor rule, checks the saved curves for the equal 1% and 5% responses the
historical rule implies, and records the git-history evidence for the rule.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
from pathlib import Path
import subprocess

from analysis.paired_rescoring import analyze as paired
from analysis.paired_rescoring import curve_stats as cs
from analysis.revision.core import IntegrityError, sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results" / "2026-10-23-v1"
CORE_OUTPUTS = ("protocol.json", "tokenization.json", "group_counts.csv", "saved_curve_check.csv", "git_history.json")
AUTHORS_OUTPUTS = ("authors_embeddings.json", "run_language_inputs.csv")
OUTPUTS = CORE_OUTPUTS + AUTHORS_OUTPUTS
SOURCES = ("analysis/language_groups/__init__.py", "analysis/language_groups/__main__.py",
           "analysis/paired_rescoring/curve_stats/__init__.py", "analysis/paired_rescoring/analyze.py",
           "analysis/revision/core.py", "analysis/revision/analyze.py", "analysis/revision/verify.py")


def load_protocol(root=ROOT):
    blob = (Path(root) / "analysis/language_groups/protocol.json").read_bytes()
    return blob, json.loads(blob)


# --- selection rules --------------------------------------------------------

def historical_k(groups, percent):
    """faithfulness.py at afd1249, b887a96 and e4ad8bf; 0 and 100 are endpoint nodes."""
    if percent == 0:
        return 0
    if percent == 100:
        return groups
    k = max(1, int(round(groups * percent / 100.0)))
    return groups if k >= groups else k


def current_k(groups, percent):
    """faithfulness.py from c9eacfb onward: no floor; argsort[:k] caps at all groups."""
    if percent == 0:
        return 0
    if percent == 100:
        return groups
    return min(groups, int(round(groups * percent / 100.0)))


def count_rows(label, groups, grid):
    rows = []
    for percent in grid:
        interior = percent not in (0, 100)
        raw = groups * percent / 100.0
        old, new = historical_k(groups, percent), current_k(groups, percent)
        rows.append(dict(groups_label=label, G=groups, nominal_percent=percent,
                         node="interior" if interior else "endpoint_not_masked_by_rule",
                         G_times_p_over_100=raw, half_way_tie=interior and raw - int(raw) == 0.5,
                         historical_k=old, historical_fraction=old / groups,
                         historical_fraction_exact=f"{old}/{groups}",
                         historical_floor_active=interior and int(round(raw)) < 1,
                         current_k=new, current_fraction=new / groups, current_fraction_exact=f"{new}/{groups}",
                         rules_agree=old == new))
    return rows


# --- tokenization -----------------------------------------------------------

def encoder_facts(source):
    """task2lang literal and whether the code appends T5 EOS id 1 after sp.Encode."""
    tree = ast.parse(source)
    instructions = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "task2lang" for t in node.targets):
            instructions = ast.literal_eval(node.value)
    appends_eos = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "append"
                      and isinstance(n.func.value, ast.Name) and n.func.value.id == "token_ids"
                      and len(n.args) == 1 and isinstance(n.args[0], ast.Constant) and n.args[0].value == 1
                      for n in ast.walk(tree))
    uses_encode = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "Encode"
                      for n in ast.walk(tree))
    if instructions is None:
        raise IntegrityError("encode_task_lang.py has no task2lang literal")
    return dict(instructions=instructions, appends_eos=appends_eos, uses_sp_encode=uses_encode)


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout


def tokenize(root, protocol):
    import sentencepiece as spm
    spec = protocol["tokenizer"]
    path = Path(root) / spec["vendored_path"]
    blob = path.read_bytes()
    if sha256(blob) != spec["sha256"] or len(blob) != spec["bytes"]:
        raise IntegrityError("Vendored tokenizer differs from the pinned file")
    processor = spm.SentencePieceProcessor(model_file=str(path))
    encoder_path = Path(root) / protocol["encoder"]["path"]
    facts = encoder_facts(encoder_path.read_text(encoding="utf-8"))
    if not (facts["appends_eos"] and facts["uses_sp_encode"]):
        raise IntegrityError("encode_task_lang.py no longer appends EOS after sp.Encode")
    tasks = {}
    for task in protocol["tasks"]:
        ids = list(processor.Encode(facts["instructions"][task])) + [1]
        tasks[task] = dict(instruction=facts["instructions"][task], token_ids=ids,
                           pieces=[processor.IdToPiece(i) for i in ids], n_attended_positions=len(ids),
                           expected=protocol["expected_positions"][task])
        if len(ids) != protocol["expected_positions"][task]:
            raise IntegrityError(f"Unexpected attended positions for {task}: {len(ids)}")
    history = []
    for commit in protocol["encoder"]["history_commits"]:
        old = encoder_facts(git(root, "show", f"{commit}:encode_task_lang.py").decode("utf-8"))
        history.append(dict(commit=git(root, "rev-parse", commit).decode().strip(),
                            tasks_present=sorted(t for t in protocol["tasks"] if t in old["instructions"]),
                            instructions_equal_to_current={t: old["instructions"][t] == facts["instructions"][t]
                                                           for t in protocol["tasks"] if t in old["instructions"]},
                            appends_eos=old["appends_eos"], uses_sp_encode=old["uses_sp_encode"]))
    month8 = month8_cross_check(root, protocol, tasks)
    return dict(tokenizer=dict(path=spec["vendored_path"], sha256=spec["sha256"], bytes=len(blob),
                               source_url=spec["source_url"], eos_id=processor.eos_id(), pad_id=processor.pad_id(),
                               unk_id=processor.unk_id(), bos_id=processor.bos_id(),
                               piece_0=processor.IdToPiece(0), piece_1=processor.IdToPiece(1),
                               vocabulary=processor.GetPieceSize()),
                encoder=dict(path=protocol["encoder"]["path"], sha256=sha256(encoder_path.read_bytes()),
                             appends_eos=facts["appends_eos"], uses_sp_encode=facts["uses_sp_encode"]),
                baseline=dict(token_ids=[0, 1], pieces=[processor.IdToPiece(0), processor.IdToPiece(1)]),
                tasks=tasks, encoder_history=history, month8_cross_check=month8)


def month8_cross_check(root, protocol, tasks):
    import torch
    spec = protocol["month8_language_inputs"]
    manifest = {}
    for line in (Path(root) / spec["manifest"]).read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        manifest[name.strip()] = digest
    out = {}
    for task in protocol["tasks"]:
        name = f"{spec['directory_in_manifest']}/{task}.pt"
        path = Path(root) / spec["directory"] / f"{task}.pt"
        observed = sha256(path.read_bytes())
        if observed != manifest[name]:
            raise IntegrityError(f"Month 8 language input differs from its manifest: {name}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        out[task] = dict(path=f"{spec['directory']}/{task}.pt", sha256=observed,
                         recorded_token_ids_equal=list(payload["token_ids"]) == tasks[task]["token_ids"],
                         recorded_attention_positions=int(payload["attn_mask"].sum().item()),
                         recorded_tokenizer_sha256=payload["identity"]["tokenizer_sha256"],
                         recorded_model_revision=payload["identity"]["model_revision"])
    return out


# --- saved curves -----------------------------------------------------------

def saved_curve_check(root, protocol, groups_by_task):
    spec = protocol["saved_curves"]
    rows = []
    paired_protocol = json.loads((Path(root) / spec["paired_protocol"]).read_bytes())
    records, _, _ = paired.authenticate(root, paired_protocol)
    for ranking, pattern in paired_protocol["cohorts"].items():
        cohort = [r for r in records if fnmatch.fnmatchcase(Path(r.source).name, pattern)]
        for direction in ("insertion", "deletion"):
            equal, total = equal_count(cohort, "lang", direction)
            rows.append(dict(scope="headline_cohort", population=f"{ranking}-ranked PickCube RDT-170M", task="PickCube-v1",
                             model="170m", modality="lang", direction=direction, n_curves=total, n_equal_1_and_5=equal,
                             G_assumed=groups_by_task["PickCube-v1"],
                             historical_rule_implies_equal=historical_k(groups_by_task["PickCube-v1"], 1)
                             == historical_k(groups_by_task["PickCube-v1"], 5)))
    names = [item["path"] for item in spec["all_faithfulness_files"]]
    loaded, files, _ = cs.authenticate_named(root, names, expected={i["path"]: i["sha256"] for i in spec["all_faithfulness_files"]})
    for item in spec["all_faithfulness_files"]:
        faith = cs.faithfulness_rows(loaded[item["path"]]["last"])
        task, model = faith[0].row["task"], faith[0].row["model"]
        for modality in ("lang", "vision"):
            equal, total = 0, 0
            for direction in ("insertion", "deletion"):
                e, t = equal_count(faith, modality, direction)
                equal, total = equal + e, total + t
            groups = 729 if modality == "vision" else groups_by_task[item["instruction_task"]]
            rows.append(dict(scope="all_faithfulness_files", population=item["path"], task=task, model=model,
                             modality=modality, direction="both", n_curves=total, n_equal_1_and_5=equal,
                             instruction_task=item["instruction_task"] if modality == "lang" else None,
                             documented_language_source=item["documented_language_source"] if modality == "lang" else None,
                             G_assumed=groups,
                             historical_rule_implies_equal=historical_k(groups, 1) == historical_k(groups, 5)))
    inputs = [dict(path=f["path"], sha256=f["sha256"], checkout_sha256=f["checkout_sha256"]) for f in files]
    return rows, inputs


def equal_count(records, modality, direction):
    equal = 0
    for record in records:
        grid = record.row["k_grid_auc"]
        if grid != [0, 1, 5, 10, 20, 30, 50, 75, 100]:
            raise IntegrityError("Unexpected saved grid")
        curve = record.row[f"{modality}_{direction}_curve"]
        equal += curve[1] == curve[2]
    return equal, len(records)


# --- git history ------------------------------------------------------------

def git_history(root, protocol):
    spec = protocol["git_history"]
    out = []
    for commit in spec["commits"]:
        blob = git(root, "show", f"{commit}:faithfulness.py")
        lines = blob.decode("utf-8").splitlines()
        hits = [dict(line=i + 1, text=line.strip()) for i, line in enumerate(lines) if "k_count =" in line]
        out.append(dict(commit=git(root, "rev-parse", commit).decode().strip(),
                        committer_date=git(root, "show", "-s", "--format=%cI", commit).decode().strip(),
                        faithfulness_py_sha256=sha256(blob), k_count_lines=hits,
                        floor_present=any(h["text"] == spec["floor_line"] for h in hits),
                        no_floor_present=any(h["text"] == spec["no_floor_line"] for h in hits)))
    archived = (Path(root) / spec["archived_copy"]).read_bytes()
    reference = git(root, "show", f"{spec['archived_copy_reference_commit']}:faithfulness.py")
    return dict(commits=out, archived_copy=dict(
        path=spec["archived_copy"], sha256=sha256(archived),
        crlf_lines=archived.count(b"\r\n"), lines=len(archived.splitlines()),
        equal_to_reference_after_crlf_to_lf=archived.replace(b"\r\n", b"\n") == reference,
        reference_commit=spec["archived_copy_reference_commit"]))


# --- the authors' precomputed embeddings ------------------------------------

def authors_embeddings(protocol, allow_download=False):
    import torch
    from huggingface_hub import hf_hub_download
    spec = protocol["authors_embeddings"]
    files = {}
    for name, digest in spec["files"].items():
        path = Path(hf_hub_download(spec["repository"], name, revision=spec["revision"],
                                    local_files_only=not allow_download))
        blob = path.read_bytes()
        if sha256(blob) != digest:
            raise IntegrityError(f"Authors' embedding differs from its pinned hash: {name}")
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] != 1:
            raise IntegrityError(f"Unexpected authors' embedding layout: {name}")
        files[name] = dict(sha256=digest, bytes=len(blob), shape=list(tensor.shape), dtype=str(tensor.dtype),
                           token_positions=int(tensor.shape[1]))
    return dict(repository=spec["repository"], revision=spec["revision"], files_last_commit=spec["files_last_commit"],
                files=files)


def run_language_inputs(protocol, tokens, authors):
    rows = []
    for run in protocol["run_language_inputs"]:
        if run["source"] == "encode_task_lang":
            positions = tokens["tasks"][run["instruction_task"]]["n_attended_positions"]
        else:
            positions = authors["files"][run["authors_file"]]["token_positions"]
        rows.append(dict(run=run["run"], files=run["files"], documented_language_input=run["documented"],
                         instruction_task=run["instruction_task"], position_source=run["source"],
                         language_positions_G=positions,
                         historical_k_at_1_and_5=[historical_k(positions, 1), historical_k(positions, 5)]))
    return rows


# --- driver -----------------------------------------------------------------

def compute(root, protocol, *, allow_download=False):
    root = Path(root)
    tokens = tokenize(root, protocol)
    groups = {task: tokens["tasks"][task]["n_attended_positions"] for task in protocol["tasks"]}
    counts = []
    for task in protocol["tasks"]:
        counts.extend(count_rows(f"{task} language", groups[task], protocol["grid"]))
    counts.extend(count_rows("vision external camera patches", protocol["vision_groups"], protocol["grid"]))
    curves, curve_inputs = saved_curve_check(root, protocol, groups)
    history = git_history(root, protocol)
    authors = authors_embeddings(protocol, allow_download=allow_download)
    mapping = run_language_inputs(protocol, tokens, authors)
    local = [protocol["encoder"]["path"], protocol["git_history"]["archived_copy"], protocol["saved_curves"]["paired_protocol"],
             protocol["month8_language_inputs"]["manifest"],
             *(v["path"] for v in tokens["month8_cross_check"].values())]
    inputs = dict(faithfulness_files=curve_inputs,
                  local_files=[dict(path=name, sha256=sha256((root / name).read_bytes())) for name in local],
                  vendored_tokenizer=dict(path=protocol["tokenizer"]["vendored_path"], sha256=protocol["tokenizer"]["sha256"]),
                  authors_embeddings={name: record["sha256"] for name, record in authors["files"].items()},
                  git_commits=[c["commit"] for c in history["commits"]])
    return dict(tokens=tokens, counts=counts, curves=curves, history=history, authors=authors, mapping=mapping,
                inputs=inputs)


def run(output, *, root=ROOT, allow_download=False):
    output, root = Path(output), Path(root)
    if output.exists():
        raise IntegrityError("Refusing an existing output directory")
    protocol_bytes, protocol = load_protocol(root)
    result = compute(root, protocol, allow_download=allow_download)
    output.mkdir(parents=True)
    (output / "protocol.json").write_bytes(protocol_bytes)
    cs.write_json(output / "tokenization.json", result["tokens"])
    (output / "group_counts.csv").write_bytes(cs.csv_bytes(result["counts"]))
    (output / "saved_curve_check.csv").write_bytes(cs.csv_bytes(result["curves"]))
    cs.write_json(output / "git_history.json", result["history"])
    cs.write_json(output / "authors_embeddings.json", result["authors"])
    (output / "run_language_inputs.csv").write_bytes(cs.csv_bytes(result["mapping"]))
    import sentencepiece
    import torch
    provenance = dict(
        schema_version=1, analysis=protocol["analysis"], analysis_version=protocol["analysis_version"],
        protocol_sha256=sha256(protocol_bytes),
        source_sha256={name: sha256((root / name).read_bytes()) for name in SOURCES},
        vendored_sha256={protocol["tokenizer"]["vendored_path"]: protocol["tokenizer"]["sha256"]},
        inputs=result["inputs"],
        artifacts={name: cs.artifact_record(output / name) for name in OUTPUTS},
        offline_artifacts=list(CORE_OUTPUTS), network_or_cache_artifacts=list(AUTHORS_OUTPUTS),
        environment=dict(python=__import__("platform").python_version(), sentencepiece=sentencepiece.__version__,
                         torch=torch.__version__),
        command="python -m analysis.language_groups --output <new-empty-directory> [--allow-download]",
        verify_command="python -m analysis.language_groups --verify <result-directory>",
        interpretation="retrospective_reconstruction_of_historical_group_counts")
    cs.write_json(output / "provenance.json", provenance)
    return provenance


def verify(output=RESULTS, *, root=ROOT):
    output, root = Path(output), Path(root)
    provenance = json.loads((output / "provenance.json").read_bytes())
    if set(provenance["artifacts"]) != set(OUTPUTS):
        raise IntegrityError("Artifact roster mismatch")
    for name, record in provenance["artifacts"].items():
        if cs.artifact_record(output / name) != record:
            raise IntegrityError(f"Artifact mismatch: {name}")
    protocol_bytes, protocol = load_protocol(root)
    if sha256(protocol_bytes) != provenance["protocol_sha256"] or (output / "protocol.json").read_bytes() != protocol_bytes:
        raise IntegrityError("Protocol mismatch")
    if {name: sha256((root / name).read_bytes()) for name in SOURCES} != provenance["source_sha256"]:
        raise IntegrityError("Source mismatch")
    vendored = root / protocol["tokenizer"]["vendored_path"]
    if sha256(vendored.read_bytes()) != protocol["tokenizer"]["sha256"]:
        raise IntegrityError("Vendored tokenizer mismatch")
    return dict(status="verified", artifacts=len(OUTPUTS), analysis_version=provenance["analysis_version"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--output", type=Path)
    operation.add_argument("--verify", type=Path)
    parser.add_argument("--allow-download", action="store_true",
                        help="fetch the pinned RDT authors' embeddings into the Hugging Face cache if absent")
    args = parser.parse_args()
    result = run(args.output, allow_download=args.allow_download) if args.output else verify(args.verify)
    print(cs.json_text(result), end="")
