"""Build curated research drafts from committed blobs or an explicit local snapshot.

The default preserves identified notices. An explicit private anonymous
candidate mode proposes the own-author attribution only for local review and
never represents external distribution approval. Third-party notices remain
intact. Private files, unresolved-rights images and model binaries are excluded.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_template_provenance import validate_template_bytes


ENTRYPOINTS = (
    "analysis/revision/__init__.py", "analysis/revision/analyze.py", "analysis/revision/verify.py", "audit.py",
    "collect_contexts.py", "per_step_ig.py", "paired_comparison.py",
    "faithfulness.py", "sanity.py", "baseline_sensitivity.py", "displacement.py",
    "encode_task_lang.py", "generate_overlays.py", "make_annotations.py",
    "ig_resnet.py", "ig_vit.py", "ig_tinyllama.py",
    "scripts/prepare_models.py", "scripts/runtime_probe.py", "scripts/run_workflow.py",
    "scripts/validate_rdt_numerics.py", "scripts/validate_downstream_precision.py",
    "scripts/validate_gradient_repeatability.py", "scripts/validate_fp32_probe.py",
    "scripts/run_numerical_queue.py", "scripts/run_probe_queue.py",
    # This runner is also opened by path for implementation hashing and runpy;
    # static import discovery cannot establish that dependency.
    "scripts/run_weight_arrangement_control.py",
    "scripts/audit_numerical_cohort.py", "scripts/build_revision_assets.py", "scripts/build_paper.py",
    "scripts/validate_template_provenance.py",
)
PENDING_ADDITIONS = (
    "paired_study_analysis.py", "scripts/calibrate_paired_intervals.py",
    "analysis/revision/response_geometry_theory.py",
    "analysis/revision/response_geometry_theory.md",
    "analysis/revision/nested_grid_aliasing.py",
    "analysis/revision/examples/nested_grid_aliasing.json",
    "tests/test_paired_study_analysis.py", "tests/test_calibrate_paired_intervals.py",
    "tests/test_response_geometry_theory.py",
)
DOCUMENTS = (
    "LICENSE", "NOTICE", "README.md", "requirements.txt", "requirements-cpu-lock.txt",
    "requirements-gpu.txt", "configs/base_170m.yaml", "analysis/revision/README.md",
    "data/README.md", "data/README-historical-2026-09-30.md",
    "docs/integrated_gradients.md", "docs/per_step_ig.md", "docs/ig_rdt.md",
    "docs/ig_resnet.md", "docs/ig_vit.md", "docs/ig_tinyllama.md", "docs/runtime_verification.md",
    "docs/legacy_workflows.md", "docs/cpu_reproduction.md", "docs/preprint_status.md", "scripts/build_paper.md", "paper/arxiv_abstract.txt",
    "scripts/setup_runtime.sh", "scripts/run_full_pass.sh", "scripts/run_faithfulness.sh",
    "scripts/run_sanity.sh", "scripts/run_displacement.sh", "scripts/run_overlays.sh",
    "paper/paper.tex", "paper/appendix_aliasing.tex", "paper/references.bib", "paper/tmlr.sty", "paper/tmlr.bst",
    "paper/tmlr-LICENSE", "paper/tmlr-source.json",
)
OMITTED_TESTS = frozenset({"tests/test_legacy_workflows.py", "tests/test_paper_packaging.py",
    "tests/test_paper_packaging_paths.py", "tests/test_research_supplement.py",
    "tests/test_approved_assembly.py"})
OUTPUT_PREFIX = "analysis/revision/results/2026-09-30-v2/"
RESULT_FILES = ("diagnostics.json", "duplicate_population_sensitivity.json", "duplicates.json",
    "population_membership.json.gz", "provenance.json", "random_order_counterexample.json",
    "raw_line_ledger.csv.gz", "reconciliation.json", "results.csv", "solver_endpoint_sensitivity.json", "summary.json")
PAPER_ASSETS = tuple("paper/tables_revision/" + name + ".tex" for name in
    ("baseline", "budget", "completeness", "faithfulness", "macros", "oneb", "rescore", "sanity", "variants", "numerical_roster", "numerical_diagnostics", "paired_rescoring", "episode_influence")) + (
    "paper/figures_revision/lineage.json", "paper/figures_revision/response_geometry.pdf",
    "paper/figures_revision/response_geometry.png", "paper/figures_revision/solver_endpoint.pdf",
    "paper/figures_revision/solver_endpoint.png")
REVISION_FILES = (
    "paper/nhsjs.bst", "paper/tables_revision/revision_facts.tex", "paper/tables_revision/paired_results.tex",
    "CITATION.cff", "paper/README.md", "paper/ijcai26.sty", "paper/named.bst", "scripts/export_online_docx.py",
    "docs/manuscript_revision/README.md", "docs/manuscript_revision/claim_map.md",
    "docs/manuscript_revision/factual_corrections.md", "docs/manuscript_revision/template_contract.md",
    "docs/manuscript_revision/final_local_verification.md",
    "docs/manuscript_revision/final_semantic_audit.md",
    "docs/manuscript_revision/revision_checklist.md",
    "docs/manuscript_revision/round3_checklist.md",
    "docs/manuscript_revision/final_v5_checklist.md",
    "analysis/manuscript_revision/__init__.py", "analysis/manuscript_revision/report.py",
    "analysis/manuscript_revision/figures.py", "analysis/manuscript_revision/README.md",
) + tuple(f"paper/figures_revision/{name}.{extension}" for name in
          ("pipeline", "response_mechanism", "paired_effects", "analytic_toys", "numerical_diagnostics")
          for extension in ("pdf", "png"))
RIGHTS_EXCLUDED = frozenset({"image.jpg", "output/ig_resnet50.png", "output/ig_vit.png", "output/ig_llava.png"})
IDENTIFIERS = ("arjun bajpai", "arjunbajpai2009", "itscool2b",
               "the-readout-not-the-denoiser-repo", "ig-response-geometry", "22133507")
MACHINE_PATH = re.compile(r"[A-Za-z]:[\\/](?:Users|workspace)[\\/]|/(?:home|root|workspace)/|~[/\\]", re.I)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def member_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", name):
        raise ValueError("Expected a normalized relative POSIX member name")
    if any(part in {".", ".."} for part in name.split("/")):
        raise ValueError("Dot path components are forbidden")
    return name


def git(root, *arguments):
    return subprocess.run(["git", "-C", str(root), *arguments], check=True, stdout=subprocess.PIPE).stdout


def snapshot(root, revision):
    commit = git(root, "rev-parse", "--verify", revision + "^{commit}").decode().strip()
    entries = git(root, "ls-tree", "-r", "-z", commit).split(b"\0")
    blobs = {}
    for entry in entries:
        if not entry:
            continue
        info, raw_name = entry.split(b"\t", 1)
        mode, kind, oid = info.split()
        name = member_name(raw_name.decode("utf-8"))
        # A curated package never dereferences a Git symlink or submodule.
        if mode not in {b"100644", b"100755"} or kind != b"blob":
            continue
        blobs[name] = (mode.decode(), oid.decode())
    return commit, blobs


def working_file(root, name):
    """Read one regular, contained file without following aliases or junctions."""
    root = Path(root).resolve()
    path = root / member_name(name)
    current = root
    for part in Path(name).parts:
        current = current / part
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ValueError("Working-copy aliases are not package inputs: " + name)
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError("Required regular working-copy file is missing: " + name)
    return path.read_bytes()


def working_snapshot(root, revision="HEAD"):
    """Add only reviewed new paths to the tracked inventory; never sweep a tree."""
    root = Path(root).resolve()
    commit, index = snapshot(root, revision)
    candidates = set(ENTRYPOINTS) | set(DOCUMENTS) | set(PAPER_ASSETS) | set(PENDING_ADDITIONS) | set(REVISION_FILES)
    reporting = root / "analysis/manuscript_revision"
    if reporting.is_dir():
        candidates.update(path.relative_to(root).as_posix() for path in reporting.rglob("*")
                          if path.is_file() and path.suffix in {".py", ".md", ".json", ".csv"}
                          and "__pycache__" not in path.parts)
    candidates.update(path.relative_to(root).as_posix() for path in (root / "tests").glob("test_*.py"))
    for name in candidates:
        if (root / name).is_file():
            index.setdefault(member_name(name), ("100644", None))
    return commit, index


def import_dependencies(path, data, available):
    tree = ast.parse(data.decode("utf-8"), filename=path)
    local, external = set(), set()
    parent = path.split("/")[:-1]
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                prefix = parent[:len(parent) - node.level + 1]
                base = ".".join([*prefix, base]).strip(".")
            names = [base] + [base + "." + item.name for item in node.names if item.name != "*"]
        else:
            continue
        found = False
        for module in names:
            stem = module.replace(".", "/")
            for candidate in (stem + ".py", stem + "/__init__.py"):
                if candidate in available:
                    local.add(candidate)
                    found = True
        if names and not found:
            top = names[0].split(".")[0]
            if top and top not in sys.stdlib_module_names and top != "__future__":
                external.add(top)
    return local, external


def audit_tokens(values):
    if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Additional audit tokens must be a list of nonempty strings")
    return tuple(sorted(set(value.strip().lower() for value in values)))


def privacy_findings(contents, extra_identifiers=()):
    identifiers = (*IDENTIFIERS, *audit_tokens(extra_identifiers))
    result = []
    for name, data in sorted(contents.items()):
        if name.endswith(".gz"):
            data = gzip.decompress(data)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            # PDF metadata needs its separate existing manuscript asset review.
            result.append(dict(path=name, category="binary_requires_format_review"))
            continue
        lower = text.lower()
        hits = [value for value in identifiers if value in lower]
        if hits:
            result.append(dict(path=name, category="identifying_text", identifiers=hits))
        lines = [i for i, line in enumerate(text.splitlines(), 1) if MACHINE_PATH.search(line)]
        if lines:
            result.append(dict(path=name, category="machine_path_or_portable_example", lines=lines))
    return result


def collect(root, revision, extra_identifiers=(), *, working_copy=False):
    extra_identifiers = audit_tokens(extra_identifiers)
    commit, index = working_snapshot(root, revision) if working_copy else snapshot(root, revision)
    cache = {}
    def read(name):
        member_name(name)
        if name not in index:
            raise ValueError("Required file absent from source snapshot: " + name)
        if name not in cache:
            cache[name] = working_file(root, name) if working_copy else git(root, "cat-file", "blob", index[name][1])
        return cache[name]
    manifest_path = "analysis/revision/input_manifest.json"
    manifest = json.loads(read(manifest_path))
    if working_copy and index[manifest_path][1] is not None:
        baseline_manifest = json.loads(git(root, "cat-file", "blob", index[manifest_path][1]))
        if manifest != baseline_manifest:
            raise ValueError("The immutable historical input manifest differs from the base commit")
    selected = set(ENTRYPOINTS) | set(DOCUMENTS) | set(PAPER_ASSETS) | {manifest_path}
    selected.update(p for p in index if p.startswith("analysis/numerical_case/"))
    selected.update(p for p in index if p.startswith("analysis/paired_rescoring/"))
    if b"\\bibliographystyle{nhsjs}" in read("paper/paper.tex"):
        selected.update(REVISION_FILES)
        selected.update(p for p in index if p.startswith("analysis/manuscript_revision/")
                        and p.endswith((".py", ".md", ".json", ".csv")))
    selected.update(OUTPUT_PREFIX + name for name in RESULT_FILES)
    selected.update(p for p in index if p.startswith("tests/") and p.endswith(".py") and p not in OMITTED_TESTS)
    selected.update(p for p in PENDING_ADDITIONS if p in index)
    raw_input_representations = {}
    for item in manifest["files"]:
        path = member_name(item["path"])
        if not path.startswith("data/") or not path.endswith(".jsonl") or path in selected:
            raise ValueError("Unexpected or repeated raw input path")
        raw = read(path)
        representation = "canonical"
        if digest(raw) != item["sha256"]:
            # Match the frozen loader's exact, separately declared Windows
            # checkout variant. Never normalize or replace exported bytes.
            if (digest(raw) != item.get("git_crlf_checkout_sha256")
                    or digest(raw.replace(b"\r\n", b"\n")) != item["sha256"]):
                raise ValueError("Raw input differs from immutable analysis manifest: " + path)
            representation = "declared_git_crlf_checkout"
        raw_input_representations[path] = representation
        selected.add(path)
    dependencies, external = {}, set()
    queue = sorted(selected)
    while queue:
        name = queue.pop()
        data = read(name)
        if name.endswith(".py"):
            local, thirdparty = import_dependencies(name, data, index)
            dependencies[name] = sorted(local)
            external.update(thirdparty)
            for dependency in sorted(local - selected):
                selected.add(dependency)
                queue.append(dependency)
    if selected & RIGHTS_EXCLUDED or any(p.startswith(("legacy/", "notebooks/", "out/", "output/")) for p in selected):
        raise ValueError("Curated scope contains a historical or rights-excluded artifact")
    contents = {name: read(name) for name in sorted(selected)}
    if working_copy:
        frozen_prefixes = (OUTPUT_PREFIX, "analysis/paired_rescoring/results/",
                           "analysis/paired_rescoring/influence_results/", "analysis/numerical_case/2026-10-01-v1/inputs/",
                           "analysis/numerical_case/2026-10-01-v1/outputs/")
        for name, data in contents.items():
            if name.startswith(frozen_prefixes) and index[name][1] is not None:
                if data != git(root, "cat-file", "blob", index[name][1]):
                    raise ValueError("Frozen scientific artifact differs from the base commit: " + name)
    provenance = json.loads(contents[OUTPUT_PREFIX + "provenance.json"])
    for name, expected in provenance["artifacts_sha256"].items():
        if digest(contents[OUTPUT_PREFIX + member_name(name)]) != expected:
            raise ValueError("Canonical analysis artifact hash mismatch: " + name)
    if working_copy:
        changed = [name for name, data in contents.items() if working_file(root, name) != data]
        if changed:
            raise ValueError("Working copy changed while its snapshot was captured: " + ", ".join(changed))
    hashes = {name: digest(data) for name, data in sorted(contents.items())}
    return contents, dict(commit=commit, scope="identified_working_draft_not_submission",
        source_kind="working_copy_snapshot" if working_copy else "committed_git_blobs",
        snapshot_sha256=digest(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()),
        raw_inputs=len(manifest["files"]), raw_input_representations=raw_input_representations,
        import_dependencies=dependencies,
        external_import_roots=sorted(external),
        pending_uncommitted_additions=[p for p in PENDING_ADDITIONS if p not in index],
        additional_audit_tokens_sha256=digest(json.dumps(extra_identifiers,separators=(",", ":")).encode()),
        additional_audit_token_count=len(extra_identifiers),
        privacy_findings=privacy_findings(contents, extra_identifiers),
        excluded_tracked_paths=sorted(set(index)-selected),
        source_members={name:dict(sha256=digest(data),bytes=len(data),mode=index[name][0],
                                 **({"base_git_blob": index[name][1]} if working_copy else {"git_blob": index[name][1]}))
                        for name,data in contents.items()})


GUIDE = """# Identified research supplement working draft

This package preserves the identified author license and third-party notices.
The exact pinned TMLR files retain the upstream repository Apache-2.0 license
and the bibliography header's LPPL-1.0-or-later notice. Neither overrides the
other by assertion here; see paper/tmlr-source.json and paper/tmlr-LICENSE.
It is not an anonymous submission. No model weights, private audit material,
historical photograph, old notebook, or old rendered attribution is included.

Start with README.md and analysis/revision/README.md. The complete immutable
99-file JSONL inventory and canonical v2 analysis outputs are supplied. Run:

    python -m pip install -r requirements.txt
    python -m analysis.revision.verify analysis/revision/results/2026-09-30-v2
    python -m analysis.revision.analyze --output runs/reproduction --draws 10000 --seed 0
    python -m analysis.revision.verify runs/reproduction

The ten scientific and population artifacts should reproduce byte-for-byte;
provenance.json records the actual environment and checkout representation.
Use a fresh extracted working copy before regenerating manuscript assets with
scripts/build_revision_assets.py, since that command writes current assets.

Runtime code and targeted tests are supplied, with separate GPU requirements.
External upstream source, weights, simulator assets, and newly encoded language
inputs must be acquired under their respective terms. See README.md for the
pinned upstream revision and bounded engineering example. No scientific budget
or future experiment is authorized by this package. The numerical case reproduces
saved diagnostics from a frozen partial snapshot; no production setting is approved.
Prospective comparisons are deferred and their broader claims are excluded.
See analysis/numerical_case/README.md for the CPU reproducer and evidence limits.

Additional CPU evidence is documented in analysis/paired_rescoring/README.md:

    python -m analysis.paired_rescoring.analyze --output runs/paired-rescoring
    python -m analysis.paired_rescoring.influence --output runs/episode-influence
    python -m analysis.paired_rescoring.influence --verify runs/episode-influence

The aliasing toy and manuscript asset regeneration require the full
CPU environment with torch2.14.1+cpu, beyond the minimal requirements.txt:

    python -m analysis.revision.nested_grid_aliasing --output runs/aliasing.json

The paired analysis retains all eight cases and reports conditional episode
intervals. The separate exploratory influence artifact retains all 240 whole-
episode omissions without changing the untrimmed primary results. Its ranges
are not confidence intervals. The toy separates exact IG from perturbation
scores and does not establish RDT efficacy or explain actual-model failures.
All these additions leave canonical v2 intact. See docs/cpu_reproduction.md
for the explicit fresh Windows CPU installation route, including wheel index.

Historical legacy executables and their tests are intentionally omitted. Tests
in this package concern retained code. Archive-only documentation links may
refer to the full repository and do not authorize absent legacy workflows.
"""


ANONYMOUS_GUIDE = """# Private anonymous research supplement candidate

This is a local review candidate, unapproved for external distribution or
submission. Its proposed own-author copyright attribution is pending review;
MIT permission/disclaimer terms and all third-party notices are preserved.
The exact pinned TMLR files retain the upstream repository Apache-2.0 license
and the bibliography header's LPPL-1.0-or-later notice. Neither overrides the
other by assertion here; see paper/tmlr-source.json and paper/tmlr-LICENSE.

The package contains the complete 99-file historical JSONL input inventory,
strict retrospective analysis, canonical numerical results, current runtime,
tests, and candidate manuscript sources. It contains no original photograph,
old photo overlays, private notes, model weights or newly collected model bank.

CPU reproduction (Python 3.12):

    python -m pip install -r requirements.txt
    python -m analysis.revision.verify analysis/revision/results/2026-09-30-v2
    python -m analysis.revision.analyze --output runs/reproduction --draws 10000 --seed 0
    python -m analysis.revision.verify runs/reproduction

See analysis/revision/README.md for estimands and historical identity limits.
The ten scientific/population artifacts should reproduce byte-for-byte;
provenance.json records the actual execution environment and checkout bytes.
Use a separate fresh copy for scripts/build_revision_assets.py, which writes
current paper assets. Its manuscript lineage binds this anonymous candidate.

The full recorded CPU environment is requirements-cpu-lock.txt; GPU dependencies
are separate in requirements-gpu.txt. External RDT source must be the official
revision cd79363a1387e8f81c7724d070ef7e45fd23150f. See docs/ig_rdt.md and
docs/per_step_ig.md. Model weights, simulator assets and language embeddings
must be separately acquired under their original terms and exact identities.
No numerical budget or prospective experiment is selected by this package.

Exported code is committed Git-blob bytes. A modern saved model run checks exact
source hashes and may require its separate frozen source bytes, including line
endings. This candidate includes no such model bank and claims no replay match.
The included CPU demonstrations establish bounded execution only, not accuracy
or GPU validation. The included numerical case reproduces a frozen partial audit
and selected saved tensors; it does not repeat model evaluation or approve a
production setting. Prospective comparisons and their broader claims are deferred.
See analysis/numerical_case/README.md. Technical validation of this exact archive
is recorded in the accompanying local verification. Human submission review
remains open.
The additional paired response analysis uses the minimal saved-data environment:

    python -m analysis.paired_rescoring.analyze --output runs/paired-rescoring
    python -m analysis.paired_rescoring.influence --output runs/episode-influence
    python -m analysis.paired_rescoring.influence --verify runs/episode-influence

The aliasing toy and manuscript asset regeneration require the full
CPU environment with torch2.14.1+cpu, beyond the minimal requirements.txt:

    python -m analysis.revision.nested_grid_aliasing --output runs/aliasing.json

requirements-cpu-lock.txt records package versions. docs/cpu_reproduction.md
supplies the explicit CPU-wheel index and installation procedure checked in
a new isolated Windows x86-64 Python3.12.14 environment. No new Linux or TeX
installation is implied. The influence ranges describe all 240 whole-episode
omissions; they are exploratory diagnostics, not confidence intervals or a
replacement for the original paired results.

See analysis/paired_rescoring/README.md for all cases, endpoint rules and interval
limitations. These additions do not approve a production budget or ranking method.
Archive-only links may refer to omitted historical files.
"""


def anonymous_candidate(contents, extra_identifiers):
    """Create a marked local proposal; never mutate or relicense canonical files."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import package_paper_sources
    transformed = dict(contents)
    old_notice = b"Copyright (c) 2026 Arjun Bajpai"
    if transformed["LICENSE"].count(old_notice) != 1:
        raise ValueError("Own-author license notice differs from the reviewed candidate proposal")
    transformed["LICENSE"] = transformed["LICENSE"].replace(old_notice, b"Copyright (c) 2026 Anonymous author")
    # Keep the exact frozen runtime recipes/model revisions, removing only the
    # two reviewed identity/deposit paragraphs. New identity syntax fails below.
    readme = contents["README.md"].decode("utf-8")
    for pattern in (r"^The sole current author is [^\n]*\n", r"^`CITATION\.cff` identifies [^\n]*\n"):
        readme, count = re.subn(pattern, "", readme, flags=re.M)
        if count != 1:
            raise ValueError("Root guide identity paragraphs differ from the reviewed transformation")
    banner = ("PRIVATE ANONYMOUS CANDIDATE. Unapproved for external distribution or submission.\n"
              "The proposed own-author notice is for local review only; third-party notices remain intact.\n"
              "See SUPPLEMENT_README.md for scope and source-byte replay limitations.\n\n")
    transformed["README.md"] = (banner + readme).encode()
    transformed["paper/paper.tex"] = package_paper_sources.anonymous_source(contents["paper/paper.tex"].decode("utf-8")).encode()
    lineage_path = "paper/figures_revision/lineage.json"
    lineage = json.loads(contents[lineage_path])
    if lineage["manuscript_source_sha256"] != digest(contents["paper/paper.tex"]):
        raise ValueError("Canonical manuscript lineage differs before transformation")
    lineage["manuscript_source_sha256"] = digest(transformed["paper/paper.tex"])
    manuscript_inputs = lineage.get("manuscript_inputs_sha256", {})
    if "paper/paper.tex" in manuscript_inputs:
        if manuscript_inputs["paper/paper.tex"] != digest(contents["paper/paper.tex"]):
            raise ValueError("Canonical manuscript input hash differs before transformation")
        manuscript_inputs["paper/paper.tex"] = digest(transformed["paper/paper.tex"])
    transformed[lineage_path] = (json.dumps(lineage, indent=2)+"\n").encode()
    findings = privacy_findings(transformed, extra_identifiers)
    if any(row["category"] == "identifying_text" for row in findings):
        raise ValueError("Known identifying text remains in the private anonymous candidate")
    changes = {name:dict(source_sha256=digest(contents[name]),candidate_sha256=digest(data))
               for name,data in transformed.items() if data != contents[name]}
    if set(changes) != {"LICENSE", "README.md", "paper/paper.tex", lineage_path}:
        raise ValueError("Anonymous candidate changed an unreviewed source member")
    return transformed, dict(transformations=changes,privacy_findings=findings,
        manuscript_transform_sha256=digest(Path(package_paper_sources.__file__).read_bytes()),
        licensing_status="proposed_review_copy_attribution_not_approved_for_external_distribution",
        third_party_notices="preserved_exactly",canonical_public_license="unchanged")


def build(root, revision, output, extra_identifiers=(), *, private_anonymous_candidate=False, working_copy=False):
    if private_anonymous_candidate and Path(output).resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Private anonymous candidate output must be outside the canonical public repository")
    contents, audit = (collect(root, revision, extra_identifiers, working_copy=True) if working_copy
                       else collect(root, revision, extra_identifiers))
    if private_anonymous_candidate:
        contents, audit["anonymous_candidate"] = anonymous_candidate(contents, extra_identifiers)
    audit["template_provenance"] = validate_template_bytes(contents)
    contents["SUPPLEMENT_README.md"] = (ANONYMOUS_GUIDE if private_anonymous_candidate else GUIDE).encode()
    if "analysis/manuscript_revision/report.py" in contents:
        contents["SUPPLEMENT_README.md"] += b"""

The manuscript build entry point is the already assembled paper/paper.tex.
This package supports scientific reproduction and compilation of that source.
Compile the current source directly. Historical reconstruction tools and
their preparation records are not required and are not included.

The current article's separate reporting layer is analysis/manuscript_revision/.
In a fresh extraction, run python scripts/build_revision_assets.py --reproduce
to regenerate its summaries and displays while checking the frozen evidence.
The current presentation uses a 12-point article layout and nhsjs.bst; the
preserved TMLR files document the historical template and retain their notices.
Build with python scripts/build_paper.py --engine tectonic, or provide its path.
Times New Roman and the public TeX resources must already be available locally.
For first-time TeX setup only, --allow-resource-downloads allows public resource
fetches into Tectonic's cache. It does not upload manuscript files.
Current technical documentation includes the claim map, factual corrections,
template contract, final local verification and scientific meaning audit.
Historical archive paths mentioned in those documents are not build inputs
and are not included. This package does not perform a journal submission.
"""
    if working_copy:
        contents["SUPPLEMENT_README.md"] = contents["SUPPLEMENT_README.md"].replace(
            b"Exported code is committed Git-blob bytes.", b"Exported code is an explicit hashed working-copy snapshot.")
        contents["SUPPLEMENT_README.md"] += ("\nSource: exact local working-copy bytes; the recorded commit is a base reference, not the exported source.\n"
            "Snapshot SHA-256: " + audit["snapshot_sha256"] + "\n").encode()
    # This supplements the unchanged embedded notice and also covers the
    # configuration derived from the same pinned upstream project.
    sampler = contents["rdt_sampling.py"].decode("utf-8")
    notice = sampler.split("Original sampler license:\n", 1)[1].split('"""', 1)[0].strip()
    contents["THIRD_PARTY_NOTICES.txt"] = ("rdt_sampling.py and configs/base_170m.yaml derive from the official RDT project.\n"
        "Source: https://github.com/thu-ml/RoboticsDiffusionTransformer/tree/cd79363a1387e8f81c7724d070ef7e45fd23150f\n\n" + notice + "\n").encode()
    package_manifest = dict(kind="private_anonymous_research_supplement_candidate" if private_anonymous_candidate else "identified_research_supplement",
        status="unapproved_for_external_distribution_or_submission" if private_anonymous_candidate else "working_draft_not_submission",
        source_kind=audit.get("source_kind", "committed_git_blobs"),
        source_snapshot_sha256=audit.get("snapshot_sha256"),
        members={p:dict(sha256=digest(b),bytes=len(b)) for p,b in sorted(contents.items())})
    contents["SUPPLEMENT_MANIFEST.json"] = (json.dumps(package_manifest,indent=2)+"\n").encode()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    archive = output / ("private-anonymous-research-supplement-candidate.zip" if private_anonymous_candidate else "identified-research-supplement-draft.zip")
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as stream:
        for name, data in sorted(contents.items()):
            member_name(name)
            info = zipfile.ZipInfo(name, date_time=(2026,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            stream.writestr(info,data)
    audit.update(packager_sha256=digest(Path(__file__).read_bytes()),
        distribution_status="not_authorized_private_candidate_only" if private_anonymous_candidate else "identified_working_draft_not_submission",
        archive=dict(file=archive.name,sha256=digest(archive.read_bytes()),bytes=archive.stat().st_size),
        packaged_members=len(contents),uncompressed_bytes=sum(map(len,contents.values())))
    (output/"packaging-audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--commit", help="Export the exact files in this Git commit")
    source.add_argument("--working-copy", action="store_true", help="Export current curated files, including reviewed new revision paths, with a byte-hashed snapshot; HEAD is recorded only as the base")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--audit-tokens-file",type=Path,
        help="Optional private JSON list of additional identity tokens. The file is never packaged; findings stay in the external audit.")
    parser.add_argument("--private-anonymous-candidate",action="store_true",
        help="Prepare a private local candidate with proposed own-author attribution. This is not an approved license change or distribution.")
    args=parser.parse_args()
    extra_identifiers = audit_tokens(json.loads(args.audit_tokens_file.read_bytes())) if args.audit_tokens_file else ()
    result=build(args.root,args.commit or "HEAD",args.out,extra_identifiers,
                 private_anonymous_candidate=args.private_anonymous_candidate,working_copy=args.working_copy)
    print(json.dumps({k:result[k] for k in ("commit","raw_inputs","packaged_members","archive","pending_uncommitted_additions")},indent=2))


if __name__=="__main__":
    main()
