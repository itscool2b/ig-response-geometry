"""Build curated research working drafts from committed Git blobs.

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
    "docs/legacy_workflows.md", "scripts/build_paper.md", "paper/arxiv_abstract.txt",
    "scripts/setup_runtime.sh", "scripts/run_full_pass.sh", "scripts/run_faithfulness.sh",
    "scripts/run_sanity.sh", "scripts/run_displacement.sh", "scripts/run_overlays.sh",
    "paper/paper.tex", "paper/appendix_aliasing.tex", "paper/references.bib", "paper/tmlr.sty", "paper/tmlr.bst",
    "paper/tmlr-LICENSE", "paper/tmlr-source.json",
)
OMITTED_TESTS = frozenset({"tests/test_legacy_workflows.py", "tests/test_paper_packaging.py",
    "tests/test_paper_packaging_paths.py", "tests/test_research_supplement.py"})
OUTPUT_PREFIX = "analysis/revision/results/2026-09-30-v2/"
RESULT_FILES = ("diagnostics.json", "duplicate_population_sensitivity.json", "duplicates.json",
    "population_membership.json.gz", "provenance.json", "random_order_counterexample.json",
    "raw_line_ledger.csv.gz", "reconciliation.json", "results.csv", "solver_endpoint_sensitivity.json", "summary.json")
PAPER_ASSETS = tuple("paper/tables_revision/" + name + ".tex" for name in
    ("baseline", "budget", "completeness", "faithfulness", "macros", "oneb", "rescore", "sanity", "variants", "numerical_roster", "numerical_diagnostics", "paired_rescoring")) + (
    "paper/figures_revision/lineage.json", "paper/figures_revision/response_geometry.pdf",
    "paper/figures_revision/response_geometry.png", "paper/figures_revision/solver_endpoint.pdf",
    "paper/figures_revision/solver_endpoint.png")
RIGHTS_EXCLUDED = frozenset({"image.jpg", "output/ig_resnet50.png", "output/ig_vit.png", "output/ig_llava.png"})
IDENTIFIERS = ("arjun bajpai", "arjunbajpai2009", "itscool2b",
               "the-readout-not-the-denoiser-repo", "22133507")
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


def collect(root, revision, extra_identifiers=()):
    extra_identifiers = audit_tokens(extra_identifiers)
    commit, index = snapshot(root, revision)
    cache = {}
    def read(name):
        member_name(name)
        if name not in index:
            raise ValueError("Required file absent from committed source: " + name)
        if name not in cache:
            cache[name] = git(root, "cat-file", "blob", index[name][1])
        return cache[name]
    manifest_path = "analysis/revision/input_manifest.json"
    manifest = json.loads(read(manifest_path))
    selected = set(ENTRYPOINTS) | set(DOCUMENTS) | set(PAPER_ASSETS) | {manifest_path}
    selected.update(p for p in index if p.startswith("analysis/numerical_case/"))
    selected.update(p for p in index if p.startswith("analysis/paired_rescoring/"))
    selected.update(OUTPUT_PREFIX + name for name in RESULT_FILES)
    selected.update(p for p in index if p.startswith("tests/") and p.endswith(".py") and p not in OMITTED_TESTS)
    selected.update(p for p in PENDING_ADDITIONS if p in index)
    for item in manifest["files"]:
        path = member_name(item["path"])
        if not path.startswith("data/") or not path.endswith(".jsonl") or path in selected:
            raise ValueError("Unexpected or repeated raw input path")
        if digest(read(path)) != item["sha256"]:
            raise ValueError("Committed raw input differs from immutable analysis manifest: " + path)
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
    provenance = json.loads(contents[OUTPUT_PREFIX + "provenance.json"])
    for name, expected in provenance["artifacts_sha256"].items():
        if digest(contents[OUTPUT_PREFIX + member_name(name)]) != expected:
            raise ValueError("Canonical analysis artifact hash mismatch: " + name)
    return contents, dict(commit=commit, scope="identified_working_draft_not_submission",
        raw_inputs=len(manifest["files"]), import_dependencies=dependencies,
        external_import_roots=sorted(external),
        pending_uncommitted_additions=[p for p in PENDING_ADDITIONS if p not in index],
        additional_audit_tokens_sha256=digest(json.dumps(extra_identifiers,separators=(",", ":")).encode()),
        additional_audit_token_count=len(extra_identifiers),
        privacy_findings=privacy_findings(contents, extra_identifiers),
        excluded_tracked_paths=sorted(set(index)-selected),
        source_members={name:dict(sha256=digest(data),bytes=len(data),mode=index[name][0],git_blob=index[name][1]) for name,data in contents.items()})


GUIDE = """# Identified research supplement working draft

This package preserves the identified author license and third-party notices.
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

The aliasing toy and manuscript asset regeneration require the recorded full
CPU environment with torch2.14.1+cpu, beyond the minimal requirements.txt:

    python -m analysis.revision.nested_grid_aliasing --output runs/aliasing.json

The paired analysis retains all eight cases and reports conditional episode
intervals. The toy aliasing example concerns recovery of exact IG, not ranking
efficacy or the cause of actual-model failures. Both leave canonical v2 intact.

Historical legacy executables and their tests are intentionally omitted. Tests
in this package concern retained code. Archive-only documentation links may
refer to the full repository and do not authorize absent legacy workflows.
"""


ANONYMOUS_GUIDE = """# Private anonymous research supplement candidate

This is a local review candidate, unapproved for external distribution or
submission. Its proposed own-author copyright attribution is pending review;
MIT permission/disclaimer terms and all third-party notices are preserved.

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
See analysis/numerical_case/README.md. Final manuscript/package review remains open.
The additional paired response analysis uses the minimal saved-data environment:

    python -m analysis.paired_rescoring.analyze --output runs/paired-rescoring

The aliasing toy and manuscript asset regeneration require the recorded full
CPU environment with torch2.14.1+cpu, beyond the minimal requirements.txt:

    python -m analysis.revision.nested_grid_aliasing --output runs/aliasing.json

requirements-cpu-lock.txt records package versions; it is not a CPU-wheel-index
installation recipe or evidence of a newly tested installation.

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


def build(root, revision, output, extra_identifiers=(), *, private_anonymous_candidate=False):
    if private_anonymous_candidate and Path(output).resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Private anonymous candidate output must be outside the canonical public repository")
    contents, audit = collect(root, revision, extra_identifiers)
    if private_anonymous_candidate:
        contents, audit["anonymous_candidate"] = anonymous_candidate(contents, extra_identifiers)
    contents["SUPPLEMENT_README.md"] = (ANONYMOUS_GUIDE if private_anonymous_candidate else GUIDE).encode()
    # This supplements the unchanged embedded notice and also covers the
    # configuration derived from the same pinned upstream project.
    sampler = contents["rdt_sampling.py"].decode("utf-8")
    notice = sampler.split("Original sampler license:\n", 1)[1].split('"""', 1)[0].strip()
    contents["THIRD_PARTY_NOTICES.txt"] = ("rdt_sampling.py and configs/base_170m.yaml derive from the official RDT project.\n"
        "Source: https://github.com/thu-ml/RoboticsDiffusionTransformer/tree/cd79363a1387e8f81c7724d070ef7e45fd23150f\n\n" + notice + "\n").encode()
    package_manifest = dict(kind="private_anonymous_research_supplement_candidate" if private_anonymous_candidate else "identified_research_supplement",
        status="unapproved_for_external_distribution_or_submission" if private_anonymous_candidate else "working_draft_not_submission",
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
    parser.add_argument("--commit",required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--audit-tokens-file",type=Path,
        help="Optional private JSON list of additional identity tokens. The file is never packaged; findings stay in the external audit.")
    parser.add_argument("--private-anonymous-candidate",action="store_true",
        help="Prepare a private local candidate with proposed own-author attribution. This is not an approved license change or distribution.")
    args=parser.parse_args()
    extra_identifiers = audit_tokens(json.loads(args.audit_tokens_file.read_bytes())) if args.audit_tokens_file else ()
    result=build(args.root,args.commit,args.out,extra_identifiers,private_anonymous_candidate=args.private_anonymous_candidate)
    print(json.dumps({k:result[k] for k in ("commit","raw_inputs","packaged_members","archive","pending_uncommitted_additions")},indent=2))


if __name__=="__main__":
    main()
