# How Scoring Responses Change Attribution Evaluation in a Diffusion Policy

This repository contains the revised manuscript, immutable saved records and reproducible analysis. The study asks how changing the scoring response changes attribution evaluation. It combines an exact response identity, saved-curve analysis, a matched methods study and numerical checks. Findings are conditional on the recorded models, tasks and contexts; they do not establish general ranking superiority or improved task success. The [October 4 submitted snapshot](https://github.com/itscool2b/ig-response-geometry/tree/manuscript-2026-10-04-polish) remains preserved separately.

The sole author is Arjun Bajpai, Independent Researcher. The [canonical source](paper/paper.tex), [identified paper](paper/paper.pdf) and [blinded reading copy](paper/paper-anonymous-draft.pdf) describe the same revision. It contains a 221-word abstract, six keywords, five consolidated figures, four numbered tables and 34 references, including 30 qualifying peer-reviewed primary articles. The delivery includes a tracked Standard DOCX, clean identified Online DOCX and anonymous response letter. The [current revision guide](docs/manuscript_revision/README.md) distinguishes production checks from journal submission.

The [reporting layer](analysis/manuscript_revision/README.md), version `2026-10-03-v2`, remains a frozen source of descriptive saved-curve results. The R1 revision adds historical 1B success splits and additional tasks, matched methods with random and fixed controls, and precision, quadrature and coordinate comparisons. The [display inputs](analysis/manuscript_revision/nhsjs_r1/inputs/manifest.json) and [lineage](paper/figures_revision/r1_lineage.json) bind the five current figures and four tables to their numerical sources. Earlier [edit](docs/manuscript_revision/exact_edits_checklist.md), [claim](docs/manuscript_revision/claim_map.md) and [verification](docs/manuscript_revision/final_local_verification.md) records describe the October 4 version.

The tag `manuscript-2026-10-04-polish` and the earlier local `artifacts/current` delivery identify the submitted version, not the R1 revision. Current title-named Word files, figures, sources, supplement and verification manifests are distributed as a separate revision delivery. Receipts always identify their exact bytes.

The identified 19-page submitted preprint was published on October 9, 2026 as [Zenodo version 3.0](https://doi.org/10.5281/zenodo.23269517), with the PDF bytes from the preserved October 4 snapshot and Arjun Bajpai as sole author. That deposit has not been replaced by the R1 revision.

The earlier Zenodo versions retain their own titles and DOIs: [Response Geometry in Integrated Gradients: A Diffusion-Policy Case Study](https://doi.org/10.5281/zenodo.23091579) is version 2, and [The Readout, Not the Denoiser: Per-Step Integrated Gradients for Diffusion-Policy Vision-Language-Action Models](https://doi.org/10.5281/zenodo.22133507) is version 1. Their histories and licenses remain preserved.

The revision retains the author-supplied affiliation, funding and competing-interest information and the approved mentor acknowledgments. The AI disclosure is preserved verbatim at the author's instruction. Repository publication and local delivery checks do not constitute journal submission or acceptance; that requires the revision channel's receipt.

## Evidence guide

- `analysis/revision` and `analysis/paired_rescoring` preserve the original analyses, populations and protocols.
- `analysis/manuscript_revision` contains the current descriptive reporting version and the preserved earlier version.
- `data` contains the 99 immutable saved JSONL inputs; historical notebooks, displays and `legacy` records remain separately labeled.
- `paper` contains the manuscript, generated displays and proofs. The private repository retains additional research material and is synchronized only for reviewed shared files.

The quadratic response Q is an auxiliary discrepancy, not the policy log likelihood. For a fixed ranking, normalized insertion and deletion AUC scored with N are no greater than those scored with Q on fully in-range curves. Strict decrease requires positive AUC weight on a squared residual r with 0 < r < B, where B is the all-baseline squared residual. The conventional quality implications are opposite for insertion and deletion. Observed overshoots can reverse the mean difference. The analytic random-order example is distinct from the R1 experiment's measured random and fixed-order controls. Constructed target reversals and aliasing examples do not establish superior RDT rankings. The numerical accuracy of the historical headline rankings remains unverified.

## Reproduce the saved-data analysis on CPU

For the complete CPU environment, use the [fresh-install procedure](docs/cpu_reproduction.md), checked on Windows x86-64 with Python 3.12.14. It explicitly selects the official Torch CPU wheel before installing the pinned dependencies. No GPU or model download is needed for these reproduction commands.

The active analysis reads all 99 preserved JSONL inputs through an immutable manifest. It records every physical line, quarantines the one known corrupted line by hash, retains all duplicate occurrences, and reports explicit primary and sensitivity populations. Point estimates and episode-bootstrap intervals share population identities. See [analysis/revision/README.md](analysis/revision/README.md).

```bash
python -m venv .venv
# Activate the environment for your shell.
python -m pip install -r requirements.txt
python -m analysis.revision.verify analysis/revision/results/2026-09-30-v2
python -m analysis.revision.analyze --output runs/saved-data-reanalysis --draws 10000 --seed 0
python -m analysis.revision.verify runs/saved-data-reanalysis
```

The output directory must be new. The canonical registry contains 284 results and 57 source populations. A clean-input rerun reproduced all ten scientific/lineage artifacts byte-for-byte; environment/checkout provenance describes the actual execution. Git preserves hashed output bytes across platforms.

The separate [paired rescoring analysis](analysis/paired_rescoring/README.md) measures `AUC_Q - AUC_L2` within each saved curve. Its frozen protocol remains mean-primary. The new [reporting layer](analysis/manuscript_revision/README.md) leads with paired-call medians and positive-change shares, while retaining means, intervals, full tails and episode influence. This is an explicitly retrospective presentation choice. All eight cases, endpoint orientation, undefined gaps and sampled overshoots are retained. The manuscript calls the norm target N. The conditional episode-bootstrap intervals do not compare the two unpaired ranking cohorts or test ranking superiority.
All eight paired medians are positive. Six means are negative, and three of those have descriptive episode-bootstrap intervals spanning zero. First calls carry most of the vision-insertion overshoot contribution. The paper retains all calls and reports both summaries without trimming the tail.

The aliasing toy and manuscript asset regeneration also require PyTorch, which is absent from the minimal `requirements.txt`. Install and activate the [checked full CPU environment](docs/cpu_reproduction.md), with `torch==2.14.1+cpu` and `torch.version.cuda is None`, before running:

```bash
python -m analysis.revision.nested_grid_aliasing --output runs/nested-grid-aliasing.json
```

The paper's practical evaluation procedure separates attribution target, evaluation response, matched ranking controls, numerical checks and the unit of statistical analysis. A chosen response defines the comparison; passing finite numerical checks does not universally certify an integration budget.

The [exploratory episode-influence supplement](analysis/paired_rescoring/influence_results/2026-10-01-v1/methods_results.md) retains all eight cases and all 240 whole-episode omissions. Every negative insertion mean retains its sign after each single-episode omission; two language-deletion signs can change. These are influence diagnostics, not confidence intervals, independent replications or a reason to trim the primary analysis.

The CPU revision environment uses Python 3.12.14. `requirements-cpu-lock.txt` records installed package versions; [the installation procedure](docs/cpu_reproduction.md) supplies the explicit CPU index and was checked in a new isolated Windows environment. `requirements.txt` is the smaller pinned saved-data analysis layer. Historical notebooks and `make_*_figs.py` generators are archival workflows; their hardcoded constants and saved outputs do not authenticate missing primary data. Do not run historical notebooks over the preserved figures.

## Revised GPU runtime

The supported setup uses Linux, CUDA 12.8, Torch 2.8.0, a working Vulkan/SAPIEN graphics stack, and the dependencies in `requirements-gpu.txt`. The actual revision environment has passed rendering, checkpoint loading, finite-gradient and controller smoke checks on a 96 GB RTX PRO 6000 Blackwell for current 170M and explicit authors-1B checkpoints. These checks establish runtime plumbing, not attribution accuracy or task competence.

Clone the official upstream source and select the exact validated revision:

```bash
git clone https://github.com/thu-ml/RoboticsDiffusionTransformer.git rdt-upstream
git -C rdt-upstream checkout cd79363a1387e8f81c7724d070ef7e45fd23150f
export RDT_SOURCE="$PWD/rdt-upstream"
python -m pip install -r requirements-gpu.txt
python encode_task_lang.py --output-dir runs/language-v1 --revision 3db67ab1af984cf10548a73467f0e5bca2aaaeb2
```

Install the correct CUDA Torch build before the additional GPU dependencies. `scripts/setup_runtime.sh` records the tested RunPod setup; inspect its paths before using it elsewhere. Upstream must be clean and at the pinned commit. The loader validates exact checkpoint keys, shapes and finite values and records checkpoint hashes. For 1B, explicitly select `--checkpoint-mode authors` with the authors' checkpoint path, `lora` with an adapter path, or `pretrained`. There is no file-presence fallback.

The actual revision loading checks covered the identified 170M and authors' 1B checkpoints. The LoRA loading branch has not received an actual adapter-application validation in this revision and is outside the retained numerical case study.

The GPU setup records tested package versions and source/model identities, but does not provide a fully immutable container reconstruction. Its base image is named by tag, system graphics packages come from mutable package repositories, and the virtual environment inherits container packages. Exact replay requires the recorded runtime contracts and additional environment verification.

Language encoding saves token IDs, task text, masks, source/model/tokenizer identity and the encoded PAD/EOS baseline. T5 has no BOS token; `baseline_bos_eos.pt` is a retained filename. Fresh encoding cannot reconstruct unidentified historical embeddings.

A bounded engineering example:

```bash
python per_step_ig.py --task PickCube-v1 --model 170m \
  --model-revision 8aa386cac3bbfd9540676c75b3d767cc7f88a10a \
  --vision-revision 9fdffc58afc957d1a03a25b10dba0329ab15c2a3 \
  --lang-dir runs/language-v1 --episodes 1 --max-policy-calls 1 \
  --seed-base 910001 --m 4 --quadrature trapezoid --no-checkpoint \
  --out runs/engineering-example/metrics.jsonl
```

m4 is a plumbing check. Choose scientific integration budgets only after coordinate, ranking, precision and response validation on the retained configurations. The revised core uses fp32 path/product/accumulation arithmetic, evaluated endpoints and explicit nonfinite failures. Low-precision model forwards can still affect accuracy; small completeness residuals do not prove correct coordinates.

New runs use unique directories, immutable manifests, stored noise and hash-bound sidecars. `--resume` accepts only identical settings/source and verified committed episodes. It retains failed attempts and rebuilds the derived JSONL. Historical JSONL files cannot be resumed. Evaluation requires matching model/language/source identities and exact reference replay; explicitly permitted legacy input remains unverified.

Vision attribution is post-image-adaptor, language attribution is post-language-adaptor, and state attribution is pre-state-adaptor. The observation pipeline uses one current external image plus five fixed background slots. One policy call predicts a 64-step chunk, subsampled into at most 16 controller steps. See [docs/ig_rdt.md](docs/ig_rdt.md), [docs/integrated_gradients.md](docs/integrated_gradients.md) and [docs/per_step_ig.md](docs/per_step_ig.md).

## Repository map and manuscript

- `analysis/revision/`: active retrospective analysis, tests, lineage and result registry.
- `data/`: immutable historical JSONL files and their current evidence guide.
- `pipeline.py`, `per_step_attribution.py`, `rdt_sampling.py`, `experiment_io.py`: revised loading, attribution, sampling and storage.
- `faithfulness.py`, `sanity.py`, `baseline_sensitivity.py`, `displacement.py`: controlled replay/evaluation entrypoints. Check each current `--help`; historical shell wrappers are not an execution specification for the revised protocol.
- `scripts/validate_*.py`: targeted numerical diagnostics requiring recorded decisions and authenticated contexts.
- `paper/`: current NHSJS-format source and PDFs, retained TMLR files and licenses, and preserved historical assets.
- `docs/`: current contracts and explicitly bounded historical demonstrations.

See [scripts/build_paper.md](scripts/build_paper.md) for identified and anonymous builds. The historical PDF is preserved under `legacy/pre-response-geometry/`. Identified builds update both current PDF filenames together. Final paper and supplement checks include source provenance, anonymity, rights, numerical/statistical validity and full rendered inspection. This repository does not claim journal acceptance or completed submission.

## Citation and licenses

`CITATION.cff` retains the code snapshot tagged `manuscript-2026-10-04-polish`, with reporting version `2026-10-03-v2`, and gives [Zenodo version 3.0](https://doi.org/10.5281/zenodo.23269517) as the preferred manuscript citation. Cite an earlier version with its own DOI and title when referring specifically to that version.

The current [Zenodo preprint](https://doi.org/10.5281/zenodo.23269517) is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Code and original project records are MIT licensed. The third-party demonstration photograph `image.jpg` and its three Month 2 derivative figures have unresolved redistribution provenance. They have been removed from the current public tree and are excluded from research packages. Existing repository history has not been rewritten and does not grant permission to reuse them. The TMLR template repository's Apache 2.0 license and the bibliography file's separate LPPL notice are both retained. The historical IJCAI template has its own provenance. See [NOTICE](NOTICE) and [template provenance](paper/tmlr-source.json).

Historical paths beginning with `previous-runs/` refer to the separate local records archive outside this repository. They are not build dependencies.
