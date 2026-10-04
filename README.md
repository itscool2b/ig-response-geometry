# How Scoring Responses Change Attribution Evaluation in a Diffusion Policy

This repository contains the manuscript, immutable saved records and reproducible CPU analysis for the [October 3, 2026 snapshot](https://github.com/itscool2b/ig-response-geometry/tree/manuscript-2026-10-03). The study asks how changing the scoring response changes evaluation when rankings and interventions remain fixed. It combines an exact response identity, descriptive saved-curve analysis and separate analytic and numerical checks. It does not establish ranking superiority or improved task success.

The sole author is Arjun Bajpai. The [canonical source](paper/paper.tex), [identified paper](paper/paper.pdf) and [blinded Standard copy](paper/paper-anonymous-draft.pdf) describe the same revision. The blinded Standard PDF has 18 pages, the identified reading copy has 19 and the Online Word document has 20. The one-paragraph abstract contains 245 words. Six keywords are retained. Both formats retain five figures, two numbered tables and 17 references.

The [reporting layer](analysis/manuscript_revision/README.md), version `2026-10-03-v2`, adds descriptive range prevalence, full-mean first-call contributions, cohort comparisons, nominal-grid sensitivity and numerical summaries while preserving the original mean-primary protocol and outputs. The [revision guide](docs/manuscript_revision/README.md), [Final-v5 ledger](docs/manuscript_revision/final_v5_checklist.md), [claim map](docs/manuscript_revision/claim_map.md) and [verification](docs/manuscript_revision/final_local_verification.md) document the current scope.

The tag `manuscript-2026-10-03` identifies the manuscript and reporting snapshot. The separate local delivery, `artifacts/current`, includes title-named PDFs, Online Word, figures, anonymous manuscript sources, an identified research supplement and exact-byte verification manifests. That submission package is not tracked in Git. Earlier receipts identify only their earlier bytes.

The [historical Zenodo deposit](https://doi.org/10.5281/zenodo.22133507) retains The Readout, Not the Denoiser. A preceding 25-page public draft was Response Geometry in Integrated Gradients: A Diffusion-Policy Case Study. Their histories and licenses remain preserved.

The author has supplied affiliation, contribution, funding and competing-interest declarations and confirmed both mentors' approval of the acknowledgments and submission. AI-policy handling, journal eligibility and form requirements, journal answers and an independent scientist's cold read remain pending. Repository publication does not constitute journal submission or acceptance.

## Evidence guide

- `analysis/revision` and `analysis/paired_rescoring` preserve the original analyses, populations and protocols.
- `analysis/manuscript_revision` contains the current descriptive reporting version and the preserved earlier version.
- `data` contains the 99 immutable saved JSONL inputs; historical notebooks, displays and `legacy` records remain separately labeled.
- `paper` contains the manuscript, generated displays and proofs. The private repository retains additional research material and is synchronized only for reviewed shared files.

The quadratic response Q is an auxiliary discrepancy, not the policy log likelihood. For a fixed ranking, normalized insertion and deletion AUC scored with N are no greater than those scored with Q on fully in-range curves. Strict decrease requires positive AUC weight on a squared residual r with 0 < r < B, where B is the all-baseline squared residual. The conventional quality implications are opposite for insertion and deletion. Observed overshoots can reverse the mean difference. The random-order reference is analytic rather than a measured RDT control. The target-reversal and aliasing constructions are existence examples, not evidence of superior RDT rankings.

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

`CITATION.cff` identifies the manuscript snapshot tagged `manuscript-2026-10-03`, with reporting version `2026-10-03-v2`. No new DOI has been assigned. When referring specifically to the earlier released paper or records, cite [the historical Zenodo version](https://doi.org/10.5281/zenodo.22133507) and its original title instead.

Code and original project records are MIT licensed. The third-party demonstration photograph `image.jpg` and its three Month 2 derivative figures have unresolved redistribution provenance. They have been removed from the current public tree and are excluded from research packages. Existing repository history has not been rewritten and does not grant permission to reuse them. The TMLR template repository's Apache 2.0 license and the bibliography file's separate LPPL notice are both retained. The historical IJCAI template has its own provenance. See [NOTICE](NOTICE) and [template provenance](paper/tmlr-source.json).

Historical paths beginning with `previous-runs/` refer to the separate local records archive outside this repository. They are not build dependencies.
