# Response Geometry in Integrated Gradients: A Diffusion-Policy Case Study

This repository contains the active TMLR revision, preserved historical records, and reproducible analysis of those records. The current paper combines exact response-geometry results, a retrospective within-ranking analysis, and an explicitly incomplete numerical case study. It does not claim ranking superiority, learned-weight specificity or behavioral improvement. Local reproduction and scientific checks are documented; human author review and external publication decisions remain pending.

The sole current author is Arjun Bajpai. The existing [Zenodo deposit](https://doi.org/10.5281/zenodo.22133507) and `paper/paper.pdf` are historical versions titled *The Readout, Not the Denoiser*. The deposit has not been changed by this revision. Current source is `paper/paper.tex`; `paper/paper-revision.pdf` and `paper/paper-anonymous-draft.pdf` are explicitly provisional candidates.

## What the evidence establishes

The historical `logpi` target is an auxiliary quadratic discrepancy from one fixed-noise predicted action chunk. It is not the diffusion policy's log likelihood. Rescoring the same recorded ranking and interventions changes median vision deletion AUC from 0.447903 under Q to 0.290759 under stabilized L2. This demonstrates response-geometry sensitivity; it does not establish a better ranking.

Normalized random-order AUC is not universally 0.5. A finite quadratic counterexample with 100 equal additive features gives 0.65976 for every ordering on the historical nine-point grid. Evaluating all feature prefixes instead gives `2/3 - 1/(6n^2)` for `n` features; 2/3 is the continuous-fraction idealization and the limit as `n` increases, not the exact finite-grid area. Actual efficacy comparisons need matched empirical controls. Changing solver step count measures solver-resolution sensitivity and does not exclude denoiser contraction.

The exact Q/L2 gradient relation has a positive, path-dependent scale factor. A smooth two-feature construction shows that this factor can reverse the integrated feature ranking, including at the implemented L2 stabilizer. This is an analytic existence result, not an observed RDT ranking improvement.

A separate smooth construction shows that perfect completeness, identical rankings and identical response curves across several nested integration grids can still disagree with the exact IG ranking. Its coordinate error is 40% in relative L1 despite zero exact completeness error. The CPU example isolates a limitation of finite-grid checks; it does not diagnose the cause of the RDT failures or establish which ranking has better perturbation performance.

The last fully audited numerical snapshot covers 12 completed 170M contexts out of 21 available and 24 planned contexts. All 2,880 repeat-related equality and coverage checks pass, including 1,800 equality checks, while 302 dependent numerical criteria violate their predeclared tolerances. The incomplete roster and failed checks are retained. Repeatability does not establish convergence, and no production integration setting is approved. Broader numerical qualification and prospective ranking/control studies are deferred; their source code and protocols do not constitute experimental results.

Original observation/attribution sidecars, exact historical checkpoint binaries and some primary raw records are unavailable. Scalar fingerprint agreement can detect inconsistencies but cannot authenticate those missing artifacts. Newly downloaded weights, embeddings and collected contexts are identified as new evidence.

## Reproduce the saved-data analysis on CPU

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

The separate [paired rescoring analysis](analysis/paired_rescoring/README.md) measures `AUC_Q - AUC_L2` within each saved curve, with equal-episode mean effects and a paired-call median sensitivity. It retains all eight ranking/modality/direction cases, endpoint orientation, undefined gaps and sampled overshoots. Its conditional episode-bootstrap intervals do not compare the two unauthenticated ranking cohorts or test ranking superiority.
All eight paired medians are positive, while six means are negative because extreme overshooting interventions dominate those means. The paper reports both summaries and traces their disagreement to the saved curves, without trimming the tail.

The aliasing toy and manuscript asset regeneration also require PyTorch, which is absent from the minimal `requirements.txt`. They were checked in the recorded full CPU environment with `torch==2.14.1+cpu` and `torch.version.cuda is None`. Activate that CPU environment before running:

```bash
python -m analysis.revision.nested_grid_aliasing --output runs/nested-grid-aliasing.json
```

The paper's practical evaluation procedure separates attribution target, evaluation response, matched ranking controls, numerical checks and the unit of statistical analysis. A chosen response defines the comparison; passing finite numerical checks does not universally certify an integration budget.

The CPU revision environment used Python 3.12.14. `requirements-cpu-lock.txt` records the installed package versions, including Torch and test dependencies; it is an environment snapshot, not a CPU-wheel-index installation recipe or a fresh-install guarantee. `requirements.txt` is the smaller pinned saved-data analysis layer. Historical notebooks and `make_*_figs.py` generators are archival workflows; their hardcoded constants and saved outputs do not authenticate missing primary data. Do not run historical notebooks over the preserved figures.

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
- `paper/`: current source, official TMLR style/license, provisional builds and preserved historical assets.
- `docs/`: current contracts and explicitly bounded historical demonstrations.

See [scripts/build_paper.md](scripts/build_paper.md) for identified and anonymous builds. The historical PDF is protected from replacement. Final paper and supplement checks include source provenance, anonymity, rights, numerical/statistical validity and full rendered inspection. This repository does not claim journal acceptance or completed submission.

## Citation and licenses

`CITATION.cff` identifies the existing released code/records and historical deposit. Cite that exact version when referring to those released artifacts; the provisional revision has no newly assigned DOI or release version.

Code and original project records are MIT licensed. `image.jpg` and the three Month 2 figures reproducing it have unresolved third-party redistribution provenance and are excluded from new submission packaging. They remain in repository history/current archival files and are not relicensed here. The official TMLR style/bibliography files carry their upstream Apache 2.0 license. The historical IJCAI template has its own provenance. See `NOTICE` and `paper/tmlr-source.json`.

