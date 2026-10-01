# Response Geometry in Per-Step Integrated Gradients

This repository contains the active TMLR revision, the preserved historical records, and reproducible analysis of those records. Revision work is in progress. Numerical validation and new paired ranking/control experiments are required before the current draft can be treated as submission-ready.

The sole current author is Arjun Bajpai. The existing [Zenodo deposit](https://doi.org/10.5281/zenodo.22133507) and `paper/paper.pdf` are historical versions titled *The Readout, Not the Denoiser*. The deposit has not been changed by this revision. Current source is `paper/paper.tex`; `paper/paper-revision.pdf` and `paper/paper-anonymous-draft.pdf` are explicitly provisional candidates.

## What the evidence establishes

The historical `logpi` target is an auxiliary quadratic discrepancy from one fixed-noise predicted action chunk. It is not the diffusion policy's log likelihood. Rescoring the same recorded ranking and interventions changes median vision deletion AUC from 0.447903 under Q to 0.290759 under stabilized L2. This demonstrates response-geometry sensitivity; it does not establish a better ranking.

Normalized random-order AUC is not universally 0.5. The equal-feature quadratic counterexample gives 2/3 for the continuous integral and 0.65976 on the historical sampled grid. Actual efficacy comparisons need matched empirical controls. Changing solver step count measures solver-resolution sensitivity and does not exclude denoiser contraction.

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

The CPU revision environment used Python 3.12.14. `requirements-cpu-lock.txt` records its full installed package set, including CPU Torch and test dependencies. `requirements.txt` is the smaller pinned analysis layer. Historical notebooks and `make_*_figs.py` generators are archival workflows; their hardcoded constants and saved outputs do not authenticate missing primary data. Do not run historical notebooks over the preserved figures.

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

