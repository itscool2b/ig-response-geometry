# R1 regeneration kit: one end-to-end result from preserved inputs

This kit lets a reader regenerate one prospective result of the NHSJS-R1 campaign from scratch: P context `e0030-c017`, PickCube-v1, the public RDT-170M checkpoint. It covers:
- the replayed reference action;
- the historical bf16 IG maps and the 513-node maps for Q and N in vision, language and state;
- every method, fixed-order and random-order ranking;
- every intervention mask and predicted active action;
- the insertion and deletion areas under Q, N and RMS.

## What the kit contains

| Path | What it is |
|---|---|
| `workspace/nhsjs/banks/r1/P/` | The complete P collection run: 60 episodes, 1,500 policy calls. Each call's observation image, proprioception, fp32 diffusion noise, reference action, masks, control frequency and policy seed sit in hash-bound sidecars. Bank authentication checks every episode, so the whole run is included. |
| `workspace/nhsjs/banks/r1/P-bank.json` | The bank: two calls per episode, drawn by the locked uniform rule. |
| `workspace/nhsjs/protocols/r1-p.json` | The production protocol. It binds the bank hash and the hashes of every worker source file. |
| `workspace/tmlr/lang-20260930/` | The T5-XXL language embeddings and the language baseline used at collection. |
| `stored/P/e0030-c017/` | The stored bf16 and scoring phase outputs of the target context, with their completion records (a kit for a context in the protocol's fp32 subset also carries its fp32 phase, which scoring needs). |
| `code/` | The code the regeneration runs, as git blobs of tag `nhsjs-r1-3`, with a per-file manifest: the import closure of the worker, the regeneration script, the pinned-model download and the collector, plus the RDT-170M config, the pod script, the license and a third-party notice (`scripts/build_regeneration_code.py`). Every source file that the worker and the collection pin by hash is included; the worker sources are byte-identical to `nhsjs-r1-1`, which produced the stored outputs. |
| `requirements-r1-venv.txt` | The 57 packages installed in the campaign venv on top of the image. |
| `MANIFEST.sha256` | A SHA-256 for every file above. |

## Steps

The paths matter, because the protocol records absolute paths under `/workspace`. Below, `$KIT` is the absolute path of the copied kit (for example `export KIT=/root/kit`).

1. Start a Linux GPU machine with an NVIDIA RTX PRO 6000 Blackwell (96 GB) for a bit-identical result. The campaign used RunPod image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (Python 3.12.3, torch 2.8.0+cu128, driver 595.91.07). Any other GPU gives a tolerance comparison instead (below). About 40 GB of disk is enough.
2. Copy the kit to the machine and check it:
   `cd $KIT && sha256sum -c --quiet MANIFEST.sha256`
3. Recreate the campaign paths:
   `mkdir -p /workspace && cp -a $KIT/workspace/. /workspace/`
4. Unpack and check the code:
   `mkdir -p /workspace/nhsjs/code/nhsjs-r1-3 && tar -x -f $KIT/code/*.tar -C /workspace/nhsjs/code/nhsjs-r1-3 && (cd /workspace/nhsjs/code/nhsjs-r1-3 && sha256sum -c --quiet $KIT/code/*.sha256)`
5. Fetch the RDT upstream code at its pinned commit:
   `git clone https://github.com/thu-ml/RoboticsDiffusionTransformer.git /workspace/tmlr/rdt-upstream && git -C /workspace/tmlr/rdt-upstream checkout --detach cd79363a1387e8f81c7724d070ef7e45fd23150f`
6. Build the venv on top of the image's torch:
   `python -m venv --system-site-packages /workspace/tmlr/venv && /workspace/tmlr/venv/bin/pip install --no-deps -r $KIT/requirements-r1-venv.txt`
   Replay refuses to run unless torch 2.8.0+cu128, numpy 1.26.4, diffusers 0.27.2 and transformers 4.41.0 are installed exactly.
7. Download the pinned models and check their hashes:
   `cd /workspace/nhsjs/code/nhsjs-r1-3 && HF_HOME=/workspace/huggingface /workspace/tmlr/venv/bin/python scripts/prepare_pinned_models.py`
8. Run the pod checks from the code directory: `cd /workspace/nhsjs/code/nhsjs-r1-3 && bash scripts/revision_pod.sh setup`. This installs the Vulkan loader for the simulator, which the replay itself does not need, and checks the upstream commit, the GPU and the package versions.
9. Regenerate and compare:
   `HF_HOME=/workspace/huggingface RDT_SOURCE=/workspace/tmlr/rdt-upstream HF_HUB_OFFLINE=1 /workspace/tmlr/venv/bin/python scripts/regenerate_result.py gpu --protocol /workspace/nhsjs/protocols/r1-p.json --bank-file /workspace/nhsjs/banks/r1/P-bank.json --bank-id P --context e0030-c017 --stored-root $KIT/stored --fresh-root /workspace/fresh --out /workspace/regeneration-report.json` (from the code directory)
10. Check the stored scoring on CPU alone. Every curve and area is recomputed from the stored predicted actions:
    `/workspace/tmlr/venv/bin/python scripts/regenerate_result.py cpu --protocol /workspace/nhsjs/protocols/r1-p.json --bank-file /workspace/nhsjs/banks/r1/P-bank.json --bank-id P --root $KIT/stored --contexts e0030-c017`

## What to expect

- **Same GPU type, same wheels.** The report says `bit_identical`: every stored map, ranking, mask, action and area is reproduced exactly. The worker replays the stored reference action first and stops if it differs by even one bit.
- **Any other GPU or software.** The report says `within_tolerance` when every map is within relative L1 0.05 and Spearman 0.95 of the stored map and every area within 0.02. It says `outside_tolerance` otherwise. These tolerances are characterization limits, not accuracy guarantees.
