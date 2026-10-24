# Fresh-pod regeneration check, 2026-10-24

The regeneration kit was run on a machine that never mounted the campaign volume, using only the kit README.

## Machine

| Item | Value |
|---|---|
| Pod | `pkm86l765v6eui`, RunPod secure cloud, EU-RO-1 (the campaign ran in US-NC-2) |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, `GPU-be9eb783-1eff-7523-58eb-19845e05fffb`, driver 595.91.07 |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, Python 3.12.3, torch 2.8.0+cu128 |
| Host | `25676a539265`; full record in `fresh-pod-identity.json` |
| Time | created 01:38:07Z, terminated 02:15:32Z (`delete-pod` 204, then `get-pod` 404) |

## Kit

| Variant | Files | MANIFEST.sha256 SHA-256 |
|---|---|---|
| identified | 1,702 | `0650d5e0f65c4d2291fc529e1a377de839914cd9e92b1fd31a8d30d3f3d1827b` |
| anonymous | 1,702 | `491e7256b3559ea3ad8198a8082759a96e569dc26ae029287a47f6c6538a8944` |

Both kits were built on campaign pod 3 by `scripts/build_regeneration_kit.py` from tag `nhsjs-r1-3`. They differ only under `code/`. The identified kit was copied to the fresh pod as `/root/kit`. For the anonymous run, only its code archive was copied.

## Runs

| Run | Script | Context | Computed in the campaign on | Result |
|---|---|---|---|---|
| README steps 2 to 10 | `fresh-pod-check.log` | P `e0030-c017` | pod 3 (`8268bf1eb0cb`) | `bit_identical` |
| Anonymous code archive | `fresh-pod-anon.log` | P `e0030-c017` | pod 3 | `bit_identical` |
| Cross-host | `fresh-pod-cross-host.log` | P `e0008-c024` | pod 1 (`415d25582a09`) | `bit_identical` |
| Cross-host | `fresh-pod-cross-host.log` | P `e0009-c012` | pod 2 (`039d6bdedd9b`) | `bit_identical` |

Each GPU report (`regeneration-report*.json`) shows the same counts:
- **bf16 phase:** 596 of 596 stored maps identical, with none differing, missing or extra. `rankings.json` is byte-equal, and the phase record is equal apart from timing.
- **Scoring phase:** the record is equal apart from timing. All 2,466 normalized and raw areas match, with a largest difference of 0.0, no definedness mismatch, and no missing or extra rankings.

Each CPU report (`regeneration-cpu-report*.json`) authenticated the bf16 and scoring phases and all three modality results. It recomputed every curve and area from the stored predicted actions, schedules and masks.

`prepare-models.json` shows that every pinned model file downloaded on the fresh pod matched its recorded SHA-256.
