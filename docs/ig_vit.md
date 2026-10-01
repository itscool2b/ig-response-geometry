# ViT-B/16 attribution demonstration

Updated September 30, 2026. `ig_vit.py` attributes an ImageNet log-softmax score to input pixels. It uses `ViT_B_16_Weights.IMAGENET1K_V1`, evaluation mode and a black reference processed through the model transforms. Its original measurements are historical.

Current verification, October 1, 2026 UTC: the revised script completed a CPU-only `m=4` smoke run on a newly generated RGB fixture, with finite raw pixel attributions and verified model-crop display geometry. Its relative completeness residual was 0.498183, so this establishes execution only. The historical photograph was not reused; attribution accuracy and GPU execution remain unverified. See [the E09 CPU verification summary](runtime_verification.md) for the exact model identity, runtime and limitations.

ViT-B/16 divides the 224 by 224 image into 196 patches, each 16 by 16 pixels, and projects each patch into a 768-coordinate token. The script computes pixel attributions. A dense patch projection has different weight columns for different pixel/channel coordinates, so sharing the projection does not require equal attribution within a patch. Visible patch-shaped patterns are observations, not a guarantee.

## Historical comparison

| m | ViT recorded residual | ResNet recorded residual |
|---|---:|---:|
| 64 | 15.08% | 44.01% |
| 128 | 1.08% | 48.09% |
| 300 | 0.14% | 13.13% |

The ViT residual decreased at these three budgets. Smooth GELU does not guarantee monotone finite-budget convergence. Architecture, learned weights, preprocessing and scalar responses differ, so the comparison does not isolate smoothness as the cause.

For an independent mathematical example, `cos(128*pi*alpha)` is smooth and integrates to zero on `[0,1]`. Its historical endpoint averages at m=63,64,128 are approximately 0.015625,1,0.007752. A finite grid can poorly resolve a smooth function.

The old classifier output was German shepherd with logit 9.15 and probability 89.7%. These values and `output/ig_vit.png` were not regenerated with the [revised core](integrated_gradients.md). The figure and source photograph have been removed from the current public tree because their redistribution provenance is unresolved. Supply an image you are authorized to process.

## Display limitations

The heatmap sums signed RGB-channel attributions per pixel, then takes magnitude. The tensor is still a 224 by 224 pixel map unless explicitly aggregated into patches. Its old patch-grid attribution label did not reflect such an aggregation.

The historical model crop and stretched original-image overlay had different geometry. The current optional script displays the actual preprocessed crop after inverting normalization and labels pixel-coordinate IG correctly. It requires explicit image, budget and fresh output arguments; historical figures were not regenerated. See [legacy workflow disposition](legacy_workflows.md). The original photograph and derivative figure remain covered by the exception in `NOTICE`, not this repository's MIT license.
