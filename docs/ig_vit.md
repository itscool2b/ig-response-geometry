# ViT-B/16 attribution demonstration

Updated September 30, 2026. `ig_vit.py` attributes an ImageNet log-softmax score to input pixels. It uses `ViT_B_16_Weights.IMAGENET1K_V1`, evaluation mode and a black reference processed through the model transforms. Its original measurements are historical.

ViT-B/16 divides the 224 by 224 image into 196 patches, each 16 by 16 pixels, and projects each patch into a 768-coordinate token. The script computes pixel attributions. A dense patch projection has different weight columns for different pixel/channel coordinates, so sharing the projection does not require equal attribution within a patch. Visible patch-shaped patterns are observations, not a guarantee.

## Historical comparison

| m | ViT recorded residual | ResNet recorded residual |
|---|---:|---:|
| 64 | 15.08% | 44.01% |
| 128 | 1.08% | 48.09% |
| 300 | 0.14% | 13.13% |

The ViT residual decreased at these three budgets. Smooth GELU does not guarantee monotone finite-budget convergence. Architecture, learned weights, preprocessing and scalar responses differ, so the comparison does not isolate smoothness as the cause.

For an independent mathematical example, `cos(128*pi*alpha)` is smooth and integrates to zero on `[0,1]`. Its historical endpoint averages at m=63,64,128 are approximately 0.015625,1,0.007752. A finite grid can poorly resolve a smooth function.

The old classifier output was German shepherd with logit 9.15 and probability 89.7%. These values and `output/ig_vit.png` were not regenerated with the [revised core](integrated_gradients.md).

## Display limitations

The heatmap sums signed RGB-channel attributions per pixel, then takes magnitude. The tensor is still a 224 by 224 pixel map unless explicitly aggregated into patches. Its old patch-grid attribution label did not reflect such an aggregation.

The model's preprocessing crop and the stretched original-image overlay have different geometry. A corrected replacement must use or map the actual model crop. The original photograph and derivative figure remain covered by the exception in `NOTICE`, not this repository's MIT license.
