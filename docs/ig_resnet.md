# ResNet50 attribution demonstration

Updated September 30, 2026. `ig_resnet.py` is a historical ImageNet example. Recorded results below were not rerun by this documentation correction.

Current verification, October 1, 2026 UTC: the revised script completed a CPU-only `m=4` smoke run on a newly generated RGB fixture, with finite raw attributions and verified model-crop display geometry. Its relative completeness residual was 1.802543, so this establishes execution only. The historical photograph was not reused; attribution accuracy and GPU execution remain unverified. See [the E09 CPU verification summary](runtime_verification.md) for the exact model identity, runtime and limitations.

The script uses `ResNet50_Weights.IMAGENET1K_V2`, evaluation mode, frozen parameters and the predicted class's log-softmax score. Freezing parameters avoids their gradient storage; it does not alter the input chain-rule derivative.

Preprocessing resizes the shorter image side to 232, center-crops to 224 by 224, and applies ImageNet normalization. The old photograph example recorded German shepherd, logit 6.24, probability 32.7%.

## Baseline and display

A black RGB image is passed through the same preprocessing. Its normalized channels are approximately `[-2.12,-2.04,-1.80]`. Zero in normalized space instead represents the ImageNet mean color `[0.485,0.456,0.406]`. Either can be a specified reference. The mean-color baseline is not intrinsically invalid, and black is not universally information-free.

The display computes `abs(sum_channels(attr))`, then rescales by its maximum. This differs from `sum_channels(abs(attr))`; opposite signs can cancel before magnitude is taken. Sign carries directional information about the chosen score and baseline. Discarding it is a visualization choice.

The historical overlay stretched the entire original photograph to a square while the model saw an aspect-preserving resize and crop. Those geometries were not aligned. The current optional script displays the actual preprocessed model crop after inverting normalization; it requires explicit image, budget and fresh output arguments. Historical figures were not regenerated. See [legacy workflow disposition](legacy_workflows.md).

## Numerical interpretation

The record reports 13.13% relative completeness residual at m=300 for log-softmax and 7.85% for an earlier raw-logit target. Those are different scalar functions, not a controlled quality comparison. ReLU creates derivative changes, but the log-softmax target means the full scalar gradient is not merely a step function. The record does not identify the number or effect of all activation crossings.

ResNet and ViT differ in architecture, learned parameters, preprocessing and score response. Their measured residual difference cannot be assigned entirely to activation choice. Larger budgets and small scalar residuals do not guarantee accurate coordinates.

Current calls use the [revised core](integrated_gradients.md), including a trapezoidal default. The named legacy rule preserves historical quadrature only. Historical output: `output/ig_resnet50.png`. The photograph and photo-containing figures retain the third-party exception in `NOTICE`; this edit does not resolve their missing rights provenance.
