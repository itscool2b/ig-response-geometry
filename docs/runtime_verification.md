# Optional demo CPU verification

On October 1, 2026 UTC, the current ResNet-50, ViT-B/16 and TinyLlama demos completed one bounded CPU smoke run each. The tested source was commit `b8010c26bb97e77ba8633223f4584d80f2f8e26e`. These are execution checks of the revised scripts. They do not regenerate the historical examples or establish attribution accuracy, ranking quality, efficacy or GPU numerical validity.

The protocols were recorded before their respective runs: `E09_resnet50_cpu_v1.json`, `E09_vit_b_16_cpu_v1.json` and `E09_tinyllama_cpu_v1.json`. Each used `--device cpu --m 4`, fp32 arithmetic and the trapezoid rule, with two CPU compute threads, one interop thread and a 1,200-second timeout. CUDA visibility was disabled before process startup and CUDA remained uninitialized. A transparent harness retained the existing IG result object's input, baseline, attribution and diagnostics without changing the path computation. It also bound model loading to the exact files identified below. Raw inputs, outputs, logs and model identities were sealed and independently hash-verified after retrieval.

## Inputs and geometry

The vision fixture was a self-generated 640 by 360 RGB image, not the historical photograph. For integer pixel coordinates `(x,y)`, its channels were `floor(255*x/639)`, `floor(255*y/359)` and `180*((floor(x/32)+floor(y/24)) mod 2)+30`. The top-left 80 by 40 block was red and the bottom-right 80 by 40 block was green. The PNG SHA256 was `1a48366a19b355fedb1af8884db59e706aa4cfa474c56aa1ccf562ade28814e4`.

ResNet resized the shorter side to 232 and ViT to 256, then both center-cropped to 224 by 224. Reconstructing the displayed model input and comparing it with an independent resize/crop produced maximum absolute RGB error `5.960464477539063e-08` for both. The resulting attribution tensors had shape `[1,3,224,224]`. This verifies the current display geometry for the tested fixture.

TinyLlama used the prompt `A robot moves a cube.` and selected next-token ID 13, a newline. Its PAD/EOS token ID was 2. Its raw embedding attribution had shape `[1,7,2048]`. The reference was the learned PAD-token embedding at every input position.

## Runtime results

| Demo | Total process time, seconds | IG time, seconds | Absolute completeness residual | Relative completeness residual |
|---|---:|---:|---:|---:|
| ResNet-50 | 3.484731 | 0.431616 | 1.007698 | 1.802543 |
| ViT-B/16 | 6.494136 | 2.189320 | 0.842720 | 0.498183 |
| TinyLlama-1.1B-Chat | 52.377479 | 5.088275 | 14.356575 | 6.155900 |

Every saved input, baseline and attribution was finite. The large residuals are retained as measured: four integration intervals are insufficient for an accuracy claim here. No budget-doubling or map-convergence test was part of this smoke protocol. The runtimes include model loading/downloads and shared-host activity and are not performance comparisons. GPU execution of these optional demos remains unverified in this revision.

The environment was Python 3.12.3, Torch 2.8.0+cu128, torchvision 0.23.0+cu128 and transformers 4.41.0. The CUDA-capable Torch build was used only on CPU. TinyLlama loaded 1,100,048,384 fp32 parameters on CPU. The total model download size across all three runs was 2,651,333,872 bytes, below the recorded 6 GB cap.

## Model identity

| Model | Exact identity |
|---|---|
| ResNet-50 | `ResNet50_Weights.IMAGENET1K_V2`; checkpoint SHA256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` |
| ViT-B/16 | `ViT_B_16_Weights.IMAGENET1K_V1`; checkpoint SHA256 `c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9` |
| TinyLlama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`; revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`; model.safetensors SHA256 `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933` |

See the [ResNet](ig_resnet.md), [ViT](ig_vit.md) and [TinyLlama](ig_tinyllama.md) pages for target, baseline and display definitions. The [numerical contract](integrated_gradients.md) explains why scalar completeness alone cannot certify attribution coordinates.
