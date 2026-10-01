# LLaVA demonstration and unresolved numerical causes

Updated September 30, 2026. `ig_llava.py` is a historical LLaVA-1.5-7B example using a 4-bit model. It attributes a next-token log-softmax score to a combined image/text embedding sequence. The original example predicted “The” at 70.0% for “What is this?” on the demonstration photograph.

The intended image representation comprises 576 CLIP patch features projected into 4096-coordinate language embeddings and inserted at image-token positions. This is representation attribution, not pixel IG through CLIP. Image and text summaries sum embedding coordinates at their respective sequence positions.

## Baseline, precision and API

The PAD-embedding reference fills image and text positions. It is a learned reference, not an information-free condition. Positive-epsilon RMSNorm has a finite zero-input Jacobian; see [the normalization explanation](ig_tinyllama.md).

The demo casts embeddings to fp16 for its quantized forward and computes log-softmax in fp32. Revised IG arithmetic is separate and fails on nonfinite values. Better forward precision is a testable intervention, not a guaranteed remedy.

`get_image_features` varies by Transformers version. The old claim that `.pooler_output` must be an unprojected CLS vector is not generally true. The audit inspected Transformers 5.18.0, whose returned structure can contain projected features in that field as well as hidden states. The old dependency and representation identity remain unresolved. Assigning a 1024-coordinate vector to 4096-coordinate positions would normally fail dimension checks rather than silently broadcast as the earlier explanation claimed. Check the actual API, shapes, selected layer, CLS removal and projection before using this legacy path.

Calling `gradient_checkpointing_enable()` alone does not establish that checkpointing is active in evaluation mode. The inspected implementation gates the relevant path on training mode. The old memory estimate therefore does not prove activation. Indiscriminately enabling training can also change the function through dropout.

## Preserved residuals

| m | Recorded relative completeness residual |
|---|---:|
| 300 | 28.26% |
| 1000 | 78.89% |

The explanation that each dequantization uses stochastic rounding whose noise necessarily accumulates with m was not established by repeated-input forward/gradient tests or controlled precision/quantizer comparisons. Averaging independent zero-mean noise would not alone imply increasing variance with more samples.

These observations do not isolate hardware, quantization, integration or an implementation defect as the cause. The promises that fp16/8-bit execution or a larger GPU must restore completeness are withdrawn. A causal diagnosis needs separately identified controls; this edit does not run them.

The historical account reports about 4.4 GB model storage in 4-bit and an 8-bit backward OOM on 12 GB hardware. Those are environmental observations, not current guarantees. Increasing m in sequential IG primarily adds work rather than retaining m graphs simultaneously.

## Visualization and status

The historical `output/ig_llava.png` displayed a 24 by 24 representation map and text bars. Correctly locating tokens on the source photograph depends on the processor's actual crop/resize geometry. A stretched-image overlay alone does not establish alignment. The photograph and figure have been removed from the current public tree because their redistribution provenance is unresolved. Earlier Git history retains the rights exception in `NOTICE`.

The current `ig_llava.py` entrypoint is retired and exits 2; `--status` reports the exact preserved source and reasons without running the model. See [legacy workflow disposition](legacy_workflows.md). This page does not certify the legacy quantized runtime. Current research and numerical validation are described in [ig_rdt.md](ig_rdt.md) and [integrated_gradients.md](integrated_gradients.md).
