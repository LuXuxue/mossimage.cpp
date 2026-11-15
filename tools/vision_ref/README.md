# Vision-conditioned prompts: reference and findings

MOSS-Image's text encoder is Qwen3.5-9B, a vision-language model. Its vision tower can turn a reference image into
tokens of the encoder's width that replace `<|image_pad|>` placeholders in the prompt; the DiT then conditions on the
last hidden state as usual (the approach of ThetaCursed/ComfyUI-MOSSImageTextEncoder). The engine does not do this
yet. These two scripts are the reference an engine implementation is checked against, and the experiment that says
whether it is worth building.

| Script | Does |
| --- | --- |
| `vision_conditioning.py` | The conditioning `[tokens, 4096]` with transformers' own `Qwen3_5Model`, from Comfy-Org's INT8 ConvRot encoder file, dequantized as needed (vision tower resident, decoder layers one at a time): fits an 8 GB GPU, about 13 s |
| `sample_conditioning.py` | Draws an image from an imported conditioning through moss-server's component session (`conditioning_import`, Euler / MOSS schedule, guided velocity, uncond row skipping layer 22) |

## What was checked (RTX 4070 Laptop 8 GB, MOSS-Image SFT INT8, tier high, 20 steps, CFG 3)

- **The sampling loop is `generate`'s.** With the engine's own `encode_text`, the image is bit-identical to `generate` with the same parameters.
- **The reference encoder is right.** Its text-only conditioning (bf16 weights) drawn through the same loop gives an image within PSNR 30.1 dB of the engine's (INT8 activations): the expected precision gap, not a different model.
- **A whole scene as the reference takes over the image.** A 1200×864 panel at 0.5 MP (494 image tokens against 25 prompt tokens) reproduced the panel's harbour, pose and paw prints and ignored the prompt's café, at image weight 1, 0.6 and 0.35 alike. Fading toward gray does not balance it. The engine's `weight` therefore guides the sampling instead, between the predictions with and without the images ([c-api.md](../../docs/c-api.md#vision-references)).
- **A character crop with few tokens carries the character, not the scene.** The same character cropped to the head and shoulders at 0.1 MP (100 image tokens): the prompt's café, window and book were drawn, with the reference's short black hair, round glasses and jacket. Text alone drew a different girl (long hair, no glasses).

So the feature is worth building, for character references: a portrait crop and a small token budget (about 0.1 MP)
by default. A front end should ask for character portraits, not scenes.

## In the engine

`generate`'s `vision_references` (`MI_CAP_VISION_REFERENCES`; [c-api.md](../../docs/c-api.md#vision-references)) is this,
checked against these scripts by `tests/unit/test_qwen35_vision.cpp` (set `MOSS_VISION_*` as its header says). On the
test image, `generate` with the image gives the same picture, bit for bit, as this loop drawing the engine's own
conditioning.

## What the engine implementation needed

- The vision tower: patch embedding (Conv3d 3×2×16×16 as a GEMM), bilinear position embedding from a 48×48 table, 27 blocks (LayerNorm with bias, qkv with bias, 2-D rotary over head dim 72, bidirectional attention per image, GELU-tanh MLP 1152→4304), the 2×2 merger (LayerNorm, 4608→4608 GELU, →4096). Linear weights are INT8 ConvRot with group 64 (the text encoder's are 256).
- The text encoder: image placeholder rows replaced by the vision output; interleaved mRoPE for image tokens (sections 11, 11, 10; text tokens keep equal t, h, w), continuing positions after the image from its largest position + 1; the embedding cache keyed on the images too.
- A `generate` parameter (images, weight, megapixels) and a `hello` capability, so front ends offer it only on engines that have it.
