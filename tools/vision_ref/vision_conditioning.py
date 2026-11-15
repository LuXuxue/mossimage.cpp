#!/usr/bin/env python3
"""Reference: MOSS-Image text conditioning with reference images seen by the Qwen3.5 vision tower.

The text encoder MOSS-Image uses is Qwen3.5-9B, a vision-language model. Its vision tower turns an image into
tokens of the text encoder's width, which take the place of `<|image_pad|>` placeholders in the prompt. The DiT
then conditions on the encoder's last hidden state as usual, so the image is part of the prompt (the approach of
ThetaCursed/ComfyUI-MOSSImageTextEncoder). This script computes that conditioning with transformers' own
Qwen3_5Model, so it is the reference an engine implementation is checked against.

Memory: the encoder file is Comfy-Org's INT8 ConvRot one. Weights are dequantized to bf16 as they are needed: the
vision tower (~0.9 GB) stays on the GPU, each decoder layer is put on the GPU just before it runs and dropped after,
and only the embedding rows a prompt uses are dequantized. It runs on an 8 GB GPU.

    python3 vision_conditioning.py --te qwen3.5_9b_int8_convrot.safetensors --config <bundle>/text_encoder/config.json \
        --tokenizer <bundle>/tokenizer --prompt "..." [--image ref.png ...] [--image-weight 1] [--megapixels 1] \
        --out cond.safetensors
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import time

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
PATCH, TEMPORAL, MERGE = 16, 2, 2


def hadamard(g: int, device) -> torch.Tensor:
    """kron^n(H4) / sqrt(g): the engine's hadamard_sign table (H4[a][b] = -1 exactly when a + b == 3)."""
    h4 = torch.ones(4, 4)
    for a in range(4):
        h4[a, 3 - a] = -1
    h = torch.ones(1, 1)
    while h.shape[0] < g:
        h = torch.kron(h, h4)
    return (h / math.sqrt(g)).to(device)


class Weights:
    """Lazy reader of the INT8 ConvRot file: bf16 tensors as stored, quantized linears dequantized on the GPU."""

    def __init__(self, path: str, device: str):
        self.file = safe_open(path, framework="pt", device="cpu")
        self.keys = set(self.file.keys())
        self.device = device
        self.groups: dict[str, int] = {}
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
            base = 8 + n
            for key, info in header.items():
                if key.endswith(".comfy_quant"):
                    a, b = info["data_offsets"]
                    fh.seek(base + a)
                    meta = json.loads(fh.read(b - a))
                    self.groups[key[: -len(".comfy_quant")]] = int(meta["convrot_groupsize"]) if meta.get("convrot") else 0
        self._h: dict[int, torch.Tensor] = {}

    def tensor(self, name: str) -> torch.Tensor:
        module = name[: -len(".weight")] if name.endswith(".weight") else None
        if module in self.groups:
            return self.dequant(module)
        t = self.file.get_tensor(name)
        return t.to(self.device, torch.bfloat16) if t.is_floating_point() else t.to(self.device)

    def dequant(self, module: str, rows: torch.Tensor | None = None) -> torch.Tensor:
        q = self.file.get_slice(f"{module}.weight")
        s = self.file.get_slice(f"{module}.weight_scale")
        if rows is None:
            qi, sc = q[:], s[:]
        else:
            idx = rows.tolist()
            qi = torch.stack([q[i : i + 1] for i in idx]).squeeze(1)
            sc = torch.stack([s[i : i + 1] for i in idx]).squeeze(1)
        w = qi.to(self.device, torch.float32) * sc.to(self.device, torch.float32)
        g = self.groups[module]
        if g:
            if g not in self._h:
                self._h[g] = hadamard(g, self.device)
            n, k = w.shape
            w = (w.view(n, k // g, g) @ self._h[g]).view(n, k)
        return w.to(torch.bfloat16)


def load_module(module: torch.nn.Module, prefix: str, weights: Weights, device: str) -> None:
    module.to_empty(device=device)
    for name, param in module.named_parameters():
        param.data = weights.tensor(f"{prefix}{name}").to(param.dtype)
    for name, buf in module.named_buffers():
        if f"{prefix}{name}" in weights.keys:
            buf.data = weights.tensor(f"{prefix}{name}")


class LazyEmbedding(torch.nn.Module):
    """embed_tokens that dequantizes only the rows it is asked for."""

    def __init__(self, weights: Weights, device: str):
        super().__init__()
        self.weights, self.device = weights, device

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).cpu()
        uniq, inverse = torch.unique(flat, return_inverse=True)
        rows = self.weights.dequant("model.language_model.embed_tokens", uniq)
        return rows[inverse.to(self.device)].view(*ids.shape, -1)


def smart_resize(h: int, w: int, max_pixels: int, factor: int = PATCH * MERGE) -> tuple[int, int]:
    """Fit within max_pixels keeping the aspect ratio, never upscale, then snap each side to the merge grid."""
    scale = min(1.0, math.sqrt(max_pixels / (h * w)))
    h2 = max(factor, round(h * scale / factor) * factor)
    w2 = max(factor, round(w * scale / factor) * factor)
    while h2 * w2 > max_pixels and min(h2, w2) > factor:
        if h2 >= w2:
            h2 -= factor
        else:
            w2 -= factor
    return h2, w2


def preprocess(path: str, max_pixels: int, weight: float) -> tuple[torch.Tensor, list[int]]:
    """Qwen3-VL image processing: resize, rescale, normalize (mean = std = 0.5), patchify with the temporal pair."""
    image = Image.open(path)
    if image.mode == "RGBA":  # the vision tower sees alpha over white
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, image)
    image = image.convert("RGB")
    h, w = smart_resize(image.height, image.width, max_pixels)
    image = image.resize((w, h), Image.LANCZOS)
    x = torch.from_numpy(__import__("numpy").asarray(image, dtype="float32") / 255.0).permute(2, 0, 1)  # [3, h, w]
    if weight != 1.0:  # fade toward (or push away from) gray, which the tower reads as no signal
        x = (0.5 + weight * (x - 0.5)).clamp(0, 1)
    x = (x - 0.5) / 0.5
    x = x.unsqueeze(0).repeat(TEMPORAL, 1, 1, 1)  # [2, 3, h, w]: a still image is its own temporal pair
    gh, gw = h // PATCH, w // PATCH
    x = x.view(1, TEMPORAL, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)  # grid_t, h/2, w/2, 2, 2, C, T, P, P
    patches = x.reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH)
    return patches, [1, gh, gw]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--te", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--image-weight", type=float, default=1.0)
    ap.add_argument("--megapixels", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-vision", help="also write the vision tower's merged tokens [N, 4096] here")
    args = ap.parse_args()
    device = "cuda"
    t0 = time.time()

    config = Qwen3_5Config.from_dict(json.load(open(args.config)))
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    # Built in bf16 like the checkpoint (F32 tensors of the file are rounded to bf16, as the engine does).
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = Qwen3_5Model(config)
    torch.set_default_dtype(torch.float32)
    model.eval()
    weights = Weights(args.te, device)

    # Vision tower: resident; rotary and position tables are recomputed from the config, not read.
    load_module(model.visual, "model.visual.", weights, device)
    model.visual.rotary_pos_emb.inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, model.visual.rotary_pos_emb.dim, 2, dtype=torch.float, device=device) / model.visual.rotary_pos_emb.dim)
    )
    lm = model.language_model
    lm.embed_tokens = LazyEmbedding(weights, device)
    load_module(lm.norm, "model.language_model.norm.", weights, device)
    lm.rotary_emb.to_empty(device=device)
    inv_freq, _ = lm.rotary_emb.compute_default_rope_parameters(lm.config, device)
    lm.rotary_emb.inv_freq = inv_freq
    lm.rotary_emb.original_inv_freq = inv_freq

    for index, layer in enumerate(lm.layers):
        prefix = f"model.language_model.layers.{index}."

        def before(mod, _args, prefix=prefix):
            load_module(mod, prefix, weights, device)

        def after(mod, _args, _out):
            mod.to("meta")
            torch.cuda.empty_cache()

        layer.register_forward_pre_hook(before)
        layer.register_forward_hook(after)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pixel_values, grids = [], []
    for path in args.image:
        p, g = preprocess(path, int(args.megapixels * 1024 * 1024), args.image_weight)
        pixel_values.append(p)
        grids.append(g)
    text = "<|im_start|>user\n"
    for g in grids:  # one placeholder per merged token
        text += VISION_BLOCK.replace("<|image_pad|>", "<|image_pad|>" * (g[1] * g[2] // (MERGE * MERGE)))
    text += args.prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    mm = (ids == config.image_token_id).int()

    kwargs = {}
    if pixel_values:
        kwargs = {
            "pixel_values": torch.cat(pixel_values).to(device, torch.bfloat16),
            "image_grid_thw": torch.tensor(grids, device=device),
            "mm_token_type_ids": mm,
        }
    if args.dump_vision and pixel_values:
        with torch.no_grad():
            vis = model.visual(kwargs["pixel_values"], grid_thw=kwargs["image_grid_thw"]).pooler_output
        save_file({"vision": vis.to(torch.bfloat16).contiguous().cpu(), "patches": kwargs["pixel_values"].cpu().contiguous()}, args.dump_vision)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, **kwargs).last_hidden_state[0]
    save_file({"conditioning": out.to(torch.bfloat16).contiguous().cpu()}, args.out)
    print(json.dumps({"tokens": out.shape[0], "image_tokens": int(mm.sum()), "grids": grids, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
