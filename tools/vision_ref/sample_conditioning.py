#!/usr/bin/env python3
"""Draw an image from an imported conditioning, through moss-server's component session.

The positive row is a conditioning computed outside the engine (vision_conditioning.py) or, with --engine-prompt,
the engine's own encode_text of a prompt; the negative row is the engine's encode of --negative with layer 22
skipped, as mi_generate() does. The loop is mi_generate()'s Euler / MOSS schedule with guided velocity
(c-api.md, "Component API"), so with --engine-prompt the image matches `generate`.

    python3 sample_conditioning.py --server build/dev/moss-server --bundle <bundle> --dit <dit.safetensors> \
        --te <te.safetensors> --vae <vae.safetensors> (--conditioning cond.safetensors | --engine-prompt "...") \
        --width 1024 --height 768 --steps 20 --cfg 3 --seed 7 --out image.png
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from safetensors.torch import load_file


class Server:
    def __init__(self, path: str):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr)
        self.next_id = 0

    def call(self, method: str, params: dict | None = None, timeout: float = 1800) -> dict:
        self.next_id += 1
        msg = {"id": self.next_id, "method": method, **({"params": params} if params is not None else {})}
        self.p.stdin.write((json.dumps(msg) + "\n").encode())
        self.p.stdin.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"moss-server exited during {method}")
            m = json.loads(line)
            if m.get("id") != self.next_id or "event" in m:
                if m.get("event") == "log" and m.get("level") in ("warning", "error"):
                    print(f"[server {m['level']}] {m.get('message')}", file=sys.stderr)
                continue
            if "error" in m:
                raise RuntimeError(f"{method}: {m['error']}")
            return m["result"]
        raise TimeoutError(method)

    def close(self) -> None:
        try:
            self.call("shutdown", timeout=60)
        finally:
            self.p.wait(timeout=60)


class Region:
    """A server-created shared-memory region, mapped and checked (server-protocol.md, Shared memory)."""

    def __init__(self, shm: dict):
        self.info = shm
        fd = os.open("/dev/shm/" + shm["name"].lstrip("/"), os.O_RDWR)
        try:
            self.map = mmap.mmap(fd, 0)
        finally:
            os.close(fd)
        magic, version, _pid = struct.unpack_from("<8sII", self.map, 0)
        token = self.map[16:32].hex()
        payload = struct.unpack_from("<Q", self.map, 40)[0]
        if magic != b"MOSSSHM\0" or version != 1 or token != shm["token"]:
            raise RuntimeError("shared-memory region failed its header check")
        self.payload = payload

    def write(self, offset: int, data: bytes) -> None:
        self.map[offset : offset + len(data)] = data

    def read(self, offset: int, n: int) -> bytes:
        return bytes(self.map[offset : offset + n])


def scratch(server: Server, nbytes: int) -> Region:
    return Region(server.call("shm_alloc", {"bytes": nbytes})["shm"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--dit", required=True)
    ap.add_argument("--te", required=True)
    ap.add_argument("--vae", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--conditioning")
    src.add_argument("--engine-prompt")
    ap.add_argument("--negative", default="")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    s = Server(args.server)
    try:
        s.call("load", {"model": args.bundle, "tier": "high", "dit_file": args.dit, "text_encoder_file": args.te, "vae_file": args.vae})
        s.call("session_open", {"max_rows": 2})
        prompts = [args.negative] if args.conditioning else [args.engine_prompt, args.negative]
        encoded = s.call("encode_text", {"prompts": prompts})["conditionings"]
        if args.conditioning:
            cond = load_file(args.conditioning)["conditioning"]
            raw = cond.contiguous().view(__import__("torch").int16).numpy().tobytes()
            tokens = cond.shape[0]
            region = scratch(s, len(raw))
            region.write(region.payload, raw)
            positive = s.call("conditioning_import", {"shm": region.info["name"], "dtype": "bf16", "shape": [tokens, 4096]})
            negative = encoded[0]
        else:
            positive, negative = encoded
        w, h = args.width, args.height
        latent_bytes = 128 * (h // 16) * (w // 16) * 4
        noise = scratch(s, latent_bytes)
        s.call("initial_noise", {"seed": args.seed, "width": w, "height": h, "shm": noise.info["name"]})
        x = np.frombuffer(noise.read(noise.payload, latent_bytes), dtype=np.float32).copy()
        run = s.call(
            "run_begin",
            {
                "width": w,
                "height": h,
                "steps": args.steps,
                "cfg": args.cfg,
                "guidance": "moss",
                "output": "guided_velocity",
                "exact_sigma": True,
                "rows": [{"conditioning": positive["id"]}, {"conditioning": negative["id"], "skip_layers": [22]}],
            },
        )
        staging = Region(run["shm"])
        sigmas = np.array(run["sigmas"], dtype=np.float32)
        xo, vo = run["tensors"]["x"], run["tensors"]["out"][0]
        t0 = time.time()
        for i in range(len(sigmas) - 1):
            staging.write(xo, x.tobytes())
            s.call("denoise_step", {"sigma": float(sigmas[i])})
            v = np.frombuffer(staging.read(vo, latent_bytes), dtype=np.float32)
            step = np.float32(sigmas[i + 1] - sigmas[i])
            x = (x + step * v).astype(np.float32)
        s.call("run_end")
        lat = scratch(s, latent_bytes)
        lat.write(lat.payload, x.tobytes())
        rgb = scratch(s, h * w * 3)
        s.call("vae_decode", {"shm": lat.info["name"], "width": w, "height": h, "out_shm": rgb.info["name"]})
        Image.fromarray(np.frombuffer(rgb.read(rgb.payload, h * w * 3), dtype=np.uint8).reshape(h, w, 3)).save(args.out)
        print(json.dumps({"out": args.out, "steps": len(sigmas) - 1, "denoise_s": round(time.time() - t0, 1), "positive_tokens": positive["tokens"]}))
        s.call("session_close")
    finally:
        s.close()


if __name__ == "__main__":
    main()
