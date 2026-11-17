#!/usr/bin/env python3
"""Convert llm-compressor FlatQuant W4A4 checkpoint to ModelSlim format.

Goal: make the checkpoint loadable by *unpatched* vllm-ascend v0.30.0
(W4A4_FLATQUANT_DYNAMIC via modelslim_config.py), with NO source patches.

Two changes relative to the raw llm-compressor export:

1. **Absorb the online `diagonal` into the upstream (source) module.**
   llm-compressor keeps `x -> d⊙x -> (L⊗R)` online; ModelSlim folds `d`
   into the source weight/norm so the checkpoint carries only L, R, clip_ratio.

   Pairing (llm-compressor default_mappings):
     input_layernorm          -> q_proj, k_proj, v_proj   (norm)
     post_attention_layernorm -> gate_proj, up_proj       (norm)
     up_proj                  -> down_proj                (linear)
     v_proj                   -> o_proj                   (add_diag=False, d==1)

   Absorption is EXACT because (L⊗R) stays online:
     norm  : (x_norm * g) ⊙ d  ==  x_norm * (g ⊙ d)      -> norm.weight *= d
     linear: (x @ W^T) ⊙ d     ==  x @ (W * d)^T         -> weight_scale *= d
   For the int4 linear case only `weight_scale` is rescaled row-wise; the
   int8 values are untouched, so the dequantized weight is exactly W*d.

2. **Rename to ModelSlim flat layout + emit quant_model_description.json.**
     {layer}.{group}_input.left        -> {layer}.left_trans
     {layer}.{group}_input.right       -> {layer}.right_trans
     {layer}.{group}_input.clip_ratio  -> {layer}.clip_ratio
     (diagonal dropped after absorption)
     + {layer}.weight_offset = zeros(out,1) float32  (symmetric -> zero)
     + weight_scale promoted to float32
     + quant_model_description.json with "W4A4_FLATQUANT_DYNAMIC"

Usage:
  python flatquant_to_modelslim.py \
      --checkpoint-dir /path/to/v6_70b_fp32diag \
      --output-dir     /path/to/v6_70b_modelslim
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

TRANS_RE = re.compile(
    r"^(?P<layer>.+?)\.(?P<group>flatquant(?:_\d+)?)_input\.(?P<param>left|right|diagonal|clip_ratio)$"
)

# target linear suffix -> (source kind, source suffix)
TARGET_TO_SOURCE = {
    "q_proj": ("norm", "input_layernorm"),
    "k_proj": ("norm", "input_layernorm"),
    "v_proj": ("norm", "input_layernorm"),
    "gate_proj": ("norm", "post_attention_layernorm"),
    "up_proj": ("norm", "post_attention_layernorm"),
    "down_proj": ("linear", "up_proj"),
    "o_proj": ("skip", None),
}

PARAM_RENAME = {"left": "left_trans", "right": "right_trans", "clip_ratio": "clip_ratio"}
QUANT_TYPE = "W4A4_FLATQUANT_DYNAMIC"


def source_for(layer_name: str):
    """Resolve the upstream module whose output feeds this target linear.

    ``model.layers.N.self_attn.q_proj``  -> ``model.layers.N.input_layernorm``
    ``model.layers.N.mlp.up_proj``       -> ``model.layers.N.post_attention_layernorm``
    ``model.layers.N.mlp.down_proj``     -> ``model.layers.N.mlp.up_proj``
    """
    parts = layer_name.split(".")
    suffix = parts[-1]
    kind, src_suffix = TARGET_TO_SOURCE[suffix]
    if kind == "skip":
        return "skip", None
    if kind == "norm":
        # drop "<submodule>.<target>" (e.g. "self_attn.q_proj")
        block = ".".join(parts[:-2])
        return kind, f"{block}.{src_suffix}"
    # linear: same submodule as the target
    return kind, ".".join(parts[:-1] + [src_suffix])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint_dir).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. copy non-weight files ----
    for item in ckpt.iterdir():
        if item.is_file() and item.suffix not in (".safetensors", ".bin"):
            shutil.copy2(item, out / item.name)

    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    # ---- 2. discover transforms ----
    transforms = defaultdict(dict)
    for key in weight_map:
        m = TRANS_RE.match(key)
        if m:
            transforms[m.group("layer")][m.group("param")] = key
    print(f"[*] {len(transforms)} layers with FlatQuant transforms")

    # ---- 3. absorption plan + read diagonals via safe_open (lazy) ----
    plan = defaultdict(list)          # source -> [(target, diag_key, kind)]
    skipped = []
    for layer, params in sorted(transforms.items()):
        if "diagonal" not in params:
            continue
        kind, src = source_for(layer)
        if kind == "skip":
            skipped.append(layer)
            continue
        plan[src].append((layer, params["diagonal"], kind))

    # group diagonal keys by shard for a single open per shard
    by_shard = defaultdict(list)
    for src, items in plan.items():
        for _, dk, _ in items:
            by_shard[weight_map[dk]].append(dk)

    diag_cache = {}
    for shard, keys in by_shard.items():
        with safe_open(str(ckpt / shard), framework="pt", device="cpu") as sf:
            for k in keys:
                diag_cache[k] = sf.get_tensor(k).float()

    absorb = {}   # source -> (kind, tensor d)
    for src, items in plan.items():
        d = diag_cache[items[0][1]]
        for tgt, dk, _ in items[1:]:
            if not torch.equal(d, diag_cache[dk]):
                raise RuntimeError(f"targets of {src} disagree on diagonal ({tgt})")
        absorb[src] = (items[0][2], d)

    print(f"[*] absorb into {len(absorb)} sources, skip {len(skipped)} (o_proj identity)")

    # ---- 4. process shards ----
    n_norm = n_linear = 0
    for shard in sorted(ckpt.glob("*.safetensors")):
        sd = load_file(str(shard), device="cpu")
        new_sd = {}

        for key, tensor in sd.items():
            m = TRANS_RE.match(key)
            if m:
                param = m.group("param")
                if param == "diagonal":
                    continue
                new_sd[f"{m.group('layer')}.{PARAM_RENAME[param]}"] = tensor.to(torch.float32)
                continue

            if key.endswith(".weight_scale"):
                base = key[: -len(".weight_scale")]
                if base in absorb:
                    kind, d = absorb[base]
                    if kind == "linear":
                        if tensor.shape[0] != d.numel():
                            raise RuntimeError(
                                f"{base}: weight_scale {tuple(tensor.shape)} vs d {tuple(d.shape)}")
                        new_sd[key] = (tensor.float() * d.view(-1, 1))
                        n_linear += 1
                        continue
                new_sd[key] = tensor.to(torch.float32)
                continue

            if key.endswith(".weight"):
                base = key[: -len(".weight")]
                if base in absorb:
                    kind, d = absorb[base]
                    if kind == "norm":
                        if tensor.numel() != d.numel():
                            raise RuntimeError(
                                f"{base}: norm weight {tuple(tensor.shape)} vs d {tuple(d.shape)}")
                        new_sd[key] = (tensor.float() * d).to(tensor.dtype)
                        n_norm += 1
                        continue
                new_sd[key] = tensor
                continue

            new_sd[key] = tensor

        # synthesize weight_offset for quantized linears
        for key in list(new_sd):
            if key.endswith(".weight_scale"):
                base = key[: -len(".weight_scale")]
                if f"{base}.weight" in new_sd and new_sd[f"{base}.weight"].dtype == torch.int8:
                    new_sd[f"{base}.weight_offset"] = torch.zeros(
                        (new_sd[key].shape[0], 1), dtype=torch.float32)

        save_file(new_sd, str(out / shard.name), metadata={"format": "pt"})

    # ---- 5. rebuild index ----
    new_weight_map = {}
    for key, shard in weight_map.items():
        m = TRANS_RE.match(key)
        if m:
            if m.group("param") == "diagonal":
                continue
            new_weight_map[f"{m.group('layer')}.{PARAM_RENAME[m.group('param')]}"] = shard
        else:
            new_weight_map[key] = shard
    for key, shard in list(new_weight_map.items()):
        if key.endswith(".weight_scale"):
            new_weight_map[key.replace(".weight_scale", ".weight_offset")] = shard
    index["weight_map"] = new_weight_map
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    # ---- 6. quant_model_description.json ----
    desc = {}
    for key in new_weight_map:
        if key.endswith((".weight", ".weight_scale", ".weight_offset")):
            base = key.rsplit(".", 1)[0]
            desc[key] = QUANT_TYPE if f"{base}.weight_scale" in new_weight_map else "FLOAT"
        elif key.endswith((".left_trans", ".right_trans", ".clip_ratio")):
            desc[key] = QUANT_TYPE
        else:
            desc[key] = "FLOAT"
    (out / "quant_model_description.json").write_text(json.dumps(
        {**desc, "model_quant_type": QUANT_TYPE, "version": "1.0.0",
         "group_size": 0, "metadata": {}, "optional": {}}, indent=2))

    # ---- 7. config.json quantization metadata ----
    cfg_path = out / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        cfg["quantization_config"] = {"quant_method": "ascend",
                                      "model_quant_type": QUANT_TYPE}
        cfg_path.write_text(json.dumps(cfg, indent=2))

    print(f"\n[+] norm absorptions  : {n_norm}")
    print(f"[+] linear absorptions: {n_linear}")
    print(f"[+] skipped identity  : {len(skipped)}")
    print(f"[+] description entries: {len(desc)}")
    print(f"[+] output: {out}")


if __name__ == "__main__":
    main()