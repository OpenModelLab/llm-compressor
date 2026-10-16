"""
Reproduce the W4A4 experiments of "FlatQuant: Flatness Matters for LLM
Quantization" (arXiv:2410.09426, Section 4) with llm-compressor's
FlatQuantModifier.

Paper setup (Section 4.1, "Implementation Details"):
  - models: LLaMA-2 (7B/13B/70B) and LLaMA-3 (8B/70B)
  - quantization: W4A4 with per-channel symmetric INT4 weights, per-token
    symmetric INT4 activations, and group-128 asymmetric INT4 KV cache
  - calibration: 128 sequences of 2048 tokens from WikiText-2, batch size 4
  - FlatQuant training: 15 epochs, AdamW, lr 5e-3 (clip factors 10x) with
    cosine annealing, Kronecker factors via Cayley parameterization ("svd")
  - weight quantizer: RTN (default) or GPTQ, giving the paper's
    "FlatQuant-RTN" / "FlatQuant-GPTQ" variants

Examples:

  # FlatQuant-RTN W4A4 on LLaMA-2-7B (paper Table 1)
  python llama_flatquant_w4a4.py --model meta-llama/Llama-2-7b-hf

  # FlatQuant-GPTQ W4A4
  python llama_flatquant_w4a4.py \\
      --model meta-llama/Llama-2-7b-hf --weight-quantizer gptq

  # component ablation on LLaMA-3-8B (paper Table 4):
  #   lt = learnable Kronecker transform only
  #   lt_ps = + per-channel scaling (learnable diagonal)
  #   lt_ps_lct = + learnable activation clipping thresholds
  #   full = + learnable weight clipping (complete method)
  python llama_flatquant_w4a4.py --model meta-llama/Meta-Llama-3-8B \\
      --ablation lt

  # RTN baseline without any transformation (paper "RTN" row)
  python llama_flatquant_w4a4.py --model meta-llama/Llama-2-7b-hf --variant rtn
"""

import argparse

from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import FlatQuantModifier

# W4A4 scheme from the paper: symmetric INT4 weights quantized per channel,
# symmetric INT4 activations quantized dynamically per token. The dynamic
# per-token activation scheme is required by FlatQuantModifier (its fake
# activation quantizers used while training are per-token dynamic).
W4A4_SCHEME = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=4,
        type="int",
        symmetric=True,
        strategy="channel",
        dynamic=False,
    ),
    input_activations=QuantizationArgs(
        num_bits=4,
        type="int",
        symmetric=True,
        strategy="token",
        dynamic=True,
    ),
)

# group-128 asymmetric INT4 KV cache, as used in the paper
KV_CACHE_SCHEME = QuantizationArgs(
    num_bits=4,
    type="int",
    symmetric=False,
    strategy="group",
    group_size=128,
    dynamic=False,
    observer="static_minmax",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-7b-hf",
        help="HF model id, e.g. meta-llama/Llama-2-7b-hf, "
        "meta-llama/Llama-2-13b-hf, meta-llama/Meta-Llama-3-8B",
    )
    parser.add_argument(
        "--variant",
        default="flatquant",
        choices=["flatquant", "rtn"],
        help="'flatquant' runs FlatQuantModifier before quantization, 'rtn' "
        "quantizes directly and serves as the paper's RTN baseline",
    )
    parser.add_argument(
        "--weight-quantizer",
        default="rtn",
        choices=["rtn", "gptq"],
        help="weight quantizer used after the transform: 'rtn' maps to "
        "QuantizationModifier (paper FlatQuant-RTN), 'gptq' maps to "
        "GPTQModifier (paper FlatQuant-GPTQ)",
    )
    parser.add_argument(
        "--ablation",
        default="full",
        choices=["lt", "lt_ps", "lt_ps_lct", "full"],
        help="FlatQuant component ablation (paper Table 4): 'lt' learnable "
        "Kronecker transform only; 'lt_ps' adds per-channel scaling; "
        "'lt_ps_lct' adds learnable activation clipping; 'full' additionally "
        "enables learnable weight clipping (complete method)",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=128,
        help="number of calibration sequences (paper: 128)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="sequence length of each calibration sample (paper: 2048)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="calibration batch size (paper: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="FlatQuant training epochs per unit (paper: 15)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-3,
        help="FlatQuant base learning rate (paper: 5e-3)",
    )
    parser.add_argument(
        "--diag-alpha",
        type=float,
        default=0.3,
        help="interpolation for the per-channel scaling initialization",
    )
    parser.add_argument(
        "--tran-type",
        default="svd",
        choices=["svd", "inv"],
        help="Kronecker factor parameterization (paper: orthogonal Cayley)",
    )
    parser.add_argument("--seed", type=int, default=0, help="transform init seed")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory for the compressed checkpoint "
        "(default: <model>-<variant><-ablation>-W4A4)",
    )
    return parser.parse_args()


def build_recipe(args: argparse.Namespace) -> list:
    recipe = []

    if args.variant == "flatquant":
        # component ablation from paper Table 4:
        #   LT  = learnable transform, PS = per-channel scaling (learnable
        #   diagonal), LCT = learnable activation clipping, LWC = learnable
        #   weight clipping
        add_diag = args.ablation in ("lt_ps", "lt_ps_lct", "full")
        lac = args.ablation in ("lt_ps_lct", "full")
        lwc = args.ablation == "full"

        recipe.append(
            FlatQuantModifier(
                epochs=args.epochs,
                lr=args.lr,
                diag_alpha=args.diag_alpha,
                add_diag=add_diag,
                tran_type=args.tran_type,
                seed=args.seed,
                w_bits=4,
                a_bits=4,
                w_sym=True,
                a_sym=True,
                lwc=lwc,
                lac=lac,
            )
        )

    if args.weight_quantizer == "gptq":
        recipe.append(
            GPTQModifier(
                config_groups={"group_0": W4A4_SCHEME},
                ignore=["lm_head"],
                kv_cache_scheme=KV_CACHE_SCHEME,
            )
        )
    else:
        recipe.append(
            QuantizationModifier(
                config_groups={"group_0": W4A4_SCHEME},
                ignore=["lm_head"],
                kv_cache_scheme=KV_CACHE_SCHEME,
            )
        )

    return recipe


def main() -> None:
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    recipe = build_recipe(args)

    model_short = args.model.split("/")[-1]
    variant_name = (
        args.variant if args.variant == "rtn" else f"{args.variant}-{args.ablation}"
    )
    output_dir = args.output_dir or f"{model_short}-{variant_name}-W4A4"

    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        dataset="wikitext",
        splits="train",
        batch_size=args.batch_size,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
        shuffle_calibration_samples=False,
    )

    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    print(f"saved compressed checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
