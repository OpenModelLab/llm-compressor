"""
Perplexity evaluation for the FlatQuant W4A4 experiments, following the
evaluation protocol of "FlatQuant: Flatness Matters for LLM Quantization"
(arXiv:2410.09426, Section 4): WikiText-2 and C4 perplexity, computed over
contiguous blocks of 2048 tokens (GPTQ-style evaluation).

Works both with FP16 baselines and with compressed-tensors checkpoints
produced by llama_flatquant_w4a4.py (transformers loads them transparently
and runs the quantized forward on the fly).

Examples:

  python evaluate_ppl.py --model meta-llama/Llama-2-7b-hf --dataset wikitext2
  python evaluate_ppl.py --model Llama-2-7b-hf-flatquant-full-W4A4 \\
      --dataset wikitext2
  python evaluate_ppl.py --model Llama-2-7b-hf-flatquant-full-W4A4 --dataset c4
"""

import argparse
import math

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_eval_text(dataset: str) -> str:
    if dataset == "wikitext2":
        data = load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="test"
        )
        return "\n\n".join(data["text"])

    if dataset == "c4":
        # standard C4 evaluation subset used by GPTQ-style evaluations:
        # first validation shard of the English partition
        data = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        return "\n\n".join(data["text"][:256])

    raise ValueError(f"unknown dataset: {dataset}")


@torch.no_grad()
def evaluate_ppl(model, input_ids: torch.Tensor, block_len: int, batch: int) -> float:
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, input_ids.shape[0], batch):
        blocks = input_ids[start : start + batch].to(model.device)
        logits = model(blocks).logits[:, :-1].float()
        targets = blocks[:, 1:]
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
        )
        total_nll += nll.item()
        total_tokens += targets.numel()
    return math.exp(total_nll / total_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="HF model id or path to a (compressed) local checkpoint",
    )
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=args.dtype, device_map="auto"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    text = load_eval_text(args.dataset)
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encodings.input_ids

    block_len = args.max_seq_length
    n_blocks = input_ids.shape[1] // block_len
    if n_blocks == 0:
        raise ValueError(
            f"evaluation text is shorter than one block of {block_len} tokens"
        )
    blocks = input_ids[:, : n_blocks * block_len].view(-1, block_len)

    ppl = evaluate_ppl(model, blocks, block_len, args.batch)
    print(f"{args.model} | {args.dataset} | ppl = {ppl:.4f}")


if __name__ == "__main__":
    main()
