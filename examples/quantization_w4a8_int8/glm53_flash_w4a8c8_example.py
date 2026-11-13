"""GLM-5.3-Flash A3: routed INT4/A8, shared and other linears INT8/A8.

Uses native model_free_ptq, including MTP checkpoint weights. C8 is a serving
option: vllm serve ... --kv-cache-dtype int8. No local utility module is needed.
Run the separate checkpoint audit before a full-model conversion.
"""

import argparse
import json
from pathlib import Path

IGNORE = [
    r"re:.*norm.*",
    r"re:.*(?:embed_tokens|word_embeddings|position_embeddings|pos_embed).*",
    r"re:.*(?:conv1d|conv_1d|patch_embed\.proj).*",
]


def make_config_groups(algorithm="minmax"):
    groups = {}
    for name, bits, target in (
        ("routed", 4, r"re:.*\.experts\.\d+\.(gate|up|down)_proj$"),
        ("shared_and_other", 8, r"re:^(?!.*\.experts\.\d+\.).*$"),
    ):
        weights = dict(
            num_bits=bits,
            type="int",
            strategy="channel",
            symmetric=True,
            dynamic=False,
            observer="memoryless_minmax",
        )
        if algorithm == "mse" and bits == 4:
            weights.update(observer="memoryless_mse", observer_kwargs={"norm": 2.0})
        groups[name] = dict(
            targets=[target],
            weights=weights,
            input_activations=dict(
                num_bits=8, type="int", strategy="token", symmetric=True, dynamic=True
            ),
            format="pack-quantized" if bits == 4 else "int-quantized",
        )
    return groups


class BareProjectionAliases:
    """Structural Converter protocol: rename bare projections, no quantization math."""

    def process(self, tensors):
        result = {}
        for name, tensor in tensors.items():
            if name.endswith((".fn", ".index_kpool_compress_gate")):
                name += ".weight"
            if name in result:
                raise ValueError(f"Alias collision: {name}")
            result[name] = tensor
        return result

    validate = process

    def get_dependencies(self, weight_name):
        return set()

    def update_config(self, config):
        return config

    update_model_config = update_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="zai-org/GLM-5.3-Flash-BF16")
    parser.add_argument(
        "--revision", default="f9fed3bbc82291073373d67452e9f25317ef7577"
    )
    parser.add_argument(
        "--save-dir", type=Path, default=Path("GLM-5.3-Flash-A3-W4A8-CT")
    )
    parser.add_argument("--algorithm", choices=("minmax", "mse"), default="minmax")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the recipe only; does not audit or load weights",
    )
    args = parser.parse_args()
    groups = make_config_groups(args.algorithm)
    if args.plan_only:
        print(json.dumps(dict(config_groups=groups, ignore=IGNORE), indent=2))
        return

    source = Path(args.model)
    if not source.is_dir():
        from huggingface_hub import snapshot_download

        source = Path(snapshot_download(args.model, revision=args.revision))
    output = args.save_dir.resolve()
    if source.resolve().is_relative_to(output) or output.is_relative_to(
        source.resolve()
    ):
        raise ValueError("Keep source and output in separate directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh empty save-dir")
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401; register NPU backend
    from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme

    from llmcompressor import model_free_ptq

    model_free_ptq(
        model_stub=source,
        save_directory=output,
        config=QuantizationConfig(
            config_groups={k: QuantizationScheme(**v) for k, v in groups.items()},
            ignore=IGNORE,
        ),
        converter=BareProjectionAliases(),
        device=args.device,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
