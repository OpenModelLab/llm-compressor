"""GLM CT deployment regressions using real safetensors headers."""

import copy
import importlib.util
import json
import stat
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


def repair_module():
    path = Path(__file__).resolve().parents[4] / (
        "examples/quantization_w4a8_int8/glm53_flash_ct_config.py"
    )
    spec = importlib.util.spec_from_file_location("glm_ct_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkpoint(tmp_path):
    groups = {}
    for name, bits, suffix in (
        ("dense", 8, r"\.(gate|up|down)_proj$"),
        ("routed", 4, r"\.experts\.\d+\.(gate|up|down)_proj$"),
        ("shared", 8, r"\.shared_experts\.(gate|up|down)_proj$"),
    ):
        groups[name] = dict(
            targets=[
                r"re:^(?:model\.language_model|language_model\.model)"
                r"\.layers\.(?:0|1)\.mlp" + suffix
            ],
            format="pack-quantized" if bits == 4 else "int-quantized",
            weights=dict(num_bits=bits, type="int", strategy="channel", symmetric=True),
            input_activations=dict(
                num_bits=8, type="int", strategy="token", symmetric=True, dynamic=True
            ),
        )
    config = dict(
        text_config=dict(num_hidden_layers=2),
        quantization_config=dict(
            quant_method="compressed-tensors",
            format="mixed-precision",
            config_groups=groups,
            ignore=["model.visual.blocks.0.attn.qkv"],
            sparsity_config={},
            transform_config={},
            version="test",
        ),
    )
    tensors = {}
    for module in (
        "layers.0.mlp",
        "layers.1.mlp.experts.0",
        "layers.1.mlp.shared_experts",
    ):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            stem = f"model.language_model.{module}.{projection}"
            if ".experts." in stem:
                tensors[stem + ".weight_packed"] = torch.zeros(8, 2, dtype=torch.int32)
                tensors[stem + ".weight_shape"] = torch.tensor([8, 16])
            else:
                tensors[stem + ".weight"] = torch.zeros(8, 16, dtype=torch.int8)
            tensors[stem + ".weight_scale"] = torch.ones(8, 1, dtype=torch.bfloat16)
    tensors["model.language_model.layers.2.mlp.experts.0.up_proj.weight"] = torch.ones(
        8, 16, dtype=torch.bfloat16
    )
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path, config, tensors


def write_config(folder, config):
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")


@pytest.mark.parametrize("failure", ["hf_regex", "missing_shared"])
def test_repair_preserves_weights_metadata_and_mtp_scope(checkpoint, failure):
    folder, config, tensors = checkpoint
    module = repair_module()
    groups = config["quantization_config"]["config_groups"]
    if failure == "missing_shared":
        del groups["shared"]
    else:
        for group in groups.values():
            group["targets"] = [
                target.replace(
                    r"(?:model\.language_model|language_model\.model)",
                    r"model\.language_model",
                )
                for target in group["targets"]
            ]
        config["quantization_config"]["config_groups"] = {
            f"group_{i}": group for i, group in enumerate(groups.values())
        }
    write_config(folder, config)
    before = (folder / "config.json").read_bytes()
    before_mode = stat.S_IMODE((folder / "config.json").stat().st_mode)
    weights = (folder / "model.safetensors").read_bytes()
    with pytest.raises(ValueError, match="Expected one target match"):
        module.audit_config(folder)
    assert (folder / "config.json").read_bytes() == before
    result = module.audit_config(folder, repair=True)
    assert result["projections"] == dict(dense=3, routed=3, shared=3)
    assert Path(result["backup"]).read_bytes() == before
    assert stat.S_IMODE((folder / "config.json").stat().st_mode) == before_mode
    assert (folder / "model.safetensors").read_bytes() == weights
    updated = json.loads((folder / "config.json").read_text())
    expected = copy.deepcopy(config)
    expected["quantization_config"]["config_groups"] = updated["quantization_config"][
        "config_groups"
    ]
    assert updated == expected
    mtp = "model.language_model.layers.2.mlp.experts.0.up_proj"
    assert not any(
        module._matches(mtp, group["targets"])
        for group in updated["quantization_config"]["config_groups"].values()
    )
    assert module.audit_config(folder)["changed"] is False
    assert module.audit_config(folder, repair=True)["changed"] is False


@pytest.mark.parametrize(
    "failure", ["scale", "float_weight", "ignore", "overlap", "non_mlp"]
)
def test_reject_inconsistent_artifact_without_writing(checkpoint, failure):
    folder, config, tensors = checkpoint
    stem = "model.language_model.layers.1.mlp.shared_experts.down_proj"
    if failure == "scale":
        del tensors[stem + ".weight_scale"]
    elif failure == "float_weight":
        tensors[stem + ".weight"] = tensors[stem + ".weight"].float()
    elif failure == "ignore":
        config["quantization_config"]["ignore"].append(stem)
    elif failure == "non_mlp":
        tensors["model.language_model.layers.0.self_attn.q_proj.weight"] = torch.zeros(
            8, 16, dtype=torch.int8
        )
    else:
        groups = config["quantization_config"]["config_groups"]
        groups["duplicate"] = copy.deepcopy(groups["shared"])
    save_file(tensors, folder / "model.safetensors")
    write_config(folder, config)
    before = (folder / "config.json").read_bytes()
    with pytest.raises(ValueError):
        repair_module().audit_config(folder)
    assert (folder / "config.json").read_bytes() == before
    assert not list(folder.glob("*.bak_ct_metadata.*"))


def test_group_names_do_not_affect_matching(checkpoint):
    folder, config, _ = checkpoint
    module = repair_module()
    expected = module.audit_config(folder)
    groups = config["quantization_config"]["config_groups"]
    config["quantization_config"]["config_groups"] = {
        f"arbitrary_{i}": group for i, group in enumerate(groups.values())
    }
    write_config(folder, config)
    assert module.audit_config(folder) == expected


def test_broad_mtp_target_is_restricted_by_repair(checkpoint):
    folder, config, _ = checkpoint
    for group in config["quantization_config"]["config_groups"].values():
        group["targets"] = [
            target.replace(
                r"re:^(?:model\.language_model|language_model\.model)"
                r"\.layers\.(?:0|1)\.mlp",
                r"re:.*\.mlp",
            )
            for target in group["targets"]
        ]
    write_config(folder, config)
    module = repair_module()
    with pytest.raises(ValueError, match="Floating MTP projection is targeted"):
        module.audit_config(folder)
    assert module.audit_config(folder, repair=True)["changed"] is True
    assert module.audit_config(folder)["changed"] is False


def test_index_missing_key_rejected(checkpoint):
    folder, _, tensors = checkpoint
    weight_map = {key: "model.safetensors" for key in tensors}
    weight_map["missing.weight"] = "model.safetensors"
    (folder / "model.safetensors.index.json").write_text(
        json.dumps(dict(weight_map=weight_map)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="index and shard keys differ"):
        repair_module().audit_config(folder, repair=True)
    assert not list(folder.glob("*.bak_ct_metadata.*"))
