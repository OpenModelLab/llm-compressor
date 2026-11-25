# GLM quantization example maintenance

When adding or changing a GLM example that calls `model.save_pretrained`:

- Check the inherited `model.generation_config` for strict save compatibility.
- For the GLM sampling-field issue, follow the existing GPTQ/AWQ guard before
  saving: when `top_p` or `temperature` is not None and `do_sample` is false,
  set `do_sample = True`. Preserve the source sampling field values.
- Explain that this guard also covers neutral defaults and changes the saved
  Transformers default to sampling. Evaluation requests must keep their sampling
  settings explicit. Do not apply it blindly to unrelated models.
- Validate the generation configuration's strict save/reload behavior locally
  before requiring a full-model quantization run. Do not change framework source
  or quantization scope merely to handle this generation configuration issue.
