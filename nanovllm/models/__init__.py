from transformers import AutoConfig

from nanovllm.models.qwen2 import Qwen2ForCausalLM
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_5 import Qwen35ForCausalLM as Qwen35HFForCausalLM
from nanovllm.models.qwen3_5_native import Qwen35ForCausalLM as Qwen35NativeForCausalLM


SUPPORTED_MODEL_CLS = {
    "qwen2": Qwen2ForCausalLM,
    "qwen3": Qwen3ForCausalLM,
    "qwen3_5": Qwen35HFForCausalLM,
    "qwen3_5_text": Qwen35HFForCausalLM,
}

# Override with native version if requested
import os as _os
if _os.environ.get("NANOVLLM_QWEN35_NATIVE", "") == "1":
    SUPPORTED_MODEL_CLS["qwen3_5"] = Qwen35NativeForCausalLM
    SUPPORTED_MODEL_CLS["qwen3_5_text"] = Qwen35NativeForCausalLM


def get_model_cls(hf_config: AutoConfig):
    model_type = getattr(hf_config, "model_type", "")
    model_cls = SUPPORTED_MODEL_CLS.get(model_type)
    if model_cls is None:
        supported = ", ".join(sorted(SUPPORTED_MODEL_CLS.keys()))
        raise ValueError(f"Unsupported model_type={model_type!r}. Supported model types: {supported}.")
    return model_cls
