import torch
from torch import nn
import torch.nn.functional as F


class Qwen35ForCausalLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(self, hf_config):
        super().__init__()
        if hasattr(hf_config, "text_config"):
            hf_config = hf_config.text_config

        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        hf_cfg = Qwen3_5TextConfig(**hf_config.to_dict())
        hf_cfg._attn_implementation = "flash_attention_2"
        self.model = nn.Module()
        self.model.language_model = Qwen3_5TextModel(hf_cfg)
        self.model.language_model.reset_cache = self.reset_cache
        self._hf_config = hf_cfg
        self._past_key_values = None

    def reset_cache(self):
        self._past_key_values = None

    def forward(self, input_ids, positions):
        from nanovllm.utils.context import get_context
        ctx = get_context()

        pkv = self._past_key_values
        if ctx.is_prefill:
            self._past_key_values = None
            pkv = None

        output = self.model.language_model(
            input_ids.unsqueeze(0),
            attention_mask=None,
            position_ids=positions.unsqueeze(0),
            past_key_values=pkv,
            use_cache=True,
        )
        self._past_key_values = output.past_key_values
        return output.last_hidden_state.squeeze(0)

    def compute_logits(self, hidden_states):
        weight = self.model.language_model.embed_tokens.weight
        return F.linear(hidden_states, weight)
