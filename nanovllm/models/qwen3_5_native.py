import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear, ColumnParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen35RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            orig_dtype = x.dtype
            x = x.float()
            var = x.pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (x * (1.0 + self.weight.float())).to(orig_dtype)
        else:
            orig_dtype = x.dtype
            x = x.float().add_(residual.float())
            residual = x.to(orig_dtype)
            var = x.pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (x * (1.0 + self.weight.float())).to(orig_dtype), residual


class _DeltaLayerState:
    __slots__ = ("conv_states", "recurrent_states")

    def __init__(self):
        self.conv_states = None
        self.recurrent_states = None


class GatedDeltaNetCache:

    def __init__(self, num_layers: int):
        self._layers = [_DeltaLayerState() for _ in range(num_layers)]

    def has_previous_state(self, layer_idx: int) -> bool:
        return self._layers[layer_idx].conv_states is not None

    def update_conv_state(self, state, layer_idx: int):
        self._layers[layer_idx].conv_states = state.detach()

    def update_recurrent_state(self, state, layer_idx: int):
        self._layers[layer_idx].recurrent_states = state.detach()

    @property
    def layers(self):
        return self._layers


class Qwen35Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position: int = 262144,
        rms_norm_eps: float = 1e-06,
        attn_bias: bool = False,
        rope_theta: float = 10000000,
        partial_factor: float = 1.0,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        assert tp_size == 1, "TP not supported for Qwen3.5 native yet"
        self.total_num_heads = num_heads
        self.num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim * 2, bias=attn_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attn_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=attn_bias)
        self.q_norm = Qwen35RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(head_dim, eps=rms_norm_eps)

        self.rotary_emb = get_rope(
            head_dim, int(head_dim * partial_factor),
            max_position, rope_theta,
        )
        self.attn = Attention(num_heads, head_dim, self.scaling, num_kv_heads)

    def forward(self, positions, hidden_states):
        bs = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(bs, self.num_heads, -1)
        k = self.k_proj(hidden_states).view(bs, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bs, self.num_kv_heads, self.head_dim)
        q, gate = q.chunk(2, dim=-1)  # (bs, nh, hd) each
        q = q.contiguous()
        gate = gate.contiguous()
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        o = o.reshape(bs, -1) * torch.sigmoid(gate.reshape(bs, -1))
        return self.o_proj(o)


class Qwen35MLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        assert dist.get_world_size() == 1, "TP not supported"
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen35DecoderLayer(nn.Module):

    def __init__(self, text_config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        interval = text_config.full_attention_interval
        is_full = layer_idx % interval == interval - 1
        rope = getattr(text_config, "rope_parameters", {}) or {}

        if is_full:
            self.self_attn = Qwen35Attention(
                hidden_size=text_config.hidden_size,
                num_heads=text_config.num_attention_heads,
                num_kv_heads=text_config.num_key_value_heads,
                head_dim=text_config.head_dim,
                max_position=text_config.max_position_embeddings,
                attn_bias=getattr(text_config, "attention_bias", False),
                rope_theta=rope.get("rope_theta", 10000000),
                partial_factor=rope.get("partial_rotary_factor", 1.0),
            )
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
            hf_cfg = Qwen3_5TextConfig(**text_config.to_dict())
            self.linear_attn = Qwen3_5GatedDeltaNet(hf_cfg, layer_idx)

        self.mlp = Qwen35MLP(text_config.hidden_size, text_config.intermediate_size)
        self.input_layernorm = Qwen35RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual, cache=None):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if hasattr(self, "self_attn"):
            hidden_states = self.self_attn(positions, hidden_states)
        else:
            if cache is not None:
                hidden_states = self.linear_attn(
                    hidden_states.unsqueeze(0), cache_params=cache,
                ).squeeze(0)
            else:
                hidden_states = self.linear_attn(hidden_states.unsqueeze(0)).squeeze(0)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen35Model(nn.Module):

    def __init__(self, text_config):
        super().__init__()
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen35DecoderLayer(text_config, i) for i in range(text_config.num_hidden_layers)
        ])
        self.norm = Qwen35RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self._cache: GatedDeltaNetCache | None = None

    def reset_cache(self):
        self._cache = GatedDeltaNetCache(len(self.layers))

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, self._cache)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen35ForCausalLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(self, hf_config):
        super().__init__()
        if hasattr(hf_config, "text_config"):
            hf_config = hf_config.text_config
        self.model = nn.Module()
        self.model.language_model = Qwen35Model(hf_config)
        self.lm_head = ParallelLMHead(hf_config.vocab_size, hf_config.hidden_size)
        if hf_config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.language_model.embed_tokens.weight.data

    def forward(self, input_ids, positions):
        return self.model.language_model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
