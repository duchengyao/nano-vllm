import torch
from torch import nn
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
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
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim * 2
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = ColumnParallelLinear(
            hidden_size, self.total_num_heads * head_dim * 2, bias=attn_bias,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size, self.total_num_kv_heads * head_dim, bias=attn_bias,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size, self.total_num_kv_heads * head_dim, bias=attn_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * head_dim, hidden_size, bias=attn_bias,
        )
        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=rms_norm_eps)

        rope_theta = rope_scaling.get("rope_theta", 10000000) if isinstance(rope_scaling, dict) else 10000000
        partial_factor = rope_scaling.get("partial_rotary_factor", 1.0) if isinstance(rope_scaling, dict) else 1.0
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=int(self.head_dim * partial_factor),
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads, self.head_dim, self.scaling, self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q, gate = q.split([self.num_heads * self.head_dim, self.num_heads * self.head_dim], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        o = o.flatten(1, -1)
        o = o * torch.sigmoid(gate)
        output = self.o_proj(o)
        return output


class Qwen35MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False,
        )
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen35LinearAttentionWrapper(nn.Module):

    def __init__(self, hf_module: nn.Module):
        super().__init__()
        self.hf_module = hf_module

    def forward(self, positions, hidden_states):
        return self.hf_module(hidden_states.unsqueeze(0)).squeeze(0)


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


class Qwen35HFAttentionWrapper(nn.Module):

    def __init__(self, hf_attn, num_kv_heads, head_dim, max_position, rope_theta, partial_factor):
        super().__init__()
        self.hf_attn = hf_attn
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        from nanovllm.layers.attention import Attention
        from nanovllm.layers.rotary_embedding import get_rope
        self.nano_attn = Attention(head_dim, head_dim, 1.0, num_kv_heads)
        self.rotary_emb = get_rope(head_dim, int(head_dim * partial_factor), max_position, rope_theta)
        self._cache_populated = False

    def forward(self, positions, hidden_states):
        from nanovllm.utils.context import get_context
        ctx = get_context()
        bs = hidden_states.shape[0]

        if ctx.is_prefill:
            # Compute Q,K,V using HF projections, store K,V in nano cache
            hf_inp = hidden_states.unsqueeze(0)
            k_out = self.hf_attn.k_proj(hf_inp).squeeze(0)
            k = k_out.reshape(bs, self.num_kv_heads, self.head_dim)
            v_out = self.hf_attn.v_proj(hf_inp).squeeze(0)
            v = v_out.reshape(bs, self.num_kv_heads, self.head_dim)
            q_out = self.hf_attn.q_proj(hf_inp).squeeze(0)
            half = q_out.shape[-1] // 2
            q, gate = q_out.split([half, half], dim=-1)
            q = q.reshape(bs, -1, self.head_dim)
            k = self.hf_attn.k_norm(k)
            q = self.hf_attn.q_norm(q)
            q, k = self.rotary_emb(positions, q, k)
            # Use nano attn for prefill (flash_attn_varlen_func)
            o = self.nano_attn(q, k, v)
            o = o.reshape(bs, -1) * torch.sigmoid(gate)
            o = self.hf_attn.o_proj(o)
            self._cache_populated = True
            return o
        else:
            if self._cache_populated:
                hf_inp = hidden_states.unsqueeze(0)
                q = self.hf_attn.q_proj(hf_inp).squeeze(0)
                k = self.hf_attn.k_proj(hf_inp).squeeze(0)
                v = self.hf_attn.v_proj(hf_inp).squeeze(0)
                q, gate = q.split([q.shape[-1] // 2, q.shape[-1] // 2], dim=-1)
                q = q.reshape(1, -1, self.head_dim)
                k = k.reshape(1, self.num_kv_heads, self.head_dim)
                v = v.reshape(1, self.num_kv_heads, self.head_dim)
                q = self.hf_attn.q_norm(q)
                k = self.hf_attn.k_norm(k)
                q, k = self.rotary_emb(positions, q, k)
                o = self.nano_attn(q, k, v)
                o = o.reshape(1, -1) * torch.sigmoid(gate)
                o = self.hf_attn.o_proj(o)
                return o.squeeze(0)
            else:
                hf_inp = hidden_states.unsqueeze(0)
                pos_3d = positions.unsqueeze(0)
                # Fallback HF forward
                attn_out, _ = self.hf_attn(hf_inp, position_embeddings=(
                    torch.zeros(1, int(self.head_dim * 0.25), device=hidden_states.device),
                    torch.zeros(1, int(self.head_dim * 0.25), device=hidden_states.device),
                ), attention_mask=None)
                return attn_out.squeeze(0)


class Qwen35DecoderLayer(nn.Module):

    def __init__(
        self,
        text_config,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        is_full_attention = layer_idx % text_config.full_attention_interval == text_config.full_attention_interval - 1

        if is_full_attention:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention as HFAttn
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
            hf_cfg = Qwen3_5TextConfig(**text_config.to_dict())
            hf_cfg._attn_implementation = "flash_attention_2"
            hf_attn = HFAttn(hf_cfg, layer_idx)
            rope_cfg = getattr(text_config, "rope_parameters", {})
            self.self_attn = Qwen35HFAttentionWrapper(
                hf_attn, text_config.num_key_value_heads, text_config.head_dim,
                text_config.max_position_embeddings,
                rope_cfg.get("rope_theta", 10000000) if isinstance(rope_cfg, dict) else 10000000,
                rope_cfg.get("partial_rotary_factor", 1.0) if isinstance(rope_cfg, dict) else 1.0,
            )
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
            hf_cfg = Qwen3_5TextConfig(**text_config.to_dict())
            self.linear_attn = Qwen3_5GatedDeltaNet(hf_cfg, layer_idx)

        self.mlp = Qwen35MLP(
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            hidden_act=text_config.hidden_act,
        )
        self.input_layernorm = Qwen35RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        cache: GatedDeltaNetCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def __init__(
        self,
        text_config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(text_config.vocab_size, text_config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen35DecoderLayer(text_config, i) for i in range(text_config.num_hidden_layers)
        ])
        self.norm = Qwen35RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self._cache: GatedDeltaNetCache | None = None

    def reset_cache(self):
        self._cache = GatedDeltaNetCache(len(self.layers))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual, self._cache)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen35ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        hf_config,
    ) -> None:
        super().__init__()
        if hasattr(hf_config, "text_config"):
            text_config = hf_config.text_config
        else:
            text_config = hf_config
        self._text_config = text_config
        self.model = nn.Module()
        self.model.language_model = Qwen35Model(text_config)
        self.lm_head = ParallelLMHead(text_config.vocab_size, text_config.hidden_size)
        if text_config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.language_model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.language_model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
