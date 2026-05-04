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
            self.self_attn = Qwen35Attention(
                hidden_size=text_config.hidden_size,
                num_heads=text_config.num_attention_heads,
                num_kv_heads=text_config.num_key_value_heads,
                head_dim=text_config.head_dim,
                max_position=text_config.max_position_embeddings,
                rms_norm_eps=text_config.rms_norm_eps,
                attn_bias=getattr(text_config, "attention_bias", False),
                rope_scaling=getattr(text_config, "rope_parameters", None),
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if hasattr(self, "self_attn"):
            hidden_states = self.self_attn(positions, hidden_states)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
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
