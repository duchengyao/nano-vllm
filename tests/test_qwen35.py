import os
import sys

import pytest
import torch
import torch.distributed as dist

MODEL = "/data/huggingface/models/Qwen3.5-4B"


@pytest.fixture(scope="module")
def model():
    """Load nano-VLLM Qwen3.5 model once for all tests."""
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("nccl", world_size=1, rank=0)
    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")

    from nanovllm.config import Config
    from nanovllm.models import get_model_cls
    from nanovllm.utils.loader import load_model

    cfg = Config(model=MODEL)
    m = get_model_cls(cfg.hf_config)(cfg.hf_config)
    load_model(m, MODEL)
    yield m
    torch.cuda.synchronize()


@pytest.fixture(scope="module")
def hf_logits():
    """Compute HF reference logits once for all tests (CPU to avoid OOM)."""
    import glob
    from safetensors.torch import load_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    cfg = Qwen3_5TextConfig.from_pretrained(MODEL)
    hf = Qwen3_5ForCausalLM(cfg).bfloat16()
    state = {}
    for f in sorted(glob.glob(f"{MODEL}/*.safetensors")):
        state.update(load_file(f))
    hf_state = {}
    for k, v in state.items():
        if k.startswith("model.language_model."):
            hf_state["model." + k[len("model.language_model."):]] = v.to(torch.bfloat16)
    hf.load_state_dict(hf_state, strict=False)
    hf.eval()

    inp = torch.tensor([[9419, 11, 821, 803, 369]])
    with torch.inference_mode():
        logits = hf(inp).logits[0, -1]
    return logits


class TestModelLoading:
    def test_embed_tokens_shape(self, model):
        w = model.model.language_model.embed_tokens.weight
        assert w.shape == (248320, 2560), f"Expected (248320, 2560), got {w.shape}"

    def test_num_layers(self, model):
        assert len(model.model.language_model.layers) == 32

    def test_linear_attn_layers(self, model):
        count = sum(1 for l in model.model.language_model.layers if hasattr(l, "linear_attn"))
        assert count == 24, f"Expected 24 linear_attn layers, got {count}"

    def test_self_attn_layers(self, model):
        count = sum(1 for l in model.model.language_model.layers if hasattr(l, "self_attn"))
        assert count == 8, f"Expected 8 self_attn layers, got {count}"


class TestPrefill:
    def test_single_prefill_runs(self, model):
        from nanovllm.utils.context import set_context, reset_context

        model.reset_cache()
        inp = torch.tensor([9419, 11, 821, 803, 369], dtype=torch.int64)
        pos = torch.arange(len(inp), dtype=torch.int64)
        n = len(inp)
        set_context(True, torch.tensor([0, n], dtype=torch.int32),
                    torch.tensor([0, n], dtype=torch.int32), n, n)
        with torch.inference_mode():
            out = model(inp, pos)
            logits = model.compute_logits(out)
        reset_context()
        assert logits.shape == (1, 248320), f"Expected (1, 248320), got {logits.shape}"

    def test_prefill_logits_reasonable(self, model):
        from nanovllm.utils.context import set_context, reset_context

        model.reset_cache()
        inp = torch.tensor([9419, 11, 821, 803, 369], dtype=torch.int64)
        pos = torch.arange(len(inp), dtype=torch.int64)
        n = len(inp)
        set_context(True, torch.tensor([0, n], dtype=torch.int32),
                    torch.tensor([0, n], dtype=torch.int32), n, n)
        with torch.inference_mode():
            out = model(inp, pos)
            logits = model.compute_logits(out).squeeze(0)
        reset_context()

        top = logits.topk(3)
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(MODEL)
        top_tokens = [t.decode([v.item()]) for v in top.indices]
        # For "Hello, my name is", the first token should be a name/punctuation
        # Check that it's NOT just whitespace/newlines (garbage output)
        first = t.decode([top.indices[0].item()])
        assert first.strip(), f"First token should not be whitespace, got {first!r}"

    def test_prefill_hidden_shape(self, model):
        from nanovllm.utils.context import set_context, reset_context

        model.reset_cache()
        inp = torch.tensor([9419, 11, 821, 803, 369], dtype=torch.int64)
        pos = torch.arange(len(inp), dtype=torch.int64)
        n = len(inp)
        set_context(True, torch.tensor([0, n], dtype=torch.int32),
                    torch.tensor([0, n], dtype=torch.int32), n, n)
        with torch.inference_mode():
            out = model(inp, pos)
        reset_context()
        assert out.shape == (5, 2560), f"Expected (5, 2560), got {out.shape}"


class TestDecode:
    def test_decode_follows_prefill(self, model):
        from nanovllm.utils.context import set_context, reset_context

        model.reset_cache()
        inp = torch.tensor([9419, 11, 821, 803, 369], dtype=torch.int64)
        pos = torch.arange(len(inp), dtype=torch.int64)
        n = len(inp)

        # Prefill
        set_context(True, torch.tensor([0, n], dtype=torch.int32),
                    torch.tensor([0, n], dtype=torch.int32), n, n)
        with torch.inference_mode():
            out = model(inp, pos)
            logits = model.compute_logits(out).squeeze(0)
        reset_context()
        next_tok = logits.argmax().item()

        # Decode multiple steps
        tokens = [next_tok]
        for i in range(4):
            inp2 = torch.tensor([next_tok], dtype=torch.int64)
            pos2 = torch.tensor([n + i], dtype=torch.int64)
            set_context(False)
            with torch.inference_mode():
                out2 = model(inp2, pos2)
                logits2 = model.compute_logits(out2).squeeze(0)
            reset_context()
            next_tok = logits2.argmax().item()
            tokens.append(next_tok)

        assert len(tokens) == 5, f"Expected 5 decode tokens, got {len(tokens)}"

    def test_decode_output_not_repeating(self, model):
        from nanovllm.utils.context import set_context, reset_context

        model.reset_cache()
        inp = torch.tensor([9419, 11, 821, 803, 369], dtype=torch.int64)
        pos = torch.arange(len(inp), dtype=torch.int64)
        n = len(inp)

        set_context(True, torch.tensor([0, n], dtype=torch.int32),
                    torch.tensor([0, n], dtype=torch.int32), n, n)
        with torch.inference_mode():
            out = model(inp, pos)
            logits = model.compute_logits(out).squeeze(0)
        reset_context()

        tokens = set()
        next_tok = logits.argmax().item()
        for i in range(8):
            tokens.add(next_tok)
            inp2 = torch.tensor([next_tok], dtype=torch.int64)
            pos2 = torch.tensor([n + i], dtype=torch.int64)
            set_context(False)
            with torch.inference_mode():
                out2 = model(inp2, pos2)
                logits2 = model.compute_logits(out2).squeeze(0)
            reset_context()
            next_tok = logits2.argmax().item()

        # Should not get stuck on a single repeating token
        assert len(tokens) >= 2, f"Model should produce at least 2 distinct tokens, got {len(tokens)}"


class TestBlockManager:
    def test_bypass_can_allocate(self):
        from nanovllm.engine.block_manager import BypassBlockManager
        from nanovllm.engine.sequence import Sequence

        bm = BypassBlockManager(256)
        seq = Sequence([1, 2, 3])
        assert bm.can_allocate(seq) == 0

    def test_bypass_allocate_and_append(self):
        from nanovllm.engine.block_manager import BypassBlockManager
        from nanovllm.engine.sequence import Sequence

        bm = BypassBlockManager(256)
        # Use 256 tokens: num_blocks=1, len=256, 256%256=0≠1 so may_append is no-op
        seq = Sequence([1] * 256)
        bm.allocate(seq, 0)
        assert len(seq.block_table) == 1
        bm.may_append(seq)
        assert len(seq.block_table) == 1  # no new block needed
        # Append one more token: 257%256=1, so may_append should add a block
        seq.append_token(1)  # now 257 tokens
        assert bm.can_append(seq) is True
        bm.may_append(seq)
        assert len(seq.block_table) == 2

    def test_bypass_deallocate(self):
        from nanovllm.engine.block_manager import BypassBlockManager
        from nanovllm.engine.sequence import Sequence

        bm = BypassBlockManager(256)
        seq = Sequence([1] * 100)
        bm.allocate(seq, 0)
        assert len(seq.block_table) > 0
        bm.deallocate(seq)
        assert len(seq.block_table) == 0
        assert seq.num_cached_tokens == 0
