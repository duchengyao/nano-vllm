import os
import sys
import time
import argparse
from random import randint, seed


SCENARIOS = {
    "long-prefill": dict(
        desc="多长输入并发 — 测试 chunked prefill 调度效率",
        num_seqs=64,
        min_input=3072,
        max_input=4096,
        min_output=64,
        max_output=128,
    ),
    "kv-pressure": dict(
        desc="KV Cache 过载 — 测试 preemption/swap 策略",
        num_seqs=256,
        min_input=1024,
        max_input=2048,
        min_output=512,
        max_output=1024,
    ),
    "decode-heavy": dict(
        desc="长输出主导 — 测试纯 decode 吞吐",
        num_seqs=128,
        min_input=64,
        max_input=256,
        min_output=1024,
        max_output=2048,
    ),
    "balanced": dict(
        desc="均衡负载 (原始 bench.py 场景)",
        num_seqs=256,
        min_input=100,
        max_input=1024,
        min_output=100,
        max_output=1024,
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(SCENARIOS), required=True)
    parser.add_argument("--engine", choices=["nanovllm", "vllm"], required=True)
    parser.add_argument("--no-cuda-graph", action="store_true", default=False,
                        help="Disable CUDA graph (enforce eager mode)")
    args = parser.parse_args()

    cfg = SCENARIOS[args.scenario]
    enforce_eager = args.no_cuda_graph

    seed(0)

    if args.engine == "nanovllm":
        from nanovllm import LLM, SamplingParams
    else:
        from vllm import LLM, SamplingParams

    path = os.path.expanduser("~/huggingface/models/Qwen3-0.6B/")
    max_model_len = 8192 if args.engine == "nanovllm" and not enforce_eager else 4096
    llm = LLM(path, enforce_eager=enforce_eager, max_model_len=max_model_len)

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(cfg["min_input"], cfg["max_input"]))]
        for _ in range(cfg["num_seqs"])
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(cfg["min_output"], cfg["max_output"]))
        for _ in range(cfg["num_seqs"])
    ]
    if args.engine == "vllm":
        prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    warmup_sp = SamplingParams()
    if args.engine == "vllm":
        warmup_sp = SamplingParams(max_tokens=1)
    llm.generate(["Warmup"], warmup_sp)

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    # Output format: SCENARIO ENGINE TOTAL_TOKENS TIME THROUGHPUT
    print(f"RESULT {args.scenario} {args.engine} {total_tokens} {t:.2f} {throughput:.2f}",
          flush=True)


if __name__ == "__main__":
    main()
