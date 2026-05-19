# DS4 64GB RAM Progress Track

This repository is intended to explore a 64GB-RAM execution path for `antirez/ds4` while preserving full DeepSeek V4 Flash functionality.

## Current status

This is a planning/prototyping update, not a finished runtime implementation.

The upstream DS4 q2 model is still larger than a 64GB laptop can comfortably hold as a normal resident working set. The practical path is not just lowering context size or changing compiler flags. The practical path is to make DS4 MoE-aware at the memory-management layer.

## Proposed approach

DeepSeek V4 Flash in DS4 is a routed MoE model. DS4 uses many routed experts but only a small subset of routed experts per token. That means most expert tensors are inactive for any given token.

The 64GB plan is:

1. Keep dense/shared model parts resident.
2. Keep active KV/session state small and use disk KV persistence.
3. Keep routed expert tensors compressed.
4. Cache routed experts by `(layer, expert, tensor_kind)`.
5. Load missing expert tensors on demand from the mmap-backed GGUF.
6. Prefetch likely next-layer experts asynchronously.
7. Evict cold experts with an LRU/score policy.
8. Add a user-facing mode such as:

```sh
./ds4-server \
  -m ds4flash.gguf \
  --ctx 4096 \
  --nothink \
  --kv-disk-dir /tmp/ds4-kv \
  --kv-disk-space-mb 32768 \
  --lowmem \
  --expert-cache-gb 32 \
  --expert-prefetch 2
```

## Expected tradeoff

This should preserve model quality better than a new ultra-low-bit quant, but it will trade some speed for fit. The target is a usable hot-cache workflow on a 64GB laptop, not matching 96GB/128GB fully-resident performance.

Cold-start and first-token latency may be slow. Repeated coding-agent turns should improve after the expert cache and disk KV cache warm up.

## Next implementation tasks

- Identify all routed expert tensor names and byte ranges during GGUF load.
- Add `ds4_expert_cache` metadata and an LRU eviction layer.
- Split routed expert access from the existing always-bound tensor assumptions.
- Add async prefetch hooks after router top-k selection.
- Add memory-budget flags to CLI and server.
- Add benchmark mode comparing:
  - no lowmem mode
  - lowmem cold cache
  - lowmem warm cache
  - lowmem with/without prefetch
- Add correctness checks using existing logprob/vector tests.

## Non-goals for first prototype

- Do not invent a q1 quant first.
- Do not remove tool calling.
- Do not remove server API compatibility.
- Do not make CPU-only the main path.
