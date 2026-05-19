# 64ram

Experimental 64GB RAM optimization track for [`antirez/ds4`](https://github.com/antirez/ds4).

This repo is being used to explore whether DS4 / DeepSeek V4 Flash can run on a 64GB RAM laptop while preserving the full model and server/tool functionality.

## Status

This is a **progress/design checkpoint**, not a completed low-memory runtime yet.

The upstream DS4 q2-imatrix model path targets 96GB/128GB-class machines. A 64GB machine is unlikely to work reliably by only lowering context or changing compile flags. The promising path is a MoE-aware memory system.

## Core idea

DeepSeek V4 Flash is a routed MoE model. DS4’s fixed model shape uses many routed experts, but only a small subset of experts are active per token. The 64GB strategy is therefore:

- keep dense/shared model parts resident;
- keep routed expert tensors compressed;
- load/cache routed experts on demand by `(layer, expert, tensor_kind)`;
- prefetch likely next experts asynchronously;
- use a bounded expert cache, for example 24–40GB;
- keep context small at first, for example `--ctx 4096` or `--ctx 8192`;
- keep disk KV cache enabled for agent/client reuse.

## Target future command shape

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

## Docs added so far

- [`README_64GB_PROGRESS.md`](README_64GB_PROGRESS.md) — progress tracker and next implementation tasks.
- [`docs/64gb-lowmem-expert-cache.md`](docs/64gb-lowmem-expert-cache.md) — proposed low-memory expert-cache design.

## Next milestone

The first safe milestone is not maximum speed. It is:

```text
ctx=4096
nothink
single-user server
q2-imatrix
disk KV enabled
expert cache enabled
correct output compared with non-lowmem for a short deterministic prompt
```

After correctness is proven, optimize warm-cache speed and prefetching.
