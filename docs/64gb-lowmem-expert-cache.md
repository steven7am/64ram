# 64GB Low-Memory Expert Cache Design

## Goal

Make DS4 run on a 64GB RAM laptop with the full DeepSeek V4 Flash q2-imatrix model, preserving server/tool functionality as much as possible.

## Why expert caching

The model is routed MoE. The routed expert tensors dominate model size, but only a small subset is needed at each layer/token.

So the main memory optimization should be expert-level demand loading, not whole-layer paging.

## Proposed data structures

```c
typedef struct {
    int layer;
    int expert;
    int tensor_kind;          /* gate/up/down or packed equivalent */
    const void *mmap_ptr;     /* source span inside GGUF mapping */
    uint64_t mmap_len;
    void *cache_ptr;          /* resident cached copy or device-visible span */
    uint64_t cache_len;
    uint64_t last_used_tick;
    float route_score_ema;
    bool loading;
    bool resident;
} ds4_expert_cache_entry;

typedef struct {
    uint64_t max_bytes;
    uint64_t used_bytes;
    uint64_t tick;
    int prefetch_distance;
    ds4_expert_cache_entry *entries;
    int entries_len;
} ds4_expert_cache;
```

## Runtime flow

Normal flow:

```text
prompt/token -> layer -> attention -> router top-k -> routed experts -> output
```

Low-memory flow:

```text
prompt/token -> layer -> attention -> router top-k
              -> ensure selected expert tensors are resident
              -> prefetch likely next expert tensors
              -> routed experts -> output
```

## Loading policy

1. Use the existing mmap-backed GGUF as the source of truth.
2. Keep expert tensors in their quantized format.
3. Do not expand q2/q2-like expert weights into f16/f32 permanently.
4. Copy or register only the selected expert spans.
5. Evict entries when `used_bytes > max_bytes`.

## Prefetch policy

The first prototype can use simple prefetch:

- After layer `L` router top-k is known, prefetch selected expert tensors for layer `L` immediately.
- Maintain an EMA route score per expert.
- During idle/overlap windows, prefetch top EMA experts for upcoming layers.
- Later, add a tiny predictor keyed by current layer top-k.

## User flags

Suggested flags:

```text
--lowmem
--expert-cache-gb N
--expert-prefetch N
--expert-cache-stats
```

Suggested environment variables:

```text
DS4_LOWMEM=1
DS4_EXPERT_CACHE_GB=32
DS4_EXPERT_PREFETCH=2
DS4_EXPERT_CACHE_VERBOSE=1
```

## Safe first milestone

A good first milestone is not maximum speed. It is:

```text
ctx=4096
nothink
single-user server
q2-imatrix
disk KV enabled
expert cache enabled
correct output compared with non-lowmem for a short deterministic prompt
```

## Speed milestones

1. Cold-cache generation works.
2. Warm-cache generation improves.
3. Repeated prompt/session with disk KV avoids full prefill.
4. Async prefetch reduces expert-load stalls.
5. Hot-cache agent loop becomes usable.
