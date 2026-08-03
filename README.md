
# A memory allocator, built one bottleneck at a time

Three progressive **rungs** of `malloc`/`free`, each fixing exactly one
weakness of the rung below. The goal is to make allocator design tradeoffs
*visible and measurable* — not to beat glibc.

make # build (uniform -O3)

make bench # throughput + utilization table

## Why not just use glibc malloc?

It's the right answer in production. This rebuilds its core single-threaded
ideas from scratch so the *reasons* behind them are demonstrable. Every rung
carves from one pre-mapped `mmap` arena, so the benchmark measures allocation
**policy**, not the kernel's page-fault path. When the arena fills, `alloc`
returns `nullptr` instead of growing — growing is orthogonal to the policy on show.

## The ladder

| Rung | Allocator | Fixes | Bottleneck it exposes |
|------|-----------|-------|------------------------|
| 1 | Bump | (baseline) | **No reclamation** — footprint only grows |
| 2 | Implicit free list | Reclaim + coalesce | **O(n) search** — walks every block |
| 3 | Explicit free list | Search free blocks only | (the fast rung) |

**Rung 1 — Bump.** A cursor moves forward; `free` is a no-op. Fastest possible
allocation, useless the moment a workload frees anything.

**Rung 2 — Implicit free list.** CS:APP-style `[header][payload][footer]`
blocks; footers make merging neighbours O(1), and a prologue/epilogue pair
removes edge cases. `free` now reclaims and coalesces — but `alloc` is
first-fit over *every* block, allocated or not, so search is linear in block count.

**Rung 3 — Explicit free list.** Free blocks are unused memory, so their
payload stores `{prev, next}` and lives on a doubly linked list. `alloc` only
walks free blocks — a large speedup, at the cost of looser packing (LIFO
insertion spreads allocations out).

## Correctness (`make test`)

Randomized alloc/free trace per rung. Every live block is stamped with a
byte unique to its id, re-verified before each free and at checkpoints —
catches both overlapping blocks and corrupted metadata. Alignment (16 bytes)
is checked on every allocation. Header/footer words go through `memcpy` to
avoid strict-aliasing UB while still compiling to a single `mov` at `-O3`.

## Results (this machine)

2M-op small-object churn:

| rung | throughput (Mops/s) | utilization |
|------|--------------------:|------------:|
| 1_bump     | ~89   | ~0% (never reclaims) |
| 2_implicit | ~0.7  | ~85% |
| 3_explicit | ~38   | ~65% |

The headline isn't one speedup number — it's the tradeoff: implicit packs
tightest but is ~50x slower; explicit is fast but looser.

## Design notes

- One file, one shared harness and block format — keeps the rungs diffable
  against each other.
- Uniform `-O3`, no per-function pragmas — speed differences come from the
  algorithm, not the optimizer.
- No `-march` — no SIMD here, so ISA level doesn't matter.

## Possible extensions (out of scope, worth discussing)

- **Segregated free lists** — one list per size class, near-O(1) placement.
- **Thread caches** — per-thread free lists over a locked central slab.
- **Large-object path** — map big requests straight to pages.
