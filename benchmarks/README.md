# Benchmarks

Two scripts. `run.py` follows
[Theine's harness](https://github.com/Yiling-J/theine/tree/main/benchmarks)
so Zipf hit rate and decorator reads are comparable. Theine's published
numbers use `nolock=True`. Corto always locks. `phases.py` splits decorate,
miss, and hit.

```text
uv sync --group bench
uv run --group bench python benchmarks/run.py
uv run --group bench python benchmarks/run.py --latency
uv run python benchmarks/phases.py
```

`--quick` is 200k Zipf requests and three capacities. Full `run.py` is 1M
requests and eight capacities.

Numbers below are one Mac, CPython 3.14, Apple Silicon, after the 0.1
freelist / two-arg / three-arg work. Re-run before you quote them.

## Decorator reads (Theine throughput)

Zipf integer keys, `DATA_LEN=65536`, cap=`2 * DATA_LEN`, 3× prefill, then
time the reads.

| impl | ns/read |
|---|---|
| corto.memoize | 110 |
| functools.lru_cache | 141 |
| cachebox.cached(LRU) | 411 |
| cachetools.cached(TTL+Lock) | 869 |
| theine.Memoize (nolock, str key) | 948 |

## Phase timings

Cheap `return x`. `n=200000`. Miss includes the function and the store.
Hit does not run the function.

| path | corto | lru_cache |
|---|---|---|
| miss, one-arg | 154 ns | 108 ns |
| hit, one-arg | 69 ns | 66 ns |
| miss, two-arg | 339 ns | 160 ns |
| hit, two-arg | 165 ns | 108 ns |
| miss, three-arg | 315 ns | 172 ns |
| hit, three-arg | 164 ns | 115 ns |

One-arg hits are already at the `lru_cache` ceiling. Two- and three-arg
wrappers hash the arguments on the hit path and allocate a tuple only on
miss. Miss stays slower: TinyLFU insert (sketch, window, admit) is more
work than an LRU link splice.

`Cache` mapping, same fill: set ~84 ns, get hit ~52 ns, get miss (no store)
~58 ns.

## Hit rate and scan

`run.py` (not `--latency`) replays one Zipf key list per capacity and
prints hit rate plus ns/request. Corto sits a little under Theine (no
hill-climb) and above LRU. The extra scan bench (not in Theine) heats 50
keys, then inserts 40k unique keys. Corto keeps the hot set. LRU does not.

## What a miss costs

A miss always runs the function. Corto then admits. On unique keys that
never repeat, both caches lose to a bare call; Corto loses by more because
the store is heavier (~300–400 ns extra vs ~70 ns for `lru_cache` on a
~1.6 µs `find_handler` walk).

That is the policy, not a defect. One-shot keys stay in the window and do
not occupy the main cache. `lru_cache` will keep the last `maxsize` junk
keys and look fast if you immediately replay them.

Do not decorate a lookup whose keys are unique ids. Decorate work whose
arguments come back.
