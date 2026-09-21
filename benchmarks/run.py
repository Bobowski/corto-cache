#!/usr/bin/env -S uv run
"""Compare caches with Theine's published harness, not a homemade mix.

Hit ratio protocol (trace_bench.py):
  Zipf(s=1.001, v=10, imax=50_000_000), 1_000_000 GET requests, string keys.
  On miss: SET the same key. Hit rate = hits / requests.
  Same key list is replayed for every implementation.
  Theine is constructed as Cache(cap, nolock=True), as in their script.

Throughput protocol (throughput_bench.py):
  Decorator, DATA_LEN = 65536, cache size = 2 * DATA_LEN.
  Zipf(s=1.0001, v=9, imax=DATA_LEN) integer keys, prefill each key 3 times,
  then time DATA_LEN reads. Theine uses Memoize(..., ttl=None, nolock=True)
  and an explicit str key function.

Scan pollution is extra. It is not in Theine's repo.

Re-run: uv run --group bench python benchmarks/run.py
"""

from __future__ import annotations

import argparse
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from corto import Cache, memoize

# Theine trace_bench.zipf_key_gen
TRACE_S = 1.001
TRACE_V = 10.0
TRACE_IMAX = 50_000_000
TRACE_REQUESTS = 1_000_000
TRACE_CAPS = (500, 1_000, 2_000, 5_000, 10_000, 20_000, 40_000, 80_000)

# Theine throughput_bench
THRU_DATA_LEN = 2 << 15  # 65536
THRU_S = 1.0001
THRU_V = 9.0


class Zipf:
    """Bounded Zipf from Theine's bounded-zipf (Go math/rand.Zipf)."""

    def __init__(self, s: float, v: float, imax: int) -> None:
        if s <= 1 or v < 1:
            raise ValueError("Zipf requires s > 1 and v >= 1")
        self.imax = float(imax)
        self.v = v
        self.q = s
        self.oneminus_q = 1.0 - self.q
        self.oneminus_qinv = 1.0 / self.oneminus_q
        self.hxm = self._h(self.imax + 0.5)
        self.hx0minus_hxm = (
            self._h(0.5) - math.exp(math.log(self.v) * (-self.q)) - self.hxm
        )
        self.s = 1 - self._hinv(
            self._h(1.5) - math.exp(-self.q * math.log(self.v + 1.0))
        )

    def _h(self, x: float) -> float:
        return math.exp(self.oneminus_q * math.log(self.v + x)) * self.oneminus_qinv

    def _hinv(self, x: float) -> float:
        return math.exp(self.oneminus_qinv * math.log(self.oneminus_q * x)) - self.v

    def get(self) -> int:
        while True:
            ur = self.hxm + random.random() * self.hx0minus_hxm
            x = self._hinv(ur)
            k = math.floor(x + 0.5)
            if k - x <= self.s:
                return int(k)
            if ur >= self._h(k + 0.5) - math.exp(-math.log(k + self.v) * self.q):
                return int(k)


@dataclass(frozen=True)
class Store:
    name: str
    factory: Callable[[int], object]
    hit: Callable[[object, object], bool]
    put: Callable[[object, object, object], None]
    close: Callable[[object], None] = lambda _c: None


def _try_import(name: str) -> object | None:
    try:
        return __import__(name)
    except ImportError:
        return None


def _close_theine(cache: object) -> None:
    close = getattr(cache, "close", None)
    if close is not None:
        close()


def _put(cache: object, key: object, value: object) -> None:
    cache[key] = value


def _get_present(cache: object, key: object) -> bool:
    return cache.get(key) is not None


def _stores() -> list[Store]:
    missing = object()
    out = [
        Store(
            "corto.Cache",
            lambda cap: Cache(cap),
            lambda c, k: c.get(k, missing) is not missing,
            _put,
        ),
        Store(
            "dict (unbounded, upper bound)",
            lambda _cap: {},
            lambda c, k: k in c,
            _put,
        ),
    ]

    cachetools = _try_import("cachetools")
    if cachetools is not None:
        out.append(
            Store(
                "cachetools.LRUCache",
                lambda cap: cachetools.LRUCache(maxsize=cap),
                _get_present,
                _put,
            )
        )
        out.append(
            Store(
                "cachetools.FIFOCache",
                lambda cap: cachetools.FIFOCache(maxsize=cap),
                _get_present,
                _put,
            )
        )

    cachebox = _try_import("cachebox")
    if cachebox is not None:
        out.append(
            Store(
                "cachebox.LRUCache",
                lambda cap: cachebox.LRUCache(cap),
                _get_present,
                _put,
            )
        )

    theine = _try_import("theine")
    if theine is not None:
        out.append(
            Store(
                "theine.Cache (nolock)",
                lambda cap: theine.Cache(cap, True),
                lambda c, k: c.get(k)[1],
                lambda c, k, v: c.set(k, v),
                _close_theine,
            )
        )

    moka = _try_import("moka_py") or _try_import("moka")
    if moka is not None:
        moka_cls = getattr(moka, "Moka", None)
        if moka_cls is not None:
            out.append(
                Store(
                    "moka.Moka",
                    lambda cap: moka_cls(cap),
                    _get_present,
                    lambda c, k, v: c.set(k, v),
                )
            )
    return out


def _replay(store: Store, cap: int, keys: list[str]) -> tuple[float, float]:
    cache = store.factory(cap)
    misses = 0
    t0 = time.perf_counter()
    for key in keys:
        if store.hit(cache, key):
            continue
        misses += 1
        store.put(cache, key, key)
    elapsed = time.perf_counter() - t0
    store.close(cache)
    n = len(keys)
    return (n - misses) / n, elapsed / n * 1e9


def _print_table(title: str, rows: list[tuple[str, ...]]) -> None:
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    print()
    print(title)
    for i, row in enumerate(rows):
        print("  ".join(str(cell).ljust(widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            print("  ".join("-" * w for w in widths))


def _hit_ratio(stores: list[Store], caps: list[int], requests: int) -> None:
    z = Zipf(TRACE_S, TRACE_V, TRACE_IMAX)
    keys = [str(z.get()) for _ in range(requests)]
    print()
    print(
        f"Hit ratio  Zipf({TRACE_S}, {TRACE_V:g}, {TRACE_IMAX})  "
        f"n={requests}  string keys  get-then-set",
        flush=True,
    )
    header = ("impl", *[str(c) for c in caps])
    rows: list[tuple[str, ...]] = [header]
    ns_rows: list[tuple[str, ...]] = [header]
    for store in stores:
        rates: list[str] = []
        costs: list[str] = []
        for cap in caps:
            rate, ns = _replay(store, cap, keys)
            rates.append(f"{rate:.3f}")
            costs.append(f"{ns:.0f}")
        rows.append((store.name, *rates))
        ns_rows.append((store.name, *costs))
    _print_table("Hit rate by capacity (higher is better)", rows)
    _print_table("ns/request on the same replay (lower is faster)", ns_rows)


def _thru_trace() -> tuple[list[int], int, int, int]:
    z = Zipf(THRU_S, THRU_V, THRU_DATA_LEN)
    keys = [z.get() for _ in range(THRU_DATA_LEN)]
    cap = THRU_DATA_LEN * 2
    start = random.randint(0, THRU_DATA_LEN - 1)
    mask = THRU_DATA_LEN - 1
    return keys, cap, start, mask


def _thru_decorator() -> None:
    keys, cap, start, mask = _thru_trace()

    def time_fn(fn: Callable[[int], int]) -> float:
        for key in keys:
            fn(key)
            fn(key)
            fn(key)
        t0 = time.perf_counter()
        for i in range(THRU_DATA_LEN):
            fn(keys[(i + start) & mask])
        return (time.perf_counter() - t0) / THRU_DATA_LEN * 1e9

    rows: list[tuple[str, ...]] = [("impl", "read ns/op")]

    @lru_cache(maxsize=cap)
    def lru_get(key: int) -> int:
        return key

    rows.append(("functools.lru_cache", f"{time_fn(lru_get):.1f}"))

    @memoize(maxsize=cap)
    def tlfu_get(key: int) -> int:
        return key

    rows.append(("corto.memoize", f"{time_fn(tlfu_get):.1f}"))

    cachebox = _try_import("cachebox")
    if cachebox is not None:

        @cachebox.cached(cachebox.LRUCache(cap))
        def box_get(key: int) -> int:
            return key

        rows.append(("cachebox.cached(LRU)", f"{time_fn(box_get):.1f}"))

    theine = _try_import("theine")
    if theine is not None:

        @theine.Memoize(cap, None, nolock=True)
        def theine_get(key: int) -> int:
            return key

        @theine_get.key
        def _(key: int) -> str:
            return str(key)

        rows.append(("theine.Memoize (nolock, str key)", f"{time_fn(theine_get):.1f}"))

    cachetools = _try_import("cachetools")
    if cachetools is not None:
        from threading import Lock

        @cachetools.cached(
            cache=cachetools.TTLCache(maxsize=cap, ttl=20_000), lock=Lock()
        )
        def ct_get(key: int) -> int:
            return key

        rows.append(("cachetools.cached(TTL+Lock)", f"{time_fn(ct_get):.1f}"))

    _print_table(
        f"Decorator reads  DATA_LEN={THRU_DATA_LEN}  cap={cap}  "
        f"Zipf({THRU_S}, {THRU_V:g}, {THRU_DATA_LEN})  3x prefill",
        rows,
    )


def _thru_api(stores: list[Store]) -> None:
    keys, cap, start, mask = _thru_trace()
    rows: list[tuple[str, ...]] = [("impl", "read ns/op")]
    for store in stores:
        if store.name.startswith("dict"):
            continue
        cache = store.factory(cap)
        for key in keys:
            store.put(cache, key, key)
            store.hit(cache, key)
            store.hit(cache, key)
        t0 = time.perf_counter()
        for i in range(THRU_DATA_LEN):
            store.hit(cache, keys[(i + start) & mask])
        ns = (time.perf_counter() - t0) / THRU_DATA_LEN * 1e9
        store.close(cache)
        rows.append((store.name, f"{ns:.1f}"))
    _print_table(
        "Cache API reads  same Zipf and cap as the decorator bench",
        rows,
    )


def _scan(stores: list[Store]) -> None:
    """Not a Theine bench. Hot set of 50, then 40k unique keys."""
    hot = 50
    noise = 40_000
    cap = 10_000
    rows: list[tuple[str, ...]] = [("impl", "hot keys kept")]
    for store in stores:
        if store.name.startswith("dict"):
            continue
        cache = store.factory(cap)
        for i in range(hot):
            store.put(cache, i, i)
        for _ in range(200):
            for i in range(hot):
                store.hit(cache, i)
        for i in range(1_000, 1_000 + noise):
            store.put(cache, i, i)
        kept = sum(1 for i in range(hot) if store.hit(cache, i))
        store.close(cache)
        rows.append((store.name, f"{kept / hot:.3f}"))
    _print_table("Scan pollution (extra, not in Theine)", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Theine-aligned corto benches")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--requests", type=int, default=TRACE_REQUESTS)
    parser.add_argument(
        "--caps",
        default=",".join(str(c) for c in TRACE_CAPS),
        help="comma-separated capacities for the Zipf hit-ratio sweep",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="200k requests and caps 1k,5k,10k",
    )
    parser.add_argument(
        "--skip-hit",
        action="store_true",
        help="skip the Zipf hit-ratio sweep",
    )
    parser.add_argument(
        "--latency",
        action="store_true",
        help="decorator + Cache API reads only (optimization loop)",
    )
    args = parser.parse_args()
    random.seed(args.seed)

    requests = 200_000 if args.quick else args.requests
    caps = (
        [1_000, 5_000, 10_000]
        if args.quick
        else [int(part) for part in args.caps.split(",") if part]
    )
    stores = _stores()

    print("Aligned with https://github.com/Yiling-J/theine/tree/main/benchmarks", flush=True)
    print(
        "Theine single-thread path uses nolock=True (their trace_bench / nolock bench).",
        flush=True,
    )
    if args.latency:
        _thru_decorator()
        _thru_api(stores)
    else:
        if not args.skip_hit:
            _hit_ratio(stores, caps, requests)
        _thru_decorator()
        _thru_api(stores)
        _scan(stores)
    print()
    print("Missing impls: uv sync --group bench")


if __name__ == "__main__":
    main()
