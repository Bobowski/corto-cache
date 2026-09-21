#!/usr/bin/env -S uv run
"""Decorate / miss / hit timings for one-, two-, and three-arg wrappers."""

from __future__ import annotations

import gc
import time
from functools import lru_cache

from corto import Cache, memoize

N = 200_000
CAP = N


def _ns(label: str, n: int, fn, *, warmup: int = 0) -> float:
    if warmup:
        fn(warmup)
    gc.collect()
    gc.disable()
    t0 = time.perf_counter()
    fn(n)
    ns = (time.perf_counter() - t0) / n * 1e9
    gc.enable()
    print(f"  {label:<44} {ns:8.1f} ns")
    return ns


def main() -> None:
    print(f"Phase timings  n={N}  cap={CAP}  cheap fn: return args")
    print()
    print("Once (setup)")

    def body1(x: int) -> int:
        return x

    def body2(a: int, b: int) -> int:
        return a + b

    def body3(a: int, b: int, c: int) -> int:
        return a + b + c

    def deco1(n: int) -> None:
        for _ in range(n):
            memoize(maxsize=128)(body1)

    def deco2(n: int) -> None:
        for _ in range(n):
            memoize(maxsize=128)(body2)

    def deco3(n: int) -> None:
        for _ in range(n):
            memoize(maxsize=128)(body3)

    def deco_lru(n: int) -> None:
        for _ in range(n):
            lru_cache(maxsize=128)(body1)

    _ns("@memoize one-arg", 20_000, deco1)
    _ns("@memoize two-arg", 20_000, deco2)
    _ns("@memoize three-arg", 20_000, deco3)
    _ns("@lru_cache one-arg", 20_000, deco_lru)

    print()
    print("Each call")

    def bare(n: int) -> None:
        def f(x: int) -> int:
            return x

        for i in range(n):
            f(i)

    _ns("bare fn, no cache", N, bare, warmup=1_000)

    @memoize(maxsize=CAP)
    def corto1(x: int) -> int:
        return x

    @memoize(maxsize=CAP)
    def corto2(a: int, b: int) -> int:
        return a + b

    @memoize(maxsize=CAP)
    def corto3(a: int, b: int, c: int) -> int:
        return a + b + c

    @lru_cache(maxsize=CAP)
    def lru1(x: int) -> int:
        return x

    @lru_cache(maxsize=CAP)
    def lru2(a: int, b: int) -> int:
        return a + b

    @lru_cache(maxsize=CAP)
    def lru3(a: int, b: int, c: int) -> int:
        return a + b + c

    def miss1(n: int) -> None:
        for i in range(n):
            corto1(i)

    def hit1(n: int) -> None:
        for i in range(n):
            corto1(i)

    def miss2(n: int) -> None:
        for i in range(n):
            corto2(i, 0)

    def hit2(n: int) -> None:
        for i in range(n):
            corto2(i, 0)

    def miss3(n: int) -> None:
        for i in range(n):
            corto3(i, 0, 0)

    def hit3(n: int) -> None:
        for i in range(n):
            corto3(i, 0, 0)

    def lru_miss1(n: int) -> None:
        for i in range(n):
            lru1(i + N)

    def lru_hit1(n: int) -> None:
        for i in range(n):
            lru1(i + N)

    def lru_miss2(n: int) -> None:
        for i in range(n):
            lru2(i + N, 0)

    def lru_hit2(n: int) -> None:
        for i in range(n):
            lru2(i + N, 0)

    def lru_miss3(n: int) -> None:
        for i in range(n):
            lru3(i + N, 0, 0)

    def lru_hit3(n: int) -> None:
        for i in range(n):
            lru3(i + N, 0, 0)

    _ns("corto miss, one-arg", N, miss1)
    _ns("corto hit, one-arg", N, hit1)
    _ns("lru_cache miss, one-arg", N, lru_miss1)
    _ns("lru_cache hit, one-arg", N, lru_hit1)
    _ns("corto miss, two-arg", N, miss2)
    _ns("corto hit, two-arg", N, hit2)
    _ns("lru_cache miss, two-arg", N, lru_miss2)
    _ns("lru_cache hit, two-arg", N, lru_hit2)
    _ns("corto miss, three-arg", N, miss3)
    _ns("corto hit, three-arg", N, hit3)
    _ns("lru_cache miss, three-arg", N, lru_miss3)
    _ns("lru_cache hit, three-arg", N, lru_hit3)

    print()
    print("Cache mapping API")
    cache = Cache(CAP)
    missing = object()

    def api_set(n: int) -> None:
        for i in range(n):
            cache[i] = i

    def api_hit(n: int) -> None:
        for i in range(n):
            cache.get(i)

    def api_miss(n: int) -> None:
        for i in range(n):
            cache.get(i + N, missing)

    _ns("Cache set (insert)", N, api_set)
    _ns("Cache get hit", N, api_hit)
    _ns("Cache get miss (absent, no store)", N, api_miss)


if __name__ == "__main__":
    main()
