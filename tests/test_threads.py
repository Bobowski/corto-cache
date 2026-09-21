from __future__ import annotations

import threading

from corto import Cache, memoize


def test_threads_share_one_cache() -> None:
    cache = Cache(256)
    errors: list[BaseException] = []

    def worker(start: int) -> None:
        try:
            for i in range(start, start + 400):
                cache[i] = i
                got = cache.get(i)
                assert got is None or got == i
                _ = i in cache
                if i % 5 == 0:
                    cache.pop(i, None)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n * 1000,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(cache) <= 256
    cache._debug_counts()


def test_memoize_releases_lock_during_compute() -> None:
    started = threading.Event()
    release = threading.Event()
    calls = 0

    @memoize(maxsize=32)
    def slow(n: int) -> int:
        nonlocal calls
        calls += 1
        if n == 1:
            started.set()
            assert release.wait(timeout=2)
        return n + 1

    first = threading.Thread(target=lambda: slow(1))
    first.start()
    assert started.wait(timeout=2)
    assert slow(2) == 3
    release.set()
    first.join()
    assert slow(1) == 2
    assert calls == 2


def test_memoize_first_store_wins() -> None:
    barrier = threading.Barrier(2)
    seen: list[int] = []

    @memoize(maxsize=8)
    def once(n: int) -> int:
        barrier.wait()
        token = threading.get_ident()
        seen.append(token)
        return token

    results: list[int] = []

    def worker() -> None:
        results.append(once(0))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 2
    assert results[0] == results[1]
    assert once(0) == results[0]
