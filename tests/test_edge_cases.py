from __future__ import annotations

import gc
import random
import sys
import weakref

import pytest

from corto import Cache, memoize


class _Collide:
    def __init__(self, n: int) -> None:
        self.n = n

    def __hash__(self) -> int:
        return 42

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Collide) and self.n == other.n


class _RaisesEq:
    def __hash__(self) -> int:
        return 7

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("eq failed")


class _AlwaysEqual:
    def __init__(self, n: int) -> None:
        self.n = n

    def __hash__(self) -> int:
        return self.n

    def __eq__(self, other: object) -> bool:
        return True


def test_none_key_roundtrip() -> None:
    cache = Cache(4)
    cache[None] = "none"
    assert cache[None] == "none"
    assert None in cache
    cache._debug_counts()


def test_unhashable_key_raises() -> None:
    cache = Cache(4)
    with pytest.raises(TypeError):
        cache[[1, 2]] = 1
    with pytest.raises(TypeError):
        _ = cache.get([1, 2])
    cache._debug_counts()


def test_rejects_huge_maxsize() -> None:
    with pytest.raises(ValueError, match="too large"):
        Cache((1 << 30) + 1)


def test_equal_keys_share_one_entry() -> None:
    cache = Cache(8)
    cache[_AlwaysEqual(1)] = "a"
    cache[_AlwaysEqual(1)] = "b"
    assert len(cache) == 1
    assert cache[_AlwaysEqual(1)] == "b"
    cache._debug_counts()


def test_hash_collisions_stay_distinct() -> None:
    cache = Cache(32)
    keys = [_Collide(i) for i in range(20)]
    for key in keys:
        cache[key] = key.n
    for key in keys:
        assert cache[key] == key.n
    assert len(cache) == 20
    del cache[keys[3]]
    assert keys[3] not in cache
    assert cache[keys[4]] == 4
    cache._debug_counts()


def test_eq_error_does_not_look_like_a_hit() -> None:
    cache = Cache(8)
    cache[_RaisesEq()] = 1
    with pytest.raises(RuntimeError, match="eq failed"):
        _ = cache[_RaisesEq()]
    with pytest.raises(RuntimeError, match="eq failed"):
        _ = cache.get(_RaisesEq())
    cache._debug_counts()


def test_many_deletes_then_insert_keeps_invariants() -> None:
    cache = Cache(64)
    for i in range(64):
        cache[i] = i
    for i in range(0, 64, 2):
        del cache[i]
    cache._debug_counts()
    for i in range(200, 280):
        cache[i] = i
    cache._debug_counts()
    assert len(cache) <= 64


def test_tombstone_compaction_survives_churn() -> None:
    cache = Cache(16)
    for i in range(400):
        cache[i] = i
        if i % 3 == 0 and (i - 1) in cache:
            del cache[i - 1]
        cache._debug_counts()
    assert len(cache) <= 16


def test_iteration_matches_len() -> None:
    cache = Cache(32)
    for i in range(20):
        cache[i] = i
    keys = list(cache)
    assert len(keys) == len(cache)
    assert set(keys) <= set(range(20))
    cache._debug_counts()


def test_clear_empties_and_accepts_new_keys() -> None:
    cache = Cache(8)
    cache["a"] = 1
    cache.clear()
    assert list(cache) == []
    cache["b"] = 2
    assert cache["b"] == 2
    cache._debug_counts()


def test_update_does_not_grow() -> None:
    cache = Cache(4)
    cache["a"] = 1
    for i in range(50):
        cache["a"] = i
    assert len(cache) == 1
    assert cache["a"] == 49
    cache._debug_counts()


def test_delete_missing_and_get_missing_stats() -> None:
    cache = Cache(4)
    with pytest.raises(KeyError):
        del cache["missing"]
    assert cache.misses == 0
    assert cache.get("missing") is None
    assert cache.misses == 1
    with pytest.raises(KeyError):
        _ = cache["missing"]
    assert cache.misses == 2


def test_cycle_with_cache_is_collected() -> None:
    class Box:
        def __init__(self) -> None:
            self.cache: Cache | None = None

    cache = Cache(8)
    box = Box()
    box.cache = cache
    cache["box"] = box
    wr = weakref.ref(box)
    del box
    del cache
    gc.collect()
    assert wr() is None


def test_refcount_is_released_on_delete_and_clear() -> None:
    cache = Cache(4)
    value = object()
    start = sys.getrefcount(value)
    cache["a"] = value
    assert sys.getrefcount(value) > start
    del cache["a"]
    assert sys.getrefcount(value) == start
    cache["a"] = value
    cache.clear()
    assert sys.getrefcount(value) == start


def test_del_must_not_break_cache_when_it_only_reads() -> None:
    seen: list[int] = []

    class Probe:
        def __del__(self) -> None:
            seen.append(1)

    cache = Cache(4)
    cache["p"] = Probe()
    del cache["p"]
    gc.collect()
    assert seen == [1]
    cache["q"] = 1
    cache._debug_counts()


def test_memoize_does_not_store_exceptions() -> None:
    calls = 0

    @memoize(maxsize=8)
    def boom(n: int) -> int:
        nonlocal calls
        calls += 1
        if n < 0:
            raise ValueError("neg")
        return n

    with pytest.raises(ValueError, match="neg"):
        boom(-1)
    with pytest.raises(ValueError, match="neg"):
        boom(-1)
    assert calls == 2
    assert boom(3) == 3
    assert boom(3) == 3
    assert calls == 3


def test_memoize_kwargs_order_and_typed() -> None:
    calls = 0

    @memoize(maxsize=16)
    def add(a: int, b: int = 0) -> int:
        nonlocal calls
        calls += 1
        return a + b

    assert add(1, b=2) == 3
    assert add(1, b=2) == 3
    assert add(a=1, b=2) == 3
    assert calls == 2


def test_memoize_one_arg_uses_fast_path() -> None:
    calls = 0

    @memoize(maxsize=16)
    def ident(x: int) -> int:
        nonlocal calls
        calls += 1
        return x

    assert ident(1) == 1
    assert ident(1) == 1
    assert calls == 1
    assert ident.cache_parameters() == {"maxsize": 16, "typed": False}


def test_memoize_two_arg_uses_fast_path() -> None:
    calls = 0

    @memoize(maxsize=16)
    def add(a: int, b: int) -> int:
        nonlocal calls
        calls += 1
        return a + b

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert add(1, 3) == 4
    assert calls == 2
    assert add.cache_info().currsize == 2


def test_memoize_three_arg_uses_fast_path() -> None:
    calls = 0

    @memoize(maxsize=16)
    def add3(a: int, b: int, c: int) -> int:
        nonlocal calls
        calls += 1
        return a + b + c

    assert add3(1, 2, 3) == 6
    assert add3(1, 2, 3) == 6
    assert add3(1, 2, 4) == 7
    assert calls == 2


def test_memoize_defaults_stay_on_generic_path() -> None:
    calls = 0

    @memoize(maxsize=16)
    def add(a: int, b: int = 1) -> int:
        nonlocal calls
        calls += 1
        return a + b

    assert add(1) == 2
    assert add(1) == 2
    assert add(1, 2) == 3
    assert calls == 2


def test_memoize_unhashable_args_raise() -> None:
    @memoize(maxsize=8)
    def join(items: list[int]) -> int:
        return sum(items)

    with pytest.raises(TypeError):
        join([1, 2])


def test_random_ops_keep_invariants() -> None:
    rng = random.Random(20260917)
    cache = Cache(48)
    last: dict[int, int] = {}
    for _ in range(4000):
        op = rng.randrange(7)
        key = rng.randrange(80)
        if op == 0:
            value = rng.randrange(1000)
            cache[key] = value
            last[key] = value
        elif op == 1:
            got = cache.get(key)
            if got is not None:
                assert got == last[key]
        elif op == 2:
            if key in cache:
                del cache[key]
        elif op == 3:
            _ = key in cache
        elif op == 4:
            cache._frequency(key)
        elif op == 5:
            list(cache)
        else:
            cache.clear()
            last.clear()
        counts = cache._debug_counts()
        assert counts["size"] == len(cache)
        assert counts["size"] <= 48
        for stored in list(cache):
            assert cache[stored] == last[stored]
