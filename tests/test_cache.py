from __future__ import annotations

import pytest

from corto import Cache, memoize


def test_set_get_and_len() -> None:
    cache = Cache(8)
    cache["a"] = 1
    cache["b"] = 2
    assert cache["a"] == 1
    assert cache.get("b") == 2
    assert cache.get("missing") is None
    assert cache.get("missing", 7) == 7
    assert len(cache) == 2
    assert "a" in cache
    assert "missing" not in cache


def test_update_keeps_one_entry() -> None:
    cache = Cache(4)
    cache["a"] = 1
    cache["a"] = 2
    assert cache["a"] == 2
    assert len(cache) == 1


def test_none_value_is_stored() -> None:
    cache = Cache(2)
    cache["x"] = None
    assert "x" in cache
    assert cache["x"] is None


def test_delete() -> None:
    cache = Cache(4)
    cache["a"] = 1
    del cache["a"]
    assert "a" not in cache
    with pytest.raises(KeyError):
        del cache["a"]
    with pytest.raises(KeyError):
        _ = cache["a"]


def test_capacity_is_honored() -> None:
    cache = Cache(16)
    for i in range(64):
        cache[i] = i
    assert len(cache) <= 16


def test_maxsize_one() -> None:
    cache = Cache(1)
    cache[1] = "one"
    cache[2] = "two"
    assert len(cache) == 1
    assert 2 in cache
    assert 1 not in cache


def test_clear_resets_stats() -> None:
    cache = Cache(4)
    cache["a"] = 1
    assert cache["a"] == 1
    cache.clear()
    assert len(cache) == 0
    assert cache.hits == 0
    assert cache.misses == 0


def test_rejects_invalid_maxsize() -> None:
    with pytest.raises(ValueError, match="maxsize"):
        Cache(0)
    with pytest.raises(ValueError, match="maxsize"):
        Cache(-1)


def test_scan_does_not_evict_hot_keys() -> None:
    """Unique keys must not push a heated set out of a 100-slot cache."""
    cache = Cache(100)
    for i in range(100):
        cache[i] = i
    for _ in range(400):
        for i in range(10):
            assert cache[i] == i
    for i in range(1000, 2500):
        cache[i] = i
    assert len(cache) <= 100
    missing = [i for i in range(10) if i not in cache]
    assert missing == []


def test_frequency_grows_on_hits() -> None:
    cache = Cache(32)
    cache["hot"] = 1
    before = cache._frequency("hot")
    for _ in range(20):
        assert cache["hot"] == 1
    assert cache._frequency("hot") > before


def test_memoize_hits_and_clear() -> None:
    calls = 0

    @memoize(maxsize=32)
    def add(a: int, b: int = 0) -> int:
        nonlocal calls
        calls += 1
        return a + b

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert calls == 1
    info = add.cache_info()
    assert info.hits == 1
    assert info.misses == 1
    assert info.currsize == 1
    add.cache_clear()
    assert add(1, 2) == 3
    assert calls == 2


def test_pop_and_setdefault() -> None:
    cache = Cache(8)
    cache["a"] = 1
    assert cache.pop("a") == 1
    assert "a" not in cache
    assert cache.pop("a", 9) == 9
    with pytest.raises(KeyError):
        cache.pop("a")
    assert cache.setdefault("b", 2) == 2
    assert cache.setdefault("b", 3) == 2
    assert cache["b"] == 2


def test_memoize_keeps_function_metadata() -> None:
    @memoize(maxsize=8)
    def greet(name: str) -> str:
        """Say hello."""
        return f"hello {name}"

    assert greet.__name__ == "greet"
    assert greet.__doc__ == "Say hello."
    assert greet.__wrapped__.__name__ == "greet"
    assert greet.cache_parameters() == {"maxsize": 8, "typed": False}


def test_memoize_typed_separates_int_and_float() -> None:
    calls = 0

    @memoize(maxsize=16, typed=True)
    def ident(x: int | float) -> int | float:
        nonlocal calls
        calls += 1
        return x

    assert ident(1) == 1
    assert ident(1.0) == 1.0
    assert calls == 2
