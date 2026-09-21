from __future__ import annotations

from corto import Cache


def test_window_is_one_percent() -> None:
    assert Cache(1)._debug_counts()["window_max"] == 1
    assert Cache(2)._debug_counts()["window_max"] == 1
    assert Cache(99)._debug_counts()["window_max"] == 1
    assert Cache(100)._debug_counts()["window_max"] == 1
    assert Cache(200)._debug_counts()["window_max"] == 2
    assert Cache(1000)._debug_counts()["window_max"] == 10


def test_main_and_protected_split() -> None:
    counts = Cache(1000)._debug_counts()
    assert counts["window_max"] == 10
    assert counts["main_max"] == 990
    assert counts["protected_max"] == 792  # 80% of main


def test_sketch_sample_size_is_ten_times_capacity() -> None:
    counts = Cache(64)._debug_counts()
    assert counts["sketch_sample_size"] == 640


def test_new_key_frequency_starts_at_one() -> None:
    cache = Cache(32)
    cache["a"] = 1
    assert cache._frequency("a") == 1
    assert cache._frequency("missing") == 0


def test_hits_raise_frequency_and_cap_at_15() -> None:
    cache = Cache(32)
    cache["hot"] = 1
    for _ in range(40):
        assert cache["hot"] == 1
    assert cache._frequency("hot") == 15


def test_get_miss_is_recorded_in_the_sketch() -> None:
    cache = Cache(32)
    for _ in range(5):
        assert cache.get("ghost") is None
    assert cache._frequency("ghost") == 5


def test_equal_frequency_rejects_window_victim() -> None:
    """A cold scan must stay in the window once main is full."""
    cache = Cache(20)
    for i in range(20):
        cache[i] = i
    cache._debug_counts()
    first_wave = {i for i in range(20) if i in cache}
    for i in range(1000, 1040):
        cache[i] = i
    cache._debug_counts()
    still_present = [i for i in first_wave if i in cache]
    assert len(still_present) >= 10
    assert len(cache) <= 20


def test_hot_key_wins_admission_against_cold_victim() -> None:
    cache = Cache(20)
    for i in range(20):
        cache[i] = i
    for _ in range(30):
        assert cache[0] == 0
    cache["new"] = "new"
    for _ in range(30):
        assert cache["new"] == "new"
    for i in range(2000, 2060):
        cache[i] = i
    assert 0 in cache
    assert "new" in cache
    cache._debug_counts()


def test_aging_halves_counters() -> None:
    cache = Cache(8)
    cache["hot"] = 1
    for _ in range(20):
        assert cache["hot"] == 1
    assert cache._frequency("hot") == 15
    sample_size = cache._debug_counts()["sketch_sample_size"]
    already = cache._debug_counts()["sketch_samples"]
    for i in range(sample_size - already + 1):
        cache[f"scan-{i}"] = i
    assert cache._frequency("hot") == 7
    cache._debug_counts()


def test_hits_move_probation_into_protected() -> None:
    cache = Cache(100)
    for i in range(100):
        cache[i] = i
    for i in range(20):
        for _ in range(5):
            assert cache[i] == i
    counts = cache._debug_counts()
    assert counts["protected"] > 0
    assert counts["protected"] <= counts["protected_max"]


def test_window_overflow_keeps_capacity() -> None:
    cache = Cache(50)
    for i in range(500):
        cache[i] = i
        counts = cache._debug_counts()
        assert counts["size"] <= 50
        assert counts["window"] <= counts["window_max"]
        assert counts["protected"] <= counts["protected_max"]
