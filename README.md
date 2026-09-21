# Corto Cache

```text
pip install corto-cache
```

```python
from corto import memoize
```

The index name is `corto-cache` (`corto` is taken). The import is `corto`.

Window-TinyLFU memoize for CPython 3.14.

Decorate a function. A hit returns the stored value. A miss runs the function
and stores the result if it returns. New keys enter a small LRU window. They
enter the main cache only when a frequency sketch says they are hotter than
what would be evicted. A scan of unique keys dies in the window. The hot set
stays.

```python
from corto import memoize

@memoize(maxsize=1024)
def parse_accept(header: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in header.split(","))

parse_accept.cache_info()    # hits, misses, maxsize, currsize
parse_accept.cache_clear()
```

`cache_info` and `cache_parameters` match `functools.lru_cache`. After a load
run, `hits / (hits + misses)` tells you if the same keys came back. Near-zero
hits: drop the decorator.

## When to use this

Use Corto when the key universe is larger than `maxsize` and traffic also
sends one-shot keys. `lru_cache` evicts whatever was least recent, so a scan
can flush a popular entry. Corto keeps the frequent ones.

Use `functools.lru_cache` when the working set fits and stays. It is the
stdlib, a bit cheaper on miss, and has `maxsize=None`.

Use neither when the function is cheaper than a lookup, or when every key is
new. A miss still runs the function; Corto then pays extra to decide admission.

## Cache mapping

When the value is not “return of this function” — insert and read in different
places — use the same store as a bounded map:

```python
from corto import Cache

cache = Cache(1024)
cache[header] = parsed
parsed = cache.get(header)
```

`pop`, `setdefault`, `clear`, `in`, `len`, `hits`, and `misses` are there.
There is no public frequency peek.

## Contract

- Keys must be hashable. Equality follows `==`.
- `maxsize` is a positive int. No unbounded mode, no TTL.
- Threads may share one wrapper. The lock is always on.
- `memoize` drops the lock while the function runs. Two threads may compute
  the same key; the first store wins.
- A key or value `__del__` must leave that cache unchanged. The mutex is not
  recursive.
- After `os.fork`, make a new cache in the child.
- Fixed 1% admission window. No hill-climb. No doorkeeper.

## Install

CPython 3.14. CI builds wheels for Linux x86_64, macOS (arm64, x86_64), and
Windows AMD64. A source build needs a C compiler and Cython.

```text
uv sync
uv run pytest
```

## Numbers

On this machine (CPython 3.14, Apple Silicon), a one-arg hit is next to
`lru_cache` (~70 ns). A Zipf decorator read is faster than Theine and
cachebox, and a bit faster than `lru_cache` on that mix. Hit *rate* under a
scan is the reason to pick Corto, not hit *time*.

Full tables and how to re-run: [benchmarks/README.md](benchmarks/README.md).

TinyLFU: https://arxiv.org/pdf/1512.00727  
Caffeine W-TinyLFU: https://github.com/ben-manes/caffeine/wiki/Efficiency
