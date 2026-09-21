# Changelog

All notable changes to Corto Cache are documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.1.0

First public release.

### Added

- `memoize(maxsize=128, typed=False)` — Window-TinyLFU decorator with
  `cache_info`, `cache_clear`, `cache_parameters`, and `functools`
  wrapper metadata.
- Dedicated wrappers for one-, two-, and three-argument functions (no
  defaults, no `*args` / `**kwargs`). Other signatures use the generic path.
- `Cache(maxsize)` — bounded mapping: get, set, delete, `in`, `len`,
  `iter`, `pop`, `setdefault`, `clear`, plus `hits`, `misses`, and
  `maxsize`.
- One `PyMutex` on every public method. `memoize` drops the lock while
  the function runs.
- Node freelist: evicted entries are reused instead of `malloc` / `free`
  on each insert.
- Wheels via `cibuildwheel` for CPython 3.14 on Linux x86_64, macOS
  (arm64, x86_64), and Windows AMD64.

### Notes

- Package name on the index is `corto-cache`. The import is `corto`.
- No TTL, no unbounded `maxsize`, no hill-climb window resize, no
  doorkeeper.
- Sketch frequency is private (`Cache._frequency`).
- `memoize` records a miss in the sketch once, when the value is stored,
  and reuses the key hash from the first lookup.
