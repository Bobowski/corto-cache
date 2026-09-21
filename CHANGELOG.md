# Changelog

All notable changes to Corto Cache are documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Changed

- Recycle evicted nodes on a freelist instead of `malloc`/`free` each insert.
- `memoize` records a miss in the sketch once, when the value is stored.
- `memoize` reuses the key hash from the first lookup on store.
- Two- and three-argument functions (no defaults, no `*args`/`**kwargs`)
  use dedicated wrappers. Other signatures stay on the generic path.
- `Cache.frequency` is private (`_frequency`). Tests still use the hook.

## 0.1.0

### Added

- `memoize(maxsize=128, typed=False)` — Window-TinyLFU decorator with
  `cache_info`, `cache_clear`, `cache_parameters`, and `functools`
  wrapper metadata.
- `Cache(maxsize)` — bounded mapping: get, set, delete, `in`, `len`,
  `iter`, `pop`, `setdefault`, `clear`, plus `hits`, `misses`,
  and `maxsize`.
- One `PyMutex` on every public method. `memoize` drops the lock while
  the function runs.
- Wheel build via `cibuildwheel` for CPython 3.14.

### Notes

- No TTL, no unbounded `maxsize`, no hill-climb window resize, no
  doorkeeper.
- Package name on the index is `corto-cache`. The import is `corto`.
