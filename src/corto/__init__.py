"""Corto Cache: Window-TinyLFU memoize and bounded mapping."""

from importlib import metadata as _metadata

from corto._core import Cache, CacheInfo, memoize

try:
    __version__ = _metadata.version("corto-cache")
except _metadata.PackageNotFoundError:
    __version__ = "0.1.1"

__all__ = ["Cache", "CacheInfo", "__version__", "memoize"]
