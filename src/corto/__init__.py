"""Corto Cache: Window-TinyLFU memoize and bounded mapping."""

from importlib.metadata import PackageNotFoundError, version

from corto._core import Cache, CacheInfo, memoize

try:
    __version__ = version("corto-cache")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["Cache", "CacheInfo", "__version__", "memoize"]
