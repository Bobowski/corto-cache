import sys
from pathlib import Path

from setuptools import Extension, setup

if sys.platform == "win32":
    extra_compile_args = ["/O2"]
else:
    extra_compile_args = ["-O3"]

pyx = Path("src/corto/_core.pyx")
if pyx.exists():
    from Cython.Build import cythonize

    ext_modules = cythonize(
        [
            Extension(
                "corto._core",
                sources=[str(pyx)],
                extra_compile_args=extra_compile_args,
            )
        ],
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
    )
else:
    ext_modules = [
        Extension(
            "corto._core",
            sources=["src/corto/_core.c"],
            extra_compile_args=extra_compile_args,
        )
    ]

setup(ext_modules=ext_modules)
