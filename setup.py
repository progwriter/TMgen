import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        f"tmgen.{name}",
        [f"src/tmgen/{name}.pyx"],
        include_dirs=[numpy.get_include()],
    )
    for name in ("hmc", "models", "tm")
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
)
