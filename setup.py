import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "cached_interpolate.build",
        ["cached_interpolate/build.pyx"],
        include_dirs=[np.get_include()],
    ),
]
setup(
    name="cached_interpolate",
    ext_modules=cythonize(extensions, language_level="3"),
    install_requires=["numpy"],
    packages=["cached_interpolate"],
)
