from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy as np

extensions = [
    Extension("cached_spline._spline", ["cached_spline/_spline.pyx"], include_dirs=[np.get_include()]),
]
setup(
    name="cached_spline",
    ext_modules=cythonize(extensions, language_level="3"),
    install_requires=["numpy"],
    packages=["cached_spline"],
)
