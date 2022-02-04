import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class LazyImportBuildExtCmd(build_ext):
    def finalize_options(self):
        from Cython.Build import cythonize

        self.distribution.ext_modules = cythonize(self.distribution.ext_modules)
        super(LazyImportBuildExtCmd, self).finalize_options()


extensions = [
    Extension(
        "cached_interpolate.build",
        ["cached_interpolate/build.pyx"],
        include_dirs=[np.get_include()],
    ),
]
setup(
    name="cached_interpolate",
    ext_modules=extensions,
    install_requires=["numpy"],
    setup_requires=["cython", "numpy"],
    cmdclass={"build_ext": LazyImportBuildExtCmd},
    packages=["cached_interpolate"],
)
