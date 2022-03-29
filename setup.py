import sys

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class LazyImportBuildExtCmd(build_ext):
    def finalize_options(self):
        from Cython.Build import cythonize

        compiler_directives = dict(
            language_level=3,
            boundscheck=False,
            wraparound=False,
            cdivision=True,
            initializedcheck=False,
            embedsignature=True,
        )
        if "develop" in sys.argv:
            compiler_directives["linetrace"] = True
            annotate = True
        else:
            annotate = False
        self.distribution.ext_modules = cythonize(
            self.distribution.ext_modules,
            compiler_directives=compiler_directives,
            annotate=annotate,
        )
        super(LazyImportBuildExtCmd, self).finalize_options()


if "develop" in sys.argv:
    macros = [
        ("CYTHON_TRACE", "1"),
        ("CYTHON_TRACE_NOGIL", "1"),
    ]
else:
    macros = list()
extensions = [
    Extension(
        "cached_interpolate.build",
        ["cached_interpolate/build.pyx"],
        include_dirs=[np.get_include()],
        define_macros=macros,
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
