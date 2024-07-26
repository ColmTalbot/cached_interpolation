import importlib

import pytest

from cached_interpolate import CachingInterpolant, RegularCachingInterpolant

interpolants = [RegularCachingInterpolant, CachingInterpolant]


@pytest.fixture(params=interpolants)
def interpolant(request):
    return request.param


@pytest.fixture(params=["nearest", "linear", "cubic"])
def kind(request):
    return request.param


@pytest.fixture(params=["numpy", "jax.numpy", "cupy"])
def backend(request):
    pytest.importorskip(request.param)
    if "jax" in request.param:
        import jax

        jax.config.update("jax_enable_x64", True)
    return importlib.import_module(request.param)
