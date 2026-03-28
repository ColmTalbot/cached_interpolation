import pytest

from cached_interpolate import CachingInterpolant, RegularCachingInterpolant

interpolants = [RegularCachingInterpolant, CachingInterpolant]


@pytest.fixture(params=interpolants)
def interpolant(request):
    return request.param


@pytest.fixture(params=["nearest", "linear", "cubic"])
def kind(request):
    return request.param


@pytest.fixture(params=["numpy", "jax.numpy", "cupy", "torch"])
def backend(request):
    module = pytest.importorskip(request.param)
    if "jax" in request.param:
        import jax

        jax.config.update("jax_enable_x64", True)
    if request.param == "cupy":
        try:
            module.array([0.0])
        except Exception as e:
            pytest.skip(f"cupy is not functional: {e}")
    return module
