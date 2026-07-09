"""FWI inversion baselines and diffusion-prior methods, unified API."""
from src.methods.inversion_methods import (
    DiffusionPrior,
    InversionResult,
    run_tikhonov,
    run_tv,
    run_red_diffeq,
    run_diffusion_fwi,
    run_diffusion_ilvr,
    run_dlo_fwi,
    run_dlo_fwi_adaptive,
    run_method_b,           # alias of run_dlo_fwi (legacy)
    apply_missing_traces,
    METHOD_REGISTRY,
)

__all__ = [
    "DiffusionPrior",
    "InversionResult",
    "run_tikhonov",
    "run_tv",
    "run_red_diffeq",
    "run_diffusion_fwi",
    "run_diffusion_ilvr",
    "run_dlo_fwi",
    "run_dlo_fwi_adaptive",
    "run_method_b",
    "apply_missing_traces",
    "METHOD_REGISTRY",
]
