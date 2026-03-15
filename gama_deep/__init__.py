"""
GAMA-Deep — Python package
ML analysis module for GAMA-Intel.
"""
from .bridge import (
    GAMADeepBridge,
    GAMAModule,
    ModuleLoader,
    run_deep_analysis,
)
__version__ = "0.1.0"
__all__ = ["GAMADeepBridge", "GAMAModule", "ModuleLoader", "run_deep_analysis"]
