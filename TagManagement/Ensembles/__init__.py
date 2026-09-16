"""Ensembles package.

Eager re-exports. Ensemble logic is model-free and never a script entry point,
so plain re-exports are safe.
"""

from .EnsembleTagger import EnsembleTagger

__all__ = ["EnsembleTagger"]
