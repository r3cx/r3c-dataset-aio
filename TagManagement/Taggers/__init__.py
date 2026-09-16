"""Taggers package.

``Tagger`` (base) and ``Eva02Tagger`` (ONNX regression baseline) are re-exported
eagerly. The PyTorch Danbooru-v4 taggers are re-exported lazily: importing them
pulls in ``torch`` / ``transformers``, so they are loaded only when requested
(``from Taggers import EvaGiantTagger``), keeping the eva02-only path light.

Taggers are never run as scripts (only autotagger leaves are entry points), so
there is no circular re-execution to avoid here — the laziness is purely to
defer the heavy deps.
"""

from .Tagger import Tagger
from .Eva02Tagger import Eva02Tagger

__all__ = ["Tagger", "Eva02Tagger", "Dbv4Tagger", "EvaGiantTagger", "ViTGiantTagger", "ConvNeXtV2Tagger"]

# PyTorch DBV4 taggers (deferred). Add each new v4 tagger here.
_LAZY = {"Dbv4Tagger", "EvaGiantTagger", "ViTGiantTagger", "ConvNeXtV2Tagger"}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module

        obj = getattr(import_module(f".{name}", __name__), name)
        globals()[name] = obj  # cache the class (import_module binds the submodule first)
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
