"""Autotaggers package.

The base classes are re-exported eagerly. The concrete leaves are re-exported
lazily (PEP 562 ``__getattr__``): only a leaf is ever run as a plain script
(``python Autotaggers/<leaf>.py``), and eagerly importing a leaf from here would
re-execute that same file a second time under its dotted name when a script-mode
leaf triggers ``import Autotaggers``. Lazy leaves keep script-mode runs to a
single execution while still giving ``from Autotaggers import <Leaf>`` the class,
because ``__getattr__`` caches the resolved class into the package namespace
(``import_module`` otherwise binds the submodule under the same name).
"""

from .Autotagger import Autotagger
from .SlidingWindowAutotagger import SlidingWindowAutotagger

__all__ = ["Autotagger", "SlidingWindowAutotagger", "Eva02SlidingWindowAutotagger", "EnsembleSWAutoTagger"]

# Concrete sliding-window leaves. Add each new leaf here.
_LEAVES = {"Eva02SlidingWindowAutotagger", "EnsembleSWAutoTagger"}


def __getattr__(name):
    if name in _LEAVES:
        from importlib import import_module

        obj = getattr(import_module(f".{name}", __name__), name)
        globals()[name] = obj  # cache the class (import_module binds the submodule first)
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
