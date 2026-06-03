"""Backwards-compatibility shims.

Historically this package was imported as ``AFToolKit`` (capital ``K``). That
only resolved on case-insensitive filesystems and has since been renamed to the
canonical lowercase ``aftoolkit``. Pickled artifacts created with the old name
(e.g. distributed pretrained adapter models and ``ProteinTask`` feature files)
embed the old module path ``AFToolKit.<...>`` and would otherwise fail to
unpickle.

Installing :func:`install_legacy_aliases` registers an import hook that
transparently redirects any ``AFToolKit`` / ``AFToolKit.*`` import to the
matching ``aftoolkit`` / ``aftoolkit.*`` module, so old pickles load unchanged.
"""

import importlib
import importlib.util
import sys
from importlib.abc import Loader, MetaPathFinder

_OLD_NAME = "AFToolKit"
_NEW_NAME = "aftoolkit"


class _LegacyNameFinder(MetaPathFinder, Loader):
    """Redirect ``AFToolKit[.*]`` imports to ``aftoolkit[.*]``."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == _OLD_NAME or fullname.startswith(_OLD_NAME + "."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        new_name = _NEW_NAME + spec.name[len(_OLD_NAME):]
        return importlib.import_module(new_name)

    def exec_module(self, module):
        # The module object is the already-imported canonical module; nothing
        # further to execute.
        return None


def install_legacy_aliases():
    """Idempotently install the legacy-name import hook.

    Inserted at the front of ``sys.meta_path`` so it pre-empts the default
    path-based finder. This matters on case-insensitive filesystems (macOS),
    where ``import AFToolKit.*`` would otherwise be resolved to the
    ``aftoolkit/`` directory as a *separate* module object, breaking pickle's
    class-identity check.
    """
    if any(isinstance(f, _LegacyNameFinder) for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _LegacyNameFinder())
