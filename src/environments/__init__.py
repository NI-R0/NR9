"""Automatically imports all environment modules in this package."""

import importlib
import pkgutil

for _finder, _name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_name}")
