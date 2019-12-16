import importlib
import pkgutil
import sys


def optional_import(*package_names):
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.import_module(name)
        except ImportError:
            pass
    return len(packages) == len(package_names), packages



def module_exists(modname):
    """Checks if a module exists without actually importing it."""
    try:
        return pkgutil.find_loader(modname) is not None
    except ImportError:
        # TODO: Temporary fix for tf 1.14.0.
        # Should be removed once fixed in tf.
        return True


def modules_exist(*module_names):
    for name in module_names:
        if pkgutil.find_loader(name) is None:
            return False
    return True


def module_loaded(module):
    """Checks if a module was imported before (is in the import cache)."""
    return module in sys.modules


#owlready2_loaded, owlready2 = optional_import("owlready2")
pytorch_exists = modules_exist("pytorch")
