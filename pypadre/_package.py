from pypadre import _name, _version
from pypadre.core.model.code.code_mixin import PipIdentifier

PACKAGE_ID = PipIdentifier(pip_package=_name.__name__, version=_version.__version__)