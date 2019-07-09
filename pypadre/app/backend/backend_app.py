from jsonschema import ValidationError

from pypadre.backend.local.file import PadreFileBackend
from pypadre.backend.http import PadreHTTPClient


class BackEndApp:

    def __init__(self, config):
        self._backends = self._parse(config)

    @staticmethod
    def _parse(config):
        _backends = config.get("backends", "GENERAL")
        backends = []
        for b in _backends:
            if hasattr(b, 'base_url'):
                # TODO check for validity
                backends.append(PadreHTTPClient(b))
            elif hasattr(b, 'root_dir'):
                # TODO check for validity
                backends.append(PadreFileBackend(b))
            else:
                raise ValidationError('{0} defined an invalid backend. Please provide either a http backend'
                                      ' or a local backend. (root_dir or base_url)'.format(b))
        return backends
