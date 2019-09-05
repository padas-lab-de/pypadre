from abc import ABC

from pypadre.pod.util.file_util import get_path


class IBackend(ABC):
    """ This is the simple entry implementation of a backend. We define a hierarchical structure
    to the other backends here. """

    def __init__(self, config):
        # TODO dp: add check for root_dir
        self.root_dir = get_path(config.get('root_dir'), "")
