from abc import abstractmethod, ABC

from pypadre.util.file_util import get_path


class IBackend(ABC):

    def __init__(self, config):
        # TODO dp: add check for root_dir
        self.root_dir = get_path(config.get('root_dir'), "")

    @property
    @abstractmethod
    def dataset(self):
        pass

    @property
    @abstractmethod
    def project(self):
        pass
