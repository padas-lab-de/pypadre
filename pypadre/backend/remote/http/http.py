from pypadre.backend.interfaces.backend.i_backend import IBackend
from pypadre.backend.remote.http.dataset.dataset_http_backend import PadreDatasetHTTPBackend
from pypadre.backend.remote.http.project.project_http_backend import PadreProjectHTTPBackend
from pypadre.util.file_util import get_path


class PadreHTTPBackend(IBackend):
    """
    Delegator class for handling padre objects at the file repository level. The following files tructure is used:

    root_dir
      |------datasets\
      |------experiments\
    """

    def __init__(self, config):
        # TODO dp: add check for root_dir
        self.root_dir = get_path(config.get('root_dir'), "")
        self._dataset = PadreDatasetHTTPBackend(self)
        self._experiment_repository = PadreProjectHTTPBackend(self, )

    @property
    def dataset(self):
        return self._dataset

    @property
    def project(self):
        return self._experiment_repository

    @property
    def experiment(self):
        return self._experiment_repository
