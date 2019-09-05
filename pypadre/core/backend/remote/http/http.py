from pypadre.core.backend.local.file.file import PadreFileBackend
from pypadre.core.backend.remote.http.dataset.dataset_http_backend import PadreDatasetHttpBackend
from pypadre.core.backend.remote.http.project.project_http_backend import PadreProjectHttpBackend


class PadreHttpBackend(PadreFileBackend):
    """
    Delegator class for handling padre objects at the file repository level. The following files tructure is used:

    root_dir
      |------datasets\
      |------experiments\
    """

    def __init__(self, config):
        # TODO defensive programing: add check for root_dir
        super().__init__(config)
        self._dataset = PadreDatasetHttpBackend(self)
        self._experiment_repository = PadreProjectHttpBackend(self)
