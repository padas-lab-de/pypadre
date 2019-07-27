from pypadre.backend.interfaces.backend.i_backend import IBackend
from pypadre.backend.local.file.dataset.dataset_file_backend import PadreDatasetFileBackend
from pypadre.backend.local.file.project.project_file_backend import PadreProjectHTTPBackend


class PadreFileBackend(IBackend):
    """
    Delegator class for handling padre objects at the file repository level. The following files tructure is used:

    root_dir
      |------datasets\
      |------experiments\
    """

    def __init__(self, config):
        # TODO dp: add check for root_dir
        super().__init__(config)
        self._dataset = PadreDatasetFileBackend(self)
        self._project = PadreProjectHTTPBackend(self)

    @property
    def dataset(self):
        return self._dataset

    @property
    def project(self):
        return self._project
