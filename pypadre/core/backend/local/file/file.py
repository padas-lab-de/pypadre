from pypadre.core.backend.interfaces.backend.i_padre_backend import IPadreBackend
from pypadre.core.backend.local.file.dataset.dataset_file_backend import PadreDatasetFileBackend
from pypadre.core.backend.local.file.project.project_file_backend import PadreProjectFileBackend


class PadreFileBackend(IPadreBackend):
    """
    Delegator class for handling padre objects at the file repository level. The following files structure is used:

    root_dir
      |------datasets\
      |------experiments\
    """

    def __init__(self, config):
        # TODO dp: add check for root_dir
        super().__init__(config)
        self._dataset = PadreDatasetFileBackend(self)
        self._project = PadreProjectFileBackend(self)

    @property
    def dataset(self) -> PadreDatasetFileBackend:
        return self._dataset

    @property
    def project(self) -> PadreProjectFileBackend:
        return self._project

    def put(self, obj):
        # Changed to dataset TODO: Modify for project as well
        self._dataset.put(obj)

    def get(self, obj):
        return self._dataset.get(obj)
