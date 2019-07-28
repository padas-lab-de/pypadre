from pypadre.backend.interfaces.backend.i_result_backend import IResultBackend
from pypadre.backend.local.file.project.experiment.execution.run.split.result.result_file_backend import \
    PadreResultFileBackend


class PadreResultHttpBackend(PadreResultFileBackend):

    def __init__(self, parent):
        super().__init__(parent)

    def list(self, search, offset=0, size=100):
        pass

    def get(self, uid):
        pass

    def put(self, obj):
        pass

    def delete(self, uid):
        pass