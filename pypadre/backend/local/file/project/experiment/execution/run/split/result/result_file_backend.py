from pypadre.backend.interfaces.backend.i_result_backend import IResultBackend
from pypadre.backend.interfaces.backend.generic.i_base_file_backend import File
from pypadre.backend.serialiser import JSonSerializer


class PadreResultFileBackend(IResultBackend):

    RESULTS_FILE = File("results.json", JSonSerializer)

    def to_folder_name(self, obj):
        pass

    def get_by_dir(self, directory):
        pass

    def __init__(self, parent):
        super().__init__(parent, name=parent.RESULTS_FILE_NAME)

    def list(self, search, offset=0, size=100):
        pass

    def get(self, uid):
        pass

    def put(self, results):

        directory = self.get_dir(self.to_folder_name(self.parent))
        self.write_file(directory, self.RESULTS_FILE, results)

    def delete(self, uid):
        pass