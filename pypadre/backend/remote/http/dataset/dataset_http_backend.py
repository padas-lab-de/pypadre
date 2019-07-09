import os

from pypadre.backend.interfaces.backend.i_dataset_backend import IDatasetBackend
from pypadre.util.file_util import dir_list


class PadreDatasetHTTPBackend(IDatasetBackend):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "datasets")

    def put_progress(self, obj):
        pass

    def list(self, search):
        """
        List all data sets in the repository
        :param **args:
        :param search_name: regular expression based search string for the title. Default None
        :param search_metadata: dict with regular expressions per metadata key. Default None
        """
        # todo apply the search metadata filter.
        dirs = dir_list(self.root_dir, search_name)
        return dirs  # [self.get(dir, metadata_only=True) for dir in dirs]

    def get(self, uid):
        pass

    def put(self, obj):
        pass

    def delete(self, uid):
        pass