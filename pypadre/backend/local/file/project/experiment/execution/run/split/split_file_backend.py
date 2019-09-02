import os
import shutil
import uuid

from pypadre.backend.interfaces.backend.generic.i_base_file_backend import File
from pypadre.backend.interfaces.backend.i_split_backend import ISplitBackend
from pypadre.backend.serialiser import JSonSerializer, PickleSerializer
from pypadre.core.model.split.split import Split


class PadreSplitFileBackend(ISplitBackend):

    @staticmethod
    def _placeholder():
        return '{SPLIT_ID}'

    @staticmethod
    def _get_parent_of(obj: Split):
        return obj.run

    RESULTS_FILE_NAME = "results.json"
    METRICS_FILE_NAME = "metrics.json"
    RESULTS_FILE = File(RESULTS_FILE_NAME, JSonSerializer)
    METRICS_FILE = File(METRICS_FILE_NAME, JSonSerializer)
    METADATA_FILE = File("metadata.json", JSonSerializer)
    NAME = "splits"

    def __init__(self, parent):

        super().__init__(parent, name=self.NAME)
        self.root_dir = os.path.join(self._parent.root_dir, self._parent._placeholder(), self.NAME)

    @property
    def result(self):
        return self._result

    @property
    def metric(self):
        return self._metric

    def to_folder_name(self, split):
        return split.id

    def get_by_dir(self, directory):
        pass

    def log(self, msg):
        pass

    def put_progress(self, split, **kwargs):
        self.log(
            "Split " + split + " PROGRESS: {curr_value}/{limit}. phase={phase} \n".format(**kwargs))

    def list(self, search, offset=0, size=100):
        pass

    def get(self, uid):
        pass

    def put(self, split, allow_overwrite=False):
        """
        Stores a split of an experiment to the file repository.
        :param allow_overwrite: allow overwrite of experiment
        :param split: split to put
        :return:
        """

        if split.id is None:  # this is a new experiment
            split.id = uuid.uuid4()

        directory = self.get_dir(self.to_folder_name(split))

        if os.path.exists(directory) and not allow_overwrite:
            raise ValueError("Split %s already exists." +
                             "Overwriting not explicitly allowed. Set allow_overwrite=True".format(split.id))
        else:
            shutil.rmtree(directory)
        os.mkdir(directory)

        self.write_file(directory, self.METADATA_FILE, split.metadata)
        # TODO updating metrics and results could be done here or in an own function

    def delete(self, uid):
        pass