from pypadre.backend.interfaces.backend.i_backend import IBackend
from pypadre.backend.interfaces.backend.i_dataset_backend import IDatasetBackend
from pypadre.backend.interfaces.backend.i_experiment_backend import IExperimentBackend
from pypadre.backend.interfaces.backend.i_result_backend import IResultBackend
from pypadre.backend.interfaces.backend.i_run_backend import IRunBackend
from pypadre.backend.interfaces.backend.i_split_backend import ISplitBackend


class PadreDatasetFileBackend(IDatasetBackend):
    def put_progress(self, obj):
        pass

    def list(self, search, offset=0, size=100):
        pass

    def get(self, uid):
        pass

    def put(self, obj):
        pass

    def delete(self, uid):
        pass


class PadreExperimentFileBackend(IExperimentBackend):

    def put_config(self, obj):
        pass

    def put_progress(self, obj):
        pass

    def list(self, search, offset=0, size=100):
        pass

    def get(self, uid):
        pass

    def put(self, obj):
        pass

    def delete(self, uid):
        pass


class PadreResultFileBackend(IResultBackend):

    def list(self, search, offset=0, size=100):
        pass

    def get(self, uid):
        pass

    def put(self, obj):
        pass

    def delete(self, uid):
        pass


class PadreRunFileBackend(IRunBackend):

    def put_progress(self, obj):
        pass

    def list(self, search, offset=0, size=100):
        pass

    def get(self, uid):
        pass

    def put(self, obj):
        pass

    def delete(self, uid):
        pass


class PadreSplitFileBackend(ISplitBackend):

    def put_progress(self, obj):pass

    def list(self, search, offset=0, size=100):
        pass

    def get(self, uid):
        pass

    def put(self, obj):
        pass

    def delete(self, uid):
        pass


class PadreFileBackend(IBackend):
    """
    Delegator class for handling padre objects at the file repository level. The following files tructure is used:

    root_dir
      |------datasets\
      |------experiments\
    """

    @property
    def dataset(self):
        pass

    @property
    def experiment(self):
        pass

    @property
    def result(self):
        pass

    @property
    def run(self):
        pass

    @property
    def split(self):
        pass
