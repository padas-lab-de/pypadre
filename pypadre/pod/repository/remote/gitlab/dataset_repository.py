import os

from pypadre.core.model.dataset.attribute import Attribute
from pypadre.core.model.dataset.dataset import Dataset
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IDatasetRepository
from pypadre.pod.repository.local.file.dataset_repository import DatasetFileRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.remote.gitlab.generic.gitlab import GitLabRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer, PickleSerializer
from pypadre.pod.util.git_util import add_and_commit

NAME = "datasets"
META_FILE = File("metadata.json", JSonSerializer)
DATA_FILE = File("data.bin", PickleSerializer)


class DatasetGitlabRepository(GitLabRepository, IDatasetRepository):

    @staticmethod
    def placeholder():
        return '{DATASET_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(root_dir=os.path.join(backend.root_dir, NAME), gitlab_url=backend.url, token=backend.token
                         , backend=backend)
        self._file_backend = DatasetFileRepository(backend=backend)
        self._group = self.get_group(name=NAME)

    def _get_by_dir(self, directory):
        return self._file_backend._get_by_dir(directory)

    def _get_by_repo(self, repo, path=''):
        if repo is None:
            return None

        metadata = self.get_file(repo, META_FILE)

        attributes = [Attribute(**a) for a in metadata.get("attributes", {})]
        metadata["attributes"] = attributes

        ds = Dataset(metadata=metadata)

        if self.get_file(repo, DATA_FILE) is not None:
            def _load_data():
                return self.get_file(repo, DATA_FILE)

            ds.add_proxy_loader(_load_data)
        return ds

    def to_folder_name(self, dataset):
        """
        Converts the object to a name for the folder (For example the name of a dataset)
        :param obj: dataset passed
        :return:
        """
        return dataset.name

    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        dataset = obj
        self._file_backend._put(obj, *args, directory=directory, merge=merge, **kwargs)
        if self.has_remote_backend(dataset):
            add_and_commit(directory, message="Adding unstaged changes in the repo")
            self.push_changes()
