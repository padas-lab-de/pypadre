import os
import re


from pypadre.core.model.dataset.attribute import Attribute
from pypadre.core.model.dataset.dataset import Dataset
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IDatasetRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.remote.gitlab.repository.gitlab import GitLabRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer, PickleSerializer
from pypadre.pod.util.git_util import add_git_lfs_attribute_file, add_and_commit

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
        self._group = self.get_group(name=NAME)

    def get_by_repo(self, repo):
        if repo is None:
            return None

        metadata = self.get_file(repo, META_FILE)

        attributes = [Attribute(**a) for a in metadata.get("attributes", {})]
        metadata["attributes"] = attributes

        ds = Dataset(metadata=metadata)

        if self.get_file(repo, DATA_FILE) is not None:
            # TODO: Implement lazy loading of the dataset
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

    def get_by_name(self, name):
        """
        Shortcut because we know name is the folder name. We don't have to search in metadata.json
        :param name: Name of the dataset
        :return:
        """
        return self.list({'folder': re.escape(name)})

    def _put(self, obj, *args, directory: str,  merge=False, **kwargs):
        dataset = obj

        if self.remote is not None:
            add_and_commit(directory)
            self.push_changes()
        else:
            self.write_file(directory, META_FILE, dataset.metadata)
            self.write_file(directory, DATA_FILE, dataset.data(), 'wb')
            add_git_lfs_attribute_file(directory, "*.bin")
