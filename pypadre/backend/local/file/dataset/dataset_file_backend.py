import os
import platform
import shutil

from git import Repo

from pypadre.backend.interfaces.backend.generic.i_base_file_backend import File
from pypadre.backend.interfaces.backend.i_dataset_backend import IDatasetBackend

from pypadre.backend.serialiser import JSonSerializer, PickleSerializer
from pypadre.core.model.dataset.attribute import Attribute
from pypadre.core.model.dataset.dataset import Dataset


class PadreDatasetFileBackend(IDatasetBackend):

    def __init__(self, parent):
        super().__init__(parent=parent, name="datasets")

    META_FILE = File("metadata.json", JSonSerializer)
    DATA_FILE = File("data.bin", PickleSerializer)
    GIT_ATTRIBUTES = '.gitattributes.'
    NAME = "name"
    MEASUREMENT_LEVEL = "measurementLevel"
    UNIT = "unit"
    DESCRIPTION = "description"
    DEFAULT_TARGET_ATTRIBUTE = "defaultTargetAttribute"
    INDEX = "index"
    CONTEXT = "context"

    DEFAULT_NAME = "Default Name"
    DEFAULT_MEASUREMENT_LEVEL = "Default Measurement Level"
    DEFAULT_UNIT = "unit"
    DEFAULT_DESCRIPTION = "Default Description"
    DEFAULT_TARGET_ATTRIBUTE_VALUE = 0
    DEFAULT_INDEX = -1
    DEFAULT_CONTEXT = "Default Context"


    def put(self, dataset: Dataset, allow_overwrite=True):
        directory = self.get_dir(self.to_folder_name(dataset))

        if os.path.exists(directory) and not allow_overwrite:
            raise ValueError("Dataset %s already exists." +
                             "Overwriting not explicitly allowed. Set allow_overwrite=True".format(dataset.name))
        else:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.mkdir(directory)

        self.write_file(directory, self.META_FILE, dataset.metadata)
        # TODO Write the actual dataset
        self.write_file_binary(directory, self.DATA_FILE, dataset.data())
        # TODO call git / git-lfs private functions here?
        # Get os version and write content to file
        path = None
        if platform.system() == 'Windows':
            path = os.path.join(directory, self.GIT_ATTRIBUTES)
        else:
            path = os.path.join(directory, self.GIT_ATTRIBUTES)

        f = open(path, "w")
        f.write("*.bin filter=lfs diff=lfs merge=lfs -text")
        repo = self._create_repo(path=directory, bare=False)
        self._add_files(repo, file_path=path)
        self._commit(repo=repo, message='Added .gitattributes file for Git LFS')

        # Add all untracked files
        self._add_untracked_files(repo=repo)
        self._commit(repo, message=self._DEFAULT_GIT_MSG)

    def get_by_dir(self, directory):
        if len(directory) == 0:
            return None

        metadata = self.get_file(os.path.join(self.root_dir, directory), self.META_FILE)

        attributes = metadata.pop("attributes", None)
        # print(type(metadata))
        # sorted(attributes, key=lambda a: a["index"])
        # assert sum([int(a["index"]) for a in attributes]) == len(attributes) * (
        #    len(attributes) - 1) / 2  # todo check attribute correctness here
        # TODO can this mapping be done in a better way???
        # TODO All of this should be done in the the Dataset constructor
        # TODO Handle scenario where attributes are not present in the metadata

        attributes = self.verify_attributes(metadata=metadata, attributes=attributes, fill_missing_attributes=True)


        attributes = [Attribute(a[self.NAME], a[self.MEASUREMENT_LEVEL], a[self.UNIT],
                                a[self.DESCRIPTION], a[self.DEFAULT_TARGET_ATTRIBUTE],
                                a[self.CONTEXT], a[self.INDEX])
                      for a in attributes] if attributes is not None else None

        ds = Dataset(id, attributes, **metadata)
        if self.has_file(os.path.join(self.root_dir, directory), self.DATA_FILE):
            ds.set_data(self.get_file(os.path.join(self.root_dir, directory), self.DATA_FILE))
        return ds

    def to_folder_name(self, obj):
        """
        Converts the object to a name for the folder (For example the name of a dataset)
        :param obj: dataset passed
        :return:
        """
        return obj.name

    def get_by_name(self, name):
        """
        Shortcut because we know name is the folder name. We don't have to search in metadata.json
        :param name: Name of the dataset
        :return:
        """
        return self.get_by_dir(self.get_dir(name))

    def verify_attributes(self, metadata, attributes, fill_missing_attributes=True):

        if fill_missing_attributes is True and attributes is not None:
            for attribute in attributes:
                if attribute.get(self.NAME, None) is None:
                    attribute[self.NAME] = metadata.get(self.NAME, self.DEFAULT_NAME)

                if attribute.get(self.MEASUREMENT_LEVEL, None) is None:
                    attribute[self.MEASUREMENT_LEVEL] = self.DEFAULT_MEASUREMENT_LEVEL

                if attribute.get(self.INDEX, None) is None:
                    attribute[self.INDEX] = self.DEFAULT_INDEX

                if attribute.get(self.UNIT, None) is None:
                    attribute[self.UNIT] = self.DEFAULT_UNIT

                if attribute.get(self.DEFAULT_TARGET_ATTRIBUTE, None) is None:
                    attribute[self.DEFAULT_TARGET_ATTRIBUTE] = self.DEFAULT_TARGET_ATTRIBUTE_VALUE

                if attribute.get(self.CONTEXT, None) is None:
                    attribute[self.CONTEXT] = self.DEFAULT_CONTEXT

                if attribute.get(self.DESCRIPTION, None) is None:
                    attribute[self.DESCRIPTION] = metadata.get(self.DESCRIPTION, self.DEFAULT_DESCRIPTION)

        return attributes

