import inspect
from typing import List, cast

from jsonschema import ValidationError

from pypadre.pod.app.base_app import BaseChildApp
from pypadre.core.model.dataset.dataset import Dataset
from pypadre.pod.importing.dataset.dataset_import import PandasLoader, IDataSetLoader, CSVLoader, NumpyLoader, \
    NetworkXLoader, SKLearnLoader, SnapLoader, KonectLoader, ICollectionDataSetLoader
from pypadre.pod.repository.i_repository import IDatasetRepository
from pypadre.pod.service.dataset_service import DatasetService


class DatasetApp(BaseChildApp):

    """
    Class providing commands for managing datasets.
    """
    def __init__(self, parent, backends: List[IDatasetRepository], dataset_loaders=None, **kwargs):
        super().__init__(parent=parent, service=DatasetService(backends=backends), **kwargs)

        if dataset_loaders is None:
            dataset_loaders = {}
        self._loaders = {CSVLoader(), PandasLoader(), NumpyLoader(), NetworkXLoader(), SKLearnLoader(), SnapLoader(),
                         KonectLoader(), OpenMlLoader()}.union(dataset_loaders)

    @property
    def loaders(self):
        """ Return all loaders """
        return self._loaders

    def loader(self, source):
        """
        Get a loader for a passed source. Each loader defines a mapper which can be used to check if source is valid for it.
        :param source: A string for collections or a object containing the data for single sources
        :return: The loader
        """
        loaders = [loader for loader in self._loaders if cast(IDataSetLoader, loader).mapping(source)]
        if len(loaders) == 1:
            return next(iter(loaders))
        elif len(loaders) > 1:
            raise ValueError("More than one loader matched. Remove a redundant loader." + self._loader_patterns())
        else:
            raise ValueError("You passed %s. " + self._loader_patterns() % str(source))

    def _loader_patterns(self):
        """
        Return string informing of all possible loader patters for the source. # TODO maybe rework this to something better
        :return: Information about which loaders are available behind which mappers
        """
        out = "Pass a source matching one of the following functions: "
        out += ",".join([str(inspect.getsource(cast(IDataSetLoader, loader).mapping)) for loader in self._loaders])
        return out

    def load_defaults(self):
        """
        Load all default data sets of the in the app defined loaders
        :return:
        """
        for l in self._loaders:
            if isinstance(l, ICollectionDataSetLoader):
                for data_set in l.load_default():
                    self.put(data_set)

    def list_from_loaders(self, search, **kwargs):
        # TODO list from external sources
        pass

    def sync(self, name: str = None, mode: str = "sync"):
        """
        syncs the specified dataset with all backends.
        :param name: name of the dataset. If None is provided, all datasets are synced
        :param mode: mode of synching. "push" uploads the local dataset to the server
        "pull" downloads the server dataset locally
        "sync" pushes dataset if it does not exist on server or pulls dataset if it does not exist locally. issues a
         warning if the remote hash of the dataset is not in sync with the local hash of the dataset.
        :return:
        """
        # TODO sync in backends if needed
        pass

    def put(self, obj: Dataset):
        """
        Puts the data set if it's format is valid
        :param obj: Data set to put
        :return: Data set
        """
        try:
            super().put(obj)
        except ValidationError as e:
            self.print_("Dataset could not be added. Please fix following problems and add manually: " + str(e))
            return obj

    def load(self, source, **kwargs) -> Dataset:
        """
        Load the dataset defined by source and parameters
        :param source: source object providing a path
        :param kwargs: parameters for the loader
        :return: dataset
        """
        loader = self.loader(source)
        data_set = cast(IDataSetLoader, loader).load(source=source, **kwargs)
        return self.put(data_set)
