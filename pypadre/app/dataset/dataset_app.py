import inspect
from typing import List, Set, cast

from jsonschema import ValidationError

from pypadre.app.base_app import BaseChildApp
from pypadre.backend.interfaces.backend.i_dataset_backend import IDatasetBackend
from pypadre.core.model.dataset.dataset import DataSetValidator, Dataset
from pypadre.importing.dataset.dataset_import import PandasLoader, IDataSetLoader, CSVLoader, NumpyLoader, \
    NetworkXLoader, SklearnLoader, SnapLoader, KonectLoader, OpenMlLoader, ICollectionDataSetLoader


class DatasetApp(BaseChildApp):

    """
    Class providing commands for managing datasets.
    """
    def __init__(self, parent, backends: List[IDatasetBackend], **kwargs):
        super().__init__(parent=parent, backends=backends, **kwargs)
        self._loaders = [CSVLoader(), PandasLoader(), NumpyLoader(), NetworkXLoader(), SklearnLoader(), SnapLoader(),
                         KonectLoader(), OpenMlLoader()]

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

    def list(self, search, offset=0, size=100) -> List[Dataset]:
        """
        Lists all data sets matching search.
        :param offset:
        :param size:
        :param search: Search object
        :return: Data sets
        """
        data_sets = super().list(search)
        return data_sets

    def put(self, obj: Dataset):
        """
        Puts the data set if it's format is valid
        :param obj: Data set to put
        :return: Data set
        """
        try:
            DataSetValidator.validate(obj)
            for b in self.backends:
                b.dataset.put(obj)
            #super().put(obj)
            return obj

        except ValidationError as e:
            self.print_("Dataset could not be added. Please fix following problems and add manually: " + str(e))
            return obj

    def get(self, criteria:dict):
        datasets = []
        for b in self.backends:
            datasets.append(b.dataset.get(criteria))
        return datasets

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

    def list_on_loaders(self, search, **kwargs):
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


# class DatasetAppOld:
#     """
#     Class providing commands for managing datasets.
#     """
#
#     def __init__(self, parent):
#         self._parent = parent
#
#     @deprecated(reason="use downloads function below")  # see download
#     def do_default_imports(self, sklearn=True):
#         if sklearn:
#             for ds in ds_import.load_sklearn_toys():
#                 self.do_import(ds)
#
#     def _print(self, output, **kwargs):
#         self._parent.print(output, **kwargs)
#
#     def has_printer(self):
#         return self._parent.has_print()

    # @deprecated(reason="use the put method below. ")
    # def do_import(self, ds):
    #     if self.has_printer():
    #         self._print("Uploading dataset %s, %s, %s" % (ds.name, str(ds.size), ds.type))
    #     self._parent.remote_backend.upload_dataset(ds, True)

    # def upload_scratchdatasets(self, auth_token, max_threads=8, upload_graphs=True):
    #     if (max_threads < 1 or max_threads > 50):
    #         max_threads = 2
    #     if ("api" in _BASE_URL):
    #         url = _BASE_URL.strip("/api")
    #     else:
    #         url = _BASE_URL
    #     ds_import.sendTop100Datasets_multi(auth_token, url, max_threads)
    #     print("All openml datasets are uploaded!")
    #     if (upload_graphs):
    #         ds_import.send_top_graphs(auth_token, url, max_threads >= 3)

    # def get(self, dataset_id, binary: bool = True,
    #         format=formats.numpy,
    #         force_download: bool = True,
    #         cache_it: bool = False):
    #     """
    #     fetches a dataset either from local or from remote repository.
    #     :param dataset_id: id of the dataset to be fetched
    #     :param binary:
    #     :param format:
    #     :param force_download:
    #     :param cache_it:
    #     :return:
    #     """
    #     # todo check force_download=False and cache_it True
    #     ds = None
    #     if isinstance(dataset_id, Dataset):
    #         dataset_id = dataset_id.id
    #     if not force_download:  # look in cache first
    #         ds = self._parent.local_backend.datasets.get(dataset_id)
    #     if ds is None and not self._parent.offline:  # no cache or not looked --> go to http client
    #         # ds = self._parent.remote_backend.datasets.get(dataset_id, binary, format=format)
    #         ds = self._parent.remote_backend.datasets.get(dataset_id)
    #         if cache_it:
    #             self._parent.local_backend.datasets.put(ds)
    #     return ds

    # @deprecated  # use get
    # def get_dataset(self, dataset_id, binary=True, format=formats.numpy,
    #                 force_download=True, cache_it=False):
    #     return self.get(dataset_id, binary, format, force_download, cache_it)

    # def put(self, ds: Dataset, overwrite=True, upload=True) -> None:
    #     """
    #     puts a dataset to the local repository as well to server if upload is True
    #
    #     :param ds: dataset to be uploaded
    #     :type ds: <class 'pypadre.core.datasets.Dataset'>
    #     :param overwrite: if false, datasets are not overwritten
    #     :param upload: True, if the dataset should be uploaded
    #     """
    #     # todo implement overwrite correctly
    #     if upload:
    #         trigger_event('EVENT_WARN', condition=self._parent.offline is False, source=self,
    #                       message="Warning: The class is set to offline put upload was set to true. "
    #                               "Backend is not expected to work properly")
    #         if self.has_printer():
    #             self._print("Uploading dataset %s, %s, %s" % (ds.name, str(ds.size), ds.type))
    #         ds.id = self._parent.remote_backend.datasets.put(ds, True)
    #     self._parent.local_backend.datasets.put(ds)

    # def delete(self, dataset_id, remote_also=False):
    #     """
    #     delete the dataset with the provided id
    #     :param dataset_id: id of dataset as string or dataset object
    #     :return:
    #     """
    #     if isinstance(dataset_id, Dataset):
    #         dataset_id = dataset_id.id
    #     self._parent.local_backend.datasets.delete(dataset_id)

    # def import_from_csv(self, csv_path, targets, name, description):
    #     """Load dataset from csv file"""
    #     return ds_import.load_csv(csv_path, targets, name, description)
