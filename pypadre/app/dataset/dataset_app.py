from pypadre.eventhandler import trigger_event

from pypadre.printing.util.print_util import print_table, to_table


class DatasetApp:
    """
    Class providing commands for managing datasets.
    """
    def __init__(self, parent):
        self._parent = parent

    def list(self, search_name=None, search_metadata=None, start=0, count=999999999, prnt=False):
        """
        lists all datasets matching the provided criterions in the configured backends (local and remote)
        :param search_name: name of the dataset as regular expression
        :param search_metadata: key/value dictionaries for metadata field / value (not implemented yet)
        :param start: paging information where to start in the returned list
        :param count: number of datasets to return.
        :param print: whether to print the details of the datasets or not.
        :return:
        """
        # todo: merge results and allow multiple repositories. all should have same signature. then iterate over repos
        local_datasets = self._parent.local_backend.datasets.list()
        if local_datasets is None:
            local_datasets = []
        remote_datasets = []
        if not self._parent.offline:
            remote_datasets = self._parent.remote_backend.datasets.list()
        local_datasets.extend(remote_datasets)

        if prnt:
            self.print_details(local_datasets)

        # Add sklearn toy datasets if they are not present in the list
        if len(local_datasets) == 0:
            local_datasets = ['Boston_House_Prices',
                              'Breast_Cancer',
                              'Diabetes',
                              'Digits',
                              'Iris',
                              'Linnerrud']

        return local_datasets

    def print_datasets(self, datasets: list):
        if self._parent.has_print():
            self._print("Loading.....")
            self._print(to_table(self._parent, datasets))

    def print(self, ds):
        if self.has_printer():
            self._print(ds)

    def search_downloads(self, name: str = None)->list:
        """
        searches for importable datasets (as specified in the datasets file).
        :param name: regexp for filtering the names
        :return: list of possible imports
        """
        oml_key = self._parent.config.get("oml_key", "GENERAL")
        root_dir = self._parent.config.get("root_dir", "LOCAL BACKEND")
        datasets = ds_import.search_oml_datasets(name, root_dir, oml_key)

        datasets_list = []
        for key in datasets.keys():
            datasets_list.append(datasets[key])
        return datasets_list

    def download(self, sources: list) -> Iterable:
        """
        Downloads the datasets from information provided as list from oml
        :return: returns a iterator of dataset objects
        """
        # todo: Extend support for more dataset sources other than openML
        for dataset_source in sources:
            dataset = self._parent.remote_backend.datasets.load_oml_dataset(str(dataset_source["did"]))
            yield dataset

    def sync(self, name: str = None, mode: str = "sync"):
        """
        syncs the specified dataset with the server.
        :param name: name of the dataset. If None is provided, all datasets are synced
        :param mode: mode of synching. "push" uploads the local dataset to the server
        "pull" downloads the server dataset locally
        "sync" pushes dataset if it does not exist on server or pulls dataset if it does not exist locally. issues a
         warning if the remote hash of the dataset is not in sync with the local hash of the dataset.
        :return:
        """
        pass

    @deprecated(reason="use downloads function below")  # see download
    def do_default_imports(self, sklearn=True):
        if sklearn:
            for ds in ds_import.load_sklearn_toys():
                self.do_import(ds)

    def _print(self, output, **kwargs):
        self._parent.print(output, **kwargs)

    def has_printer(self):
        return self._parent.has_print()

    @deprecated(reason="use the put method below. ")
    def do_import(self, ds):
        if self.has_printer():
            self._print("Uploading dataset %s, %s, %s" % (ds.name, str(ds.size), ds.type))
        self._parent.remote_backend.upload_dataset(ds, True)

    def upload_scratchdatasets(self, auth_token, max_threads=8, upload_graphs=True):
        if(max_threads < 1 or max_threads > 50):
            max_threads = 2
        if("api"in _BASE_URL):
            url=_BASE_URL.strip("/api")
        else:
            url =_BASE_URL
        ds_import.sendTop100Datasets_multi(auth_token, url, max_threads)
        print("All openml datasets are uploaded!")
        if(upload_graphs):
            ds_import.send_top_graphs(auth_token, url, max_threads >= 3)

    def get(self, dataset_id, binary: bool = True,
            format = formats.numpy,
            force_download: bool = True,
            cache_it: bool = False):
        """
        fetches a dataset either from local or from remote repository.
        :param dataset_id: id of the dataset to be fetched
        :param binary:
        :param format:
        :param force_download:
        :param cache_it:
        :return:
        """
        # todo check force_download=False and cache_it True
        ds = None
        if isinstance(dataset_id, Dataset):
            dataset_id = dataset_id.id
        if not force_download:  # look in cache first
            ds = self._parent.local_backend.datasets.get(dataset_id)
        if ds is None and not self._parent.offline:  # no cache or not looked --> go to http client
            # ds = self._parent.remote_backend.datasets.get(dataset_id, binary, format=format)
            ds = self._parent.remote_backend.datasets.get(dataset_id)
            if cache_it:
                self._parent.local_backend.datasets.put(ds)
        return ds

    @deprecated  # use get
    def get_dataset(self, dataset_id, binary=True, format=formats.numpy,
                    force_download=True, cache_it=False):
        return self.get(dataset_id, binary, format, force_download, cache_it)

    def put(self, ds: Dataset, overwrite=True, upload=True)->None:
        """
        puts a dataset to the local repository as well to server if upload is True

        :param ds: dataset to be uploaded
        :type ds: <class 'pypadre.core.datasets.Dataset'>
        :param overwrite: if false, datasets are not overwritten
        :param upload: True, if the dataset should be uploaded
        """
        # todo implement overwrite correctly
        if upload:
            trigger_event('EVENT_WARN', condition=self._parent.offline is False, source=self,
                          message="Warning: The class is set to offline put upload was set to true. "
                                  "Backend is not expected to work properly")
            if self.has_printer():
                self._print("Uploading dataset %s, %s, %s" % (ds.name, str(ds.size), ds.type))
            ds.id = self._parent.remote_backend.datasets.put(ds, True)
        self._parent.local_backend.datasets.put(ds)

    def delete(self, dataset_id, remote_also=False):
        """
        delete the dataset with the provided id
        :param dataset_id: id of dataset as string or dataset object
        :return:
        """
        if isinstance(dataset_id, Dataset):
            dataset_id = dataset_id.id
        self._parent.local_backend.datasets.delete(dataset_id)

    def import_from_csv(self, csv_path, targets, name, description):
        """Load dataset from csv file"""
        return ds_import.load_csv(csv_path, targets, name, description)
