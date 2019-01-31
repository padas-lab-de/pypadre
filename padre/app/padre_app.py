"""
Padre app as single point of interaction.

Defaults:

- The default configuration is provided under `.padre.cfg` in the user home directory


Architecture of the module
++++++++++++++++++++++++++

- `PadreConfig` wraps the configuration for the app. It can read/write the config from a file (if provided)

"""

# todo merge with cli. cli should use app and app should be configurable via builder pattern and configuration files
import logging
import os
import configparser

import copy
from collections import Iterable

from deprecated import deprecated
from beautifultable import BeautifulTable
from beautifultable.enums import Alignment
from scipy.stats.stats import DescribeResult

from padre.datasets import formats, Dataset

from padre.backend.file import DatasetFileRepository, PadreFileBackend
from padre.backend.http import PadreHTTPClient
from padre.backend.dual_backend import DualBackend
import padre.ds_import
from padre.experimentcreator import ExperimentCreator
from padre.experiment import Experiment
from padre.metrics import ReevaluationMetrics
from padre.metrics import CompareMetrics
from padre.base import default_logger

if "PADRE_BASE_URL" in os.environ:
    _BASE_URL = os.environ["PADRE_BASE_URL"]
else:
    _BASE_URL = "http://localhost:8080/api"

if "PADRE_CFG_FILE" in os.environ:
    _PADRE_CFG_FILE = os.environ["PADRE_CFG_FILE"]
else:
    _PADRE_CFG_FILE = os.path.expanduser('~/.padre.cfg')

logger = logging.getLogger('pypadre')

def _sub_list(l, start=-1, count=9999999999999):
    start = max(start, 0)
    stop = min(start + count, len(l))
    if start >= len(l):
        return []
    else:
        return l[start:stop]


def _wheel_char(n_max):
    chars = ["/", "-", "\\", "|", "/", "-", "\\", "|"]
    for i in range(n_max):
        yield "\r" + chars[i % len(chars)]

def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    import collections
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

def get_default_table():
    table = BeautifulTable(max_width=150, default_alignment=Alignment.ALIGN_LEFT)
    table.row_separator_char = ""
    return table

class PadreConfig:
    """
    PadreConfig class covering functionality for viewing or updating default
    configurations for PadreApp.
    Configuration file is placed at ~/.padre.cfg

    Expected values in config are following
    ---------------------------------------
    [HTTP BACKEND]
    user = username
    base_url = http://localhost:8080/api
    token = oauth_token

    [LOCAL BACKEND]
    root_dir = ~/.pypadre/

    [GENERAL]
    offline = True
    oml_key = openML_api_key
    ---------------------------------------

    Implemented functionality.

    1- Get list of dicts containing key, value pairs for all sections in config
    2- Get value for given key.
    3- Set value for given key in config
    4- Authenticate given user and update new token in the config
    """
    def __init__(self, config_file: str=_PADRE_CFG_FILE, create: bool=True, config:dict=None):
        """
        PRecedence of Configurations: default gets overwritten by file which gets overwritten by config parameter
        :param config: str pointing to the config file or None if no config file should be used
        :param create: true if the config file should be created
        :param config: Additional configuration
        """
        self._config = self.default()
        # handle file here
        self._config_file = config_file
        if self._config_file is not None:
            self.__load_config()
            if not os.path.exists(self._config_file) and create:
                self.save()
        # now merge
        self.__merge_config(config)


    def __merge_config(self, to_merge):
        # merges the provided dictionary into the config.
        if to_merge is not None:
            dict_merge(self._config, to_merge)

    def __load_config (self):
        """
        loads a padre configuration from the given file or from the standard file ~/.padre.cfg if no file is provided
        :param config_file: filename of config file
        :return: config accessable as dictionary
        """
        config = configparser.ConfigParser()
        if os.path.exists(self._config_file):
            config.read(self._config_file)
            self.__merge_config(dict(config._sections))

    def default(self):
        """
        :return: default values for
        """
        return {
            "HTTP BACKEND": {
                    "base_url": _BASE_URL,
                     "user": "mgrani",
            },
            "LOCAL BACKEND": {
                "root_dir": os.path.join(os.path.expanduser("~"), ".pypadre")
            },
            "GENERAL": {
                "offline": True
            }
        }

    @property
    def config_file(self):
        return self._config_file

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config

    @property
    def http_backend_config(self):
        # TODO: only return the necessayr parameters to avoid instantiation error
        return self._config["HTTP BACKEND"]

    @property
    def general(self):
        return self._config["GENERAL"]

    @property
    def local_backend_config(self):
        # TODO: only return the necessayr parameters to avoid instantiation error
        return self._config["LOCAL BACKEND"]

    def load(self) -> None:
        """
        reloads the configuration specified under the current config path
        """
        self.__load_config()
    def save(self) -> None:
        """
        saves the current configuration to the configured file
        """
        pconfig = configparser.ConfigParser()
        for k, v in self._config.items():
            pconfig[k] = v
        with open(self._config_file, "w") as cfile:
            pconfig.write(cfile)

    def sections(self) -> list:
        """
        :return: returns all sections of the config file
        """
        return self._config.keys()

    def set(self, key, value, section='HTTP BACKEND'):
        """
        Set value for given key in config

        :param key: Any key in config
        :type key: str
        :param value: Value to be set for given key
        :type value: str
        :param section: Section to be changed in config, default HTTP
        :type section: str
        """
        if not self.config[section]:
            self.config[section] = dict()
        self.config[section][key]=value

    def get(self, key, section='HTTP BACKEND'):
        """
        Get value for given key.
        :param key: Any key in config for any section
        :type key: str
        :return: Found value or False
        """
        return self._config[section][key]

    def authenticate(self, user=None, passwd=None):
        """
        Authenticate given user and update new token in the config.

        :param url: url of the server
        :type url: str
        :param user: Given user
        :type user: str
        :param passwd: Given password
        :type passwd: str
        """
        self.http_backend_config["user"]=user
        http = PadreHTTPClient(**self.http_backend_config)
        token = http.authenticate(passwd, user)
        self.set('token', token)
        self.save()
        self.general["offline"] = False


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
        :return:
        """
        # todo: merge results and allow multiple repositories. all should have same signature. then iterate over repos
        local_datasets = self._parent.local_backend.datasets.list(search_name, search_metadata)
        if local_datasets is None:
            local_datasets = []
        remote_datasets = []
        if not self._parent.offline:
            remote_datasets = self._parent.remote_backend.datasets.list(search_name, search_metadata, start, count)
        local_datasets.extend(remote_datasets)
        if prnt:
            self.print_datasets_details(local_datasets)
        return local_datasets

    def print_datasets_details(self, datasets: list):
        if self._parent.has_print():
            table = get_default_table()
            table.column_headers = ["ID", "Name", "Type", "#att"]
            ch_it = _wheel_char(9999999999)
            self._print("Loading.....")
            for ds in datasets:
                self._print(next(ch_it), end="")
                table.append_row(
                    [str(x) for x in [ds.id, ds.name, ds.type, ds.num_attributes]])
            self._print(table)

    def print_dataset_details(self, ds):
        if self.has_printer():
            self._print(f"Metadata for dataset {ds.id}")
            for k, v in ds.metadata.items():
                self._print("\t%s=%s" % (k, str(v)))
            self._print("Binary description:")
            for k, v in ds.describe().items():
                # todo printing the statistics is not ideal. needs to be improved
                if k == "stats" and isinstance(v, DescribeResult):
                    table = get_default_table()
                    h = ["statistic"]
                    for a in ds.attributes:
                        h.append(a.name)
                    table.column_headers = h
                    for m in [("min", v.minmax[0]), ("max", v.minmax[1]), ("mean", v.mean),
                              ("kurtosis", v.kurtosis), ("skewness", v.skewness)]:
                        r = [m[0]]
                        for val in m[1]: r.append(val)
                        table.append_row(r)
                    self._print(table)
                else:
                    self._print("\t%s=%s" % (k, str(v)))

    def search_downloads(self, name: str=None)->list:
        """
        searches for importable datasets (as specified in the datasets file).
        :param name: regexp for filtering the names
        :return: list of possible imports
        """
        from padre import ds_import
        oml_key = self._parent.config.get("oml_key", "GENERAL")
        root_dir = self._parent.config.get("root_dir", "LOCAL BACKEND")
        datasets = ds_import.search_oml_datasets(name, root_dir, oml_key)

        datasets_list = []
        for key in datasets.keys():
            datasets_list.append(datasets[key])
        return datasets_list

    def download(self, source: str, name: str) -> Iterable:
        """
        Downloads the dataset defined by name from source and stores it into the local file backend.
        :param source: name of the source
        :param name:
        :return: returns a iterator of dataset objects
        """
        # todo implement using a generator pattern to avoid loading every dataset in main memory
        pass

    def download_external(self, sources: list) -> Iterable:
        """
        Downloads the datasets their information provided as list provided as list from oml
        :return: returns a iterator of dataset objects
        """
        # todo: Merge download and download_external functions into one
        for dataset_source in sources:
            dataset = self._parent.remote_backend.datasets.load_oml_dataset(str(dataset_source["did"]))
            yield dataset


    def sync(self, name: str=None, mode: str = "sync"):
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

    @deprecated(reason="use downloads function below") # see download
    def do_default_imports(self, sklearn=True):
        if sklearn:
            for ds in padre.ds_import.load_sklearn_toys():
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

    def upload_scratchdatasets(self,auth_token,max_threads=8,upload_graphs=True):
        if(max_threads<1 or max_threads>50):
            max_threads=2
        if("api"in _BASE_URL):
            url=_BASE_URL.strip("/api")
        else:
            url=_BASE_URL
        padre.ds_import.sendTop100Datasets_multi(auth_token,url,max_threads)
        print("All openml datasets are uploaded!")
        if(upload_graphs):
            padre.ds_import.send_top_graphs(auth_token,url,max_threads>=3)

    def get(self, dataset_id, binary: bool =True,
            format=formats.numpy,
            force_download: bool =True,
            cache_it: bool =False):
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
        if not force_download: # look in cache first
            ds = self._parent.local_backend.datasets.get(dataset_id)
        if ds is None and not self._parent.offline:  # no cache or not looked --> go to http client
            ds = self._parent.remote_backend.datasets.get(dataset_id, binary, format=format)
            if cache_it:
                self._parent.local_backend.datasets.put(ds)
        return ds

    @deprecated # use get
    def get_dataset(self, dataset_id, binary=True, format=formats.numpy,
            force_download=True, cache_it=False):
        return self.get(dataset_id, binary, format, force_download,cache_it)

    def put(self, ds: Dataset, overwrite= True, upload= True)->None:
        """
        puts a dataset to the local repository
        :param dataset: dataset to be uploaded
        :param overwrite: if false, datasets are not overwritten
        :param upload: True, if the dataset should be uploaded
        """
        # todo implement overwrite correctly
        if upload:
            if self._parent.offline:
                logger.warning("Warning: The class is set to offline put upload was set to true. "+
                               "Backend is not expected to work properly")
            if self.has_printer():
                self._print("Uploading dataset %s, %s, %s" % (ds.name, str(ds.size), ds.type))
            self._parent.remote_backend.datasets.put_dataset(ds, True)
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



class ExperimentApp:
    """
    Class providing commands for managing datasets.
    """
    def __init__(self, parent):
        self._parent = parent

    def delete_experiments(self, search):
        """
           lists the experiments and returns a list of experiment names matching the criterions
           :param search: str to search experiment name only or
           dict object with format {field : regexp<String>} pattern to search in particular fields using a regexp.
           None for all experiments
        """
        if isinstance(search, dict):
            s = copy.deepcopy(search)
            file_name = s.pop("name")
        else:
            file_name = search
            s = None

        self._parent.local_backend.experiments.delete_experiments(search_id=file_name, search_metadata=s)



    def list_experiments(self, search=None, start=-1, count=999999999, ):
        """
        lists the experiments and returns a list of experiment names matching the criterions
        :param search: str to search experiment name only or
        dict object with format {field : regexp<String>} pattern to search in particular fields using a regexp.
        None for all experiments
        :param start: start in the list to be returned
        :param count: number of elements in the list to be returned
        :return:
        """
        if search is not None:
            if isinstance(search, dict):
                s = copy.deepcopy(search)
                file_name = s.pop("name")
            else:
                file_name = search
                s = None
            return _sub_list(self._parent.local_backend.experiments.list_experiments(search_id=file_name,
                                                                                     search_metadata=s)
                             , start, count)
        else:
            return _sub_list(self._parent.local_backend.experiments.list_experiments(), start, count)

    def list_runs(self, ex_id, start=-1, count=999999999, search=None):
        return _sub_list(self._parent.local_backend.experiments.list_runs(ex_id), start, count)

    def run(self, **ex_params):
        """
        runs an experiment either with the given parameters or, if there is a parameter decorated=True, runs all
        decorated experiments.
        Befor running the experiments, the backend for storing results is configured as file_repository.experiments
        :param ex_params: kwargs for an experiment or decorated=True
        :return:
        """
        if "decorated" in ex_params and ex_params["decorated"]:
            from padre.decorators import run
            return run(backend=self._parent.local_backend.experiments)
        else:
            p = ex_params.copy()
            p["backend"] = self._parent.local_backend.experiments
            ex = Experiment(**p)
            ex.run()
            return ex


class PadreApp:

    # todo improve printing. Configure a proper printer or find a good ascii printing package

    def __init__(self, config=None, printer=None):
        if config is None:
            self._config = PadreConfig()
        else:
            self._config = config
#        self._offline = "offline" not in self._config.general or self._config.general["offline"]
        self._http_repo = PadreHTTPClient(**self._config.http_backend_config)
        self._file_repo = PadreFileBackend(**self._config.local_backend_config)
        self._dual_repo = DualBackend(self._file_repo, self._http_repo)
        self._print = printer
        self._dataset_app = DatasetApp(self)
        self._experiment_app = ExperimentApp(self)
        self._experiment_creator = ExperimentCreator()
        self._metrics_evaluator = CompareMetrics()
        self._metrics_reevaluator = ReevaluationMetrics()

    @property
    def offline(self):
        """
        sets the current offline / online status of the app. Permanent changes need to be done via the config.
        :return: True, if requests are not passed to the server
        """
        return self._config.general["offline"]

    @offline.setter
    def offline(self, offline):
        self._offline = offline

    @property
    def datasets(self):
        return self._dataset_app

    @property
    def experiments(self):
        return self._experiment_app

    @property
    def experiment_creator(self):
        return self._experiment_creator

    @property
    def metrics_evaluator(self):
        return self._metrics_evaluator

    @property
    def metrics_reevaluator(self):
        return self._metrics_reevaluator

    @property
    def config(self):
        return self._config

    def set_printer(self, printer):
        """
        sets the printer, i.e. the output of console text. If None, there will be not text output
        :param printer: object with .print(str) function like sys.stdout or None
        """
        self._print = printer

    def status(self):
        """
        returns the status of the app, i.e. if the server is running, the client, the config etc.
        :return:
        """
        pass

    def print(self, output, **kwargs):
        if self.has_print():
            self._print(output, **kwargs)

    def has_print(self):
        return self._print is not None

    @property
    def remote_backend(self):
        return self._http_repo

    @property
    def local_backend(self):
        return self._file_repo

    @property
    def repository(self):
        return self._dual_repo

import sys
pypadre = PadreApp(printer=print)   # load the default app
