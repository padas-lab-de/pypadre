"""
Padre app as single point of interaction.
"""

# todo merge with cli. cli should use app and app should be configurable via builder pattern and configuration files

import os
import configparser

import copy
from beautifultable import BeautifulTable
from beautifultable.enums import Alignment
from scipy.stats.stats import DescribeResult

from padre.datasets import formats

from padre.backend.file import DatasetFileRepository, PadreFileBackend
from padre.backend.http import PadreHTTPClient
from padre.backend.dual_backend import DualBackend
from padre.ds_import import load_sklearn_toys
from padre.ExperimentCreator import ExperimentCreator
from padre.experiment import Experiment
from padre.metrics import ReevaluationMetrics
from padre.metrics import CompareMetrics

if "PADRE_BASE_URL" in os.environ:
    _BASE_URL = os.environ["PADRE_BASE_URL"]
else:
    _BASE_URL = "http://localhost:8080/api"

if "PADRE_CFG_FILE" in os.environ:
    _PADRE_CFG_FILE = os.environ["PADRE_CFG_FILE"]
else:
    _PADRE_CFG_FILE = os.path.expanduser('~/.padre.cfg')

_DEFAULT_HTTP_CONFIG = {
        "base_url": _BASE_URL,
        "user": "hmafnan",
        "passwd": "test"
    }


def _sub_list(l, start=-1, count=9999999999999):
    start = max(start, 0)
    stop = min(start + count, len(l))
    if start >= len(l):
        return []
    else:
        return l[start:stop]


def load_padre_config(config_file = _PADRE_CFG_FILE):
    """
    loads a padre configuration from the given file or from the standard file ~/.padre.cfg if no file is provided
    :param config_file: filename of config file
    :return: config accessable as dictionary
    """
    config = configparser.ConfigParser()
    if os.path.exists(config_file):
        config.read(config_file)
        return dict(config._sections)
    else:
        return {
            "HTTP": _DEFAULT_HTTP_CONFIG,
            "FILE_CACHE": {
                "root_dir": os.path.expanduser("~/.pypadre/")
            }
        }

def save_padre_config(config, config_file = _PADRE_CFG_FILE):
    """
    saves the given config file which should contain the config in a dict like structure. If the config is None,
    the default configuration will be written to the file
    :param config: dict like structure with config or None,
    :param config_file: filename of config file.
    :return:
    """
    if config is None:
        config = default_config
    pconfig = configparser.ConfigParser()
    for k, v in config.items():
        pconfig[k] = v
    with open(config_file, "w") as cfile:
        config.write(cfile)


default_config = load_padre_config()
http_client = PadreHTTPClient(**default_config["HTTP"])
file_cache = PadreFileBackend(**default_config["FILE_CACHE"])

def _wheel_char(n_max):
    chars = ["/", "-", "\\", "|", "/", "-", "\\", "|"]
    for i in range(n_max):
        yield "\r" + chars[i % len(chars)]


def get_default_table():
    table = BeautifulTable(max_width=150, default_alignment=Alignment.ALIGN_LEFT)
    table.row_seperator_char = ""
    return table


class PadreConfig:
    """
    PadreConfig class covering functionality for viewing or updating default
    configurations for PadreApp.
    Configuration file is placed at ~/.padre.cfg

    Expected values in config are following
    ---------------------------------------
    [HTTP]
    user = username
    passwd = user_password
    base_url = http://localhost:8080/api
    token = oauth_token

    [FILE_CACHE]
    root_dir = ~/.pypadre/
    ---------------------------------------

    Implemented functionality.

    1- Get list of dicts containing key, value pairs for all sections in config
    2- Get value for given key.
    3- Set value for given key in config
    4- Authenticate given user and update new token in the config
    """
    def __init__(self, config_file=_PADRE_CFG_FILE):
        self._config_file = config_file

    def list(self):
        """
        Get list of dicts containing key, value pairs for all sections in config

        :return: List of dicts
        :rtype: list
        """
        config = configparser.ConfigParser()
        config_list = []
        if os.path.exists(self._config_file):
            config.read(self._config_file)
            for section in config.sections():
                data = dict()
                data[section] = dict()
                for (k, v) in config.items(section):
                    data[section][k] = v
                config_list.append(data)
        return config_list

    def set_section(self, data, section='HTTP'):
        """
        Set section in config for given list of (key, value) pairs

        :param data: dict containing key, value pair
        :type data: dict
        :return: None
        """
        config = configparser.ConfigParser()
        if os.path.exists(self._config_file):
            config.read(self._config_file)
        for k, v in data.items():
            config[section][k] = v
        with open(self._config_file, 'w') as configfile:
            config.write(configfile)

    def set(self, key, value, section='HTTP'):
        """
        Set value for given key in config

        :param key: Any key in config
        :type key: str
        :param value: Value to be set for given key
        :type value: str
        :param section: Section to be changed in config, default HTTP
        :type section: str
        """
        data = dict()
        data[key] = value
        self.set_section(data, section)

    def get(self, key):
        """
        Get value for given key.
        :param key: Any key in config for any section
        :type key: str
        :return: Found value or False
        """
        config = configparser.ConfigParser()
        if os.path.exists(self._config_file):
            config.read(self._config_file)
            for section in config.sections():
                for k, v in config.items(section):
                    if k == key:
                        return v
        return False

    def authenticate(self, url, username=_DEFAULT_HTTP_CONFIG['user'],
                     password=_DEFAULT_HTTP_CONFIG['passwd']):
        """
        Authenticate given user and update new token in the config.

        :param url: url of the server
        :type url: str
        :param username: Given user or default user from config
        :type username: str
        :param password: Given password or default password from config
        :type username: str
        """
        import requests
        import json
        token = None
        api = url
        csrf = requests.get(url).cookies.get("XSRF-TOKEN")
        url = api + "/oauth/token?=" + csrf
        data = {'username': username, 'password': password, 'grant_type': 'password'}
        response = requests.post(url, data)
        if response.status_code == 200:
            token = "Bearer " + json.loads(response.content)['access_token']
        self.set('token', token)


class DatasetApp:
    """
    Class providing commands for managing datasets.
    """
    def __init__(self, parent):
        self._parent = parent

    def list_datasets(self, start=0, count=999999999, search=None):
        datasets = self._parent.http_repository.list_datasets(start, count, search)

        if self._parent.has_print():
            table = get_default_table()
            table.column_headers = ["ID", "Name", "Type", "#att", "created"]
            ch_it = _wheel_char(9999999999)
            self._print("Loading.....")
            for ds in datasets:
                print(next(ch_it), end="")
                table.append_row([str(x) for x in [ds.id, ds.name, ds.type, ds.num_attributes, ds.metadata["createdAt"]]])
            self._print(table)
        return datasets

    def do_default_imports(self, sklearn=True):
        if sklearn:
            for ds in load_sklearn_toys():
               self.do_import(ds)

    def _print(self, output):
        self._parent.print(output)

    def has_printer(self):
        return self._parent.has_print()

    def do_import(self, ds):
        if self.has_printer():
            self._print("Uploading dataset %s, %s, %s" % (ds.name, str(ds.size), ds.type))
        self._parent.http_repository.upload_dataset(ds, True)

    def get_dataset(self, dataset_id, binary=True, format=formats.numpy,
            force_download=True, cache_it=False):
        # todo check force_download=False and cache_it True
        ds = None
        if not force_download: # look in cache first
            ds = self._parent.file_repository.get_dataset(dataset_id)
        if ds is None: # no cache or not looked --> go to http client
            ds = self._parent.http_repository.get_dataset(dataset_id, binary, format=format)
            if cache_it:
                self._parent.file_repository.put_dataset(ds)

        if self.has_printer():
            self._print(f"Metadata for dataset {ds.id}")
            for k, v in ds.metadata.items():
                self._print("\t%s=%s" % (k, str(v)))
            self._print("Available formats:")
            formats = self._parent.http_repository.get_dataset_formats(dataset_id)
            for f in formats:
                self._print("\t%s" % (f))
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
        return ds

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

        self._parent.file_repository.experiments.delete_experiments(search_id=file_name, search_metadata=s)



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
            return _sub_list(self._parent.file_repository.experiments.list_experiments(search_id=file_name,
                                                                                       search_metadata=s)
                             , start, count)
        else:
            return _sub_list(self._parent.file_repository.experiments.list_experiments(), start, count)

    def list_runs(self, ex_id, start=-1, count=999999999, search=None):
        return _sub_list(self._parent.file_repository.experiments.list_runs(ex_id), start, count)

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
            return run(backend=self._parent.file_repository.experiments)
        else:
            p = ex_params.copy()
            p["backend"] = self._parent.file_repository.experiments
            ex = Experiment(**p)
            ex.run()
            return ex



class PadreApp:

    # todo improve printing. Configure a proper printer or find a good ascii printing package

    def __init__(self, http_repo, file_repo, printer=None):
        self._http_repo = http_repo
        self._file_repo = file_repo
        self._dual_repo = DualBackend(file_repo, http_repo)
        self._print = printer
        self._dataset_app = DatasetApp(self)
        self._experiment_app = ExperimentApp(self)
        self._experiment_creator = ExperimentCreator()
        self._metrics_evaluator = CompareMetrics()
        self._metrics_reevaluator = ReevaluationMetrics()
        self._config = PadreConfig()


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

    def print(self, output):
        if self.has_print():
            self._print(output)

    def has_print(self):
        return self._print is not None

    @property
    def http_repository(self):
        return self._http_repo

    @property
    def file_repository(self):
        return self._file_repo

    @property
    def repository(self):
        return self._dual_repo

pypadre = PadreApp(http_client, file_cache)
