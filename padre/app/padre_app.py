"""
Padre app as single point of interaction.
"""

# todo merge with cli. cli should use app and app should be configurable via builder pattern and configuration files

import os
import configparser
from beautifultable import BeautifulTable
from beautifultable.enums import Alignment
from scipy.stats.stats import DescribeResult

from padre.datasets import formats

from padre.backend.file import DatasetFileRepository, PadreFileBackend
from padre.backend.http import PadreHTTPClient
from padre.ds_import import load_sklearn_toys
from padre.ExperimentCreator import ExperimentCreator
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
        "user": "",
        "passwd": ""
    }


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

    def list_experiments(self, start=0, count=999999999, search=None):
        return self._parent.file_repository.experiments.list_experiments()

    def list_runs(self, ex_id, start=0, count=999999999, search=None):
        return self._parent.file_repository.experiments.list_runs(ex_id)


class PadreApp:

    # todo improve printing. Configure a proper printer or find a good ascii printing package

    def __init__(self, http_repo, file_repo, printer=None):
        self._http_repo = http_repo
        self._file_repo = file_repo
        self._print = printer
        self._dataset_app = DatasetApp(self)
        self._experiment_app = ExperimentApp(self)
        self._experiment_creator = ExperimentCreator()
        self._metrics_evaluator = CompareMetrics()


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
        raise NotImplemented  # todo implement a joint repository where file is used to cache http

pypadre = PadreApp(http_client, file_cache)
