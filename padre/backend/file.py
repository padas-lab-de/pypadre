"""
  Module handling different repositories. A repository manages datasets and allows to load / store datasets.

  Currently we distinguish between a FileRepository and a HTTPRepository.
  In addition, the module defines serialiser for the individual binary data sets
"""

import os
import re
import shutil
import uuid

from padre.backend.serialiser import JSonSerializer, PickleSerializer
from padre.base import result_logger
from padre.datasets import Dataset, Attribute
from padre.experiment import Experiment


def _get_path(root_dir, name):
    # internal get or create path function
    _dir = os.path.expanduser(os.path.join(root_dir, name))
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    return _dir


def _dir_list(root_dir, search_id, search_metadata, strip_postfix=None):
    # todo implement search in metadata using some kind of syntax (e.g. jsonpath, grep),
    # then search the metadata files one by one.
    files = [f for f in os.listdir(root_dir) if strip_postfix==None or f.endswith(strip_postfix)]
    if search_id is not None:
        rid = re.compile(search_id)
        files = [f for f in files if rid.match(f)]

    if search_metadata is not None:
        raise NotImplemented()

    return [file[:-1*len(strip_postfix)] for file in files if file is not None and len(file)>=len(strip_postfix)]


class PadreFileBackend(object):
    """
    Delegator class for handling padre objects at the file repository level. The following files tructure is used:

    root_dir
      |------datasets\
      |------experiments\
    """

    def __init__(self, root_dir):
        self.root_dir = _get_path(root_dir, "")
        # todo: can we duck patch this class to contain all functions of the delegates directly?
        self._dataset_repository = DatasetFileRepository(os.path.join(root_dir, "datasets"))
        self._experiment_repository = ExperimentFileRepository(os.path.join(root_dir, "experiments"),
                                                               self._dataset_repository)

    @property
    def datasets(self):
        return self._dataset_repository

    @property
    def experiments(self):
        return self._experiment_repository


class ExperimentFileRepository:
    """
    repository for handling experiments as File Directory with the following format

    ```
    root_dir
       |-<experiment name>.ex
               |-- metadata.json
               |-- hyperparameter.json
               |-- aggregated_scores.json
               |-- runs
                  |-- scores.json
                  |-- events.json
                  |-- split 0
                      |-- model.bin
                      |-- split_idx.bin
                      |-- results.bin
                      |-- log.json
                  |-- split 1
                      .....
       |-<experiment2 name>.ex
       ...

    Note that `events.json` and `scores.json` contain the events / scores of the individual splits.
    So logically they would belong to the splits.
    However, for convenience reasons they are aggregated at the run level.
    """

    def __init__(self, root_dir, data_repository):
        self.root_dir = _get_path(root_dir, "")
        self._metadata_serializer = JSonSerializer
        self._binary_serializer = PickleSerializer
        self._data_repository = data_repository

    def _dir(self, ex_id, run_id=None, split_num=None):
        r = [str(ex_id)+".ex"]
        if run_id is not None:
            r.append(str(run_id)+".run")
            if split_num is not None:
                r.append(str(split_num)+".split")
        return r

    def list_experiments(self, search_id=None, search_metadata=None):
        """
        list the experiments available
        :param search_id:
        :param search_metadata:
        :return:
        """

        return _dir_list(self.root_dir, search_id, search_metadata, ".ex")

    def put_experiment(self, experiment, allow_overwrite=False):
        """
        Stores an experiment to the file. Only metadata, hyperparameter and the workflow is stored.
        :param experiment:
        :param allow_overwrite: True if an existing experiment can be overwritten
        :return:
        """
        from padre.base import default_logger
        if experiment.id is None:  #  this is a new experiment
            if experiment.name is None or experiment.name == "":
                experiment.id = uuid.uuid1()
            else:
                experiment.id = experiment.name
        dir = os.path.join(self.root_dir, *self._dir(experiment.id))
        if os.path.exists(dir) and not allow_overwrite:
            raise ValueError("Experiment %s already exists." +
                             "Overwriting not explicitly allowed. Set allow_overwrite=True")
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)
        with open(os.path.join(dir, "metadata.json"), 'w') as f:
            f.write(self._metadata_serializer.serialise(experiment.metadata))

        with open(os.path.join(dir, "workflow.bin"), 'wb') as f:
            f.write(self._binary_serializer.serialise(experiment._workflow))

        default_logger.open_log_file(dir)

    def get_experiment(self, id_, load_workflow=True):
        dir = os.path.join(self.root_dir, *self._dir(id_))

        with open(os.path.join(dir, "metadata.json"), 'r') as f:
            metadata = self._metadata_serializer.deserialize(f.read())

        with open(os.path.join(dir, "hyperparameter.json"), 'r') as f:
            hyperparameters = self._metadata_serializer.deserialize(f.read())

        workflow = None
        if load_workflow:
            with open(os.path.join(dir, "workflow.bin"), 'r') as f:
                workflow = self._binary_serializer.deserialize(f.read())

        ex = Experiment(id_=id_, workflow=workflow,
                        dataset=self._data_repository.get_dataset(metadata["dataset_id"]))
        ex.set_hyperparameters(hyperparameters)
        return ex

    def list_runs(self, experiment_id, search_id=None, search_metadata=None):
        """
        list the runs of an experiment
        :param experiment_id:
        :param search_id:
        :param search_metadata:
        :return:
        """
        return _dir_list(os.path.join(self.root_dir, *self._dir(experiment_id)), search_id, search_metadata, ".run")

    def put_run(self, experiment, run):
        """
        Stores a run of an experiment to the file repository.
        :param experiment: experiment the run is part of
        :param run: run to put
        :return:
        """
        if run.id is None:  # this is a new experiment
            run.id = uuid.uuid1()

        dir = os.path.join(self.root_dir, *self._dir(experiment.id, run.id))
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)

        with open(os.path.join(dir, "metadata.json"), 'w') as f:
            f.write(self._metadata_serializer.serialise(experiment.metadata))

        with open(os.path.join(dir, "hyperparameter.json"), 'w') as f:
            params = experiment.hyperparameters()
            #for key in params:
                # This writes all data present within the params to the JSON file
            f.write(self._metadata_serializer.serialise(params))

        with open(os.path.join(dir, "workflow.bin"), 'wb') as f:
            f.write(self._binary_serializer.serialise(experiment._workflow))

    def get_run(self, ex_id, run_id):
        """
        get the run with the particular id from the experiment.
        :param ex_id:
        :param run_id:
        :return:
        """
        dir = os.path.join(self.root_dir, *self._dir(ex_id, run_id))

        with open(os.path.join(dir, "metadata.json"), 'r') as f:
            metadata = self._metadata_serializer.deserialize(f.read())

        return None

    def list_splits(self, experiment_id, run_id, search_id=None, search_metadata=None):
        """
        list the runs of an experiment
        :param experiment_id:
        :param search_id:
        :param search_metadata:
        :return:
        """
        return _dir_list(os.path.join(self.root_dir, *self._dir(experiment_id, run_id)),
                         search_id, search_metadata, ".split")

    def put_split(self, experiment, run, split):
        """
        Stores a run of an experiment to the file repository.
        :param experiment: experiment the run is part of
        :param run: run to put
        :return:
        """
        if split.id is None:  # this is a new experiment
            split.id = str(split.number)+":"+str(uuid.uuid1())

        dir = os.path.join(self.root_dir, *self._dir(experiment.id, run.id, split.id))
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)
        with open(os.path.join(dir, "metadata.json"), 'w') as f:
            f.write(self._metadata_serializer.serialise(experiment.metadata))

        # Set the directory for logging
        result_logger.set_log_directory(dir)

    def get_split(self, ex_id, run_id, split_id):
        """
        get the run with the particular id from the experiment.
        :param ex_id:
        :param run_id:
        :return:
        """
        dir = os.path.join(self.root_dir, *self._dir(ex_id, run_id, split_id))

        with open(os.path.join(dir, "metadata.json"), 'r') as f:
            metadata = self._metadata_serializer.deserialize(f.read())

        return None

    def _do_print(self):
        return True



class DatasetFileRepository(object):
    """
    repository for handling datasets as File Directory with the following format

    ```
    root_dir
       |-<dataset name>
               |-- data.bin
               |-- metadata.json
       |-<dataset2 name>
       ...

    """

    def __init__(self, root_dir):
        self.root_dir = _get_path(root_dir, "")
        self._metadata_serializer = JSonSerializer
        self._data_serializer = PickleSerializer

    def list_datasets(self, search_id=None, search_metadata=None):
        """
        List all data sets in the repository
        :param search_name: regular expression based search string for the title. Default None
        :param search_metadata: dict with regular expressions per metadata key. Default None
        """
        return _dir_list(self.root_dir, search_id, search_metadata)

    def put_dataset(self, dataset):
        _dir = _get_path(self.root_dir, dataset.id)
        try:
            if dataset.has_data():
                with open(os.path.join(_dir, "data.bin"), 'wb') as f:
                    f.write(self._data_serializer.serialise(dataset.data))

            with open(os.path.join(_dir, "metadata.json"), 'w') as f:
                metadata = dict(dataset.metadata)
                metadata["attributes"] = dataset.attributes
                f.write(self._metadata_serializer.serialise(metadata))

        except Exception as e:
            shutil.rmtree(_dir)
            raise e


    def get_dataset(self, id, metadata_only=False):
        """
        Fetches a data set with `name` and returns it (plus some metadata)

        :param id:
        :return: returns the dataset or the metadata if metadata_only is True
        """
        _dir = _get_path(self.root_dir, id)

        with open(os.path.join(_dir, "metadata.json"), 'r') as f:
            metadata = self._metadata_serializer.deserialize(f.read())
            attributes = metadata.pop("attributes")

        ds = Dataset(id, metadata)
        sorted(attributes, key=lambda a: a["index"])
        assert sum([int(a["index"]) for a in attributes]) == len(attributes) * (
            len(attributes) - 1) / 2  # check attribute correctness here
        ds.set_data(None,
                    [Attribute(a["name"], a["measurementLevel"], a["unit"], a["description"],
                               a["defaultTargetAttribute"])
                     for a in attributes])
        if metadata_only:
            return ds
        elif os.path.exists(os.path.join(_dir, "data.bin")):
            data = None
            with open(os.path.join(_dir, "data.bin"), 'rb') as f:
                data = self._data_serializer.deserialize(f.read())
            ds.set_data(data, ds.attributes)

