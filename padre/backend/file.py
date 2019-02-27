"""
  Module handling different repositories. A repository manages datasets and allows to load / store datasets.

  Currently we distinguish between a FileRepository and a HTTPRepository.
  In addition, the module defines serialiser for the individual binary data sets
"""

import os
import re
import shutil
import uuid

from deprecated import deprecated

from padre.backend.serialiser import JSonSerializer, PickleSerializer
from padre.core.datasets import Dataset, Attribute
from padre.core import Experiment


def _get_path(root_dir, name, create=True):
    # internal get or create path function
    _dir = os.path.expanduser(os.path.join(root_dir, name))
    if not os.path.exists(_dir) and create:
        os.mkdir(_dir)
    return _dir


def _dir_list(root_dir, matcher, strip_postfix=""):
    files = [f for f in os.listdir(root_dir) if f.endswith(strip_postfix)]
    if matcher is not None:
        rid = re.compile(matcher)
        files = [f for f in files if rid.match(f)]

    if len(strip_postfix) == 0:
        return files
    else:
        return [file[:-1*len(strip_postfix)] for file in files
            if file is not None and len(file) >= len(strip_postfix)]


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
    _file = None

    def __init__(self, root_dir, data_repository):
        self.root_dir = _get_path(root_dir, "")
        self._split_dir = _get_path(root_dir, "")
        self._metadata_serializer = JSonSerializer
        self._binary_serializer = PickleSerializer
        self._data_repository = data_repository
        self._file = None

    def __del__(self):
        """
        Destructor that closes the opened log file at the end of the experiments
        :return:
        """
        if self._file is not None:
            self._file.close()
            self._file = None

    def _dir(self, ex_id, run_id=None, split_num=None):
        r = [str(ex_id)+".ex"]
        if run_id is not None:
            r.append(str(run_id)+".run")
            if split_num is not None:
                r.append(str(split_num)+".split")
        return r

    def list_experiments(self, search_id=".*", search_metadata=None):
        """
        list the experiments available
        :param search_id:
        :param search_metadata:
        :return:
        """
        returns = _dir_list(self.root_dir, search_id+".ex", ".ex")
        # todo: implement search metadata filter
        return returns

    def delete_experiments(self, search_id=".*", search_metadata=None):
        """
        list the experiments available
        :param search_id:
        :param search_metadata:
        :return:
        """
        dirs = _dir_list(self.root_dir, search_id+".ex")
        # todo: implement search metadata filter
        for d in dirs:
            shutil.rmtree(_get_path(self.root_dir, d, False))


    def put_experiment(self, experiment, append_runs=False, allow_overwrite=True):
        """
        Stores an experiment to the file. Only metadata, hyperparameter and the workflow is stored.
        :param experiment:
        :param append_runs: True if runs can be added to an existing experiment.
        If false, any existing experiment will be removed
        :return:
        """

        if experiment.id is None:  #  this is a new experiment
            if experiment.name is None or experiment.name == "":
                experiment.id = uuid.uuid4()
            else:
                experiment.id = experiment.name
        dir = os.path.join(self.root_dir, *self._dir(experiment.id))
        if os.path.exists(dir) and not allow_overwrite:
            raise ValueError("Experiment %s already exists." +
                             "Overwriting not explicitly allowed. Set allow_overwrite=True")

        # Create the log file for the experiment here
        if self._file is not None:
            self._file.close()
            self._file = None

        if os.path.exists(dir):
            if not append_runs:
                shutil.rmtree(dir)
                os.mkdir(dir)
        else:
            os.mkdir(dir)

        self._file = open(os.path.join(dir, "log.txt"), "a")

        with open(os.path.join(dir, "metadata.json"), 'w') as f:
            f.write(self._metadata_serializer.serialise(experiment.metadata))

        with open(os.path.join(dir, "workflow.bin"), 'wb') as f:
            f.write(self._binary_serializer.serialise(experiment.workflow))

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
        # todo: impelement filter on metadata
        return _dir_list(os.path.join(self.root_dir, *self._dir(experiment_id)), search_id, ".run")

    def put_experiment_configuration(self, experiment):
        """
        Writes the experiment details as a json file

        :param experiment: the experiment to be written to the disk

        :return:
        """
        if experiment.experiment_configuration is not None:
            with open(os.path.join(self.root_dir, *self._dir(experiment.id), "experiment.json"), 'w') as f:
                f.write(self._metadata_serializer.serialise(experiment.experiment_configuration))

    def put_run(self, experiment, run):
        """
        Stores a run of an experiment to the file repository.
        :param experiment: experiment the run is part of
        :param run: run to put
        :return:
        """
        if run.id is None:  # this is a new experiment
            run.id = uuid.uuid4()

        dir = os.path.join(self.root_dir, *self._dir(experiment.id, run.id))
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)

        with open(os.path.join(dir, "metadata.json"), 'w') as f:
            f.write(self._metadata_serializer.serialise(experiment.metadata))

        #Commented for pytorch integration

        with open(os.path.join(dir, "hyperparameter.json"), 'w') as f:
            params = experiment.hyperparameters()
            # This writes all data present within the params to the JSON file
            f.write(self._metadata_serializer.serialise(params))


        with open(os.path.join(dir, "workflow.bin"), 'wb') as f:
            f.write(self._binary_serializer.serialise(experiment.workflow))

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
            split.id = str(split.number)+"_"+str(uuid.uuid4())

        dir = os.path.join(self.root_dir, *self._dir(experiment.id, run.id, split.id))
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)
        with open(os.path.join(dir, "metadata.json"), 'w') as f:
            f.write(self._metadata_serializer.serialise(experiment.metadata))
        self._split_dir = dir

    def get_split(self, ex_id, run_id, split_id):
        """
        get the run with the particular id from the experiment.
        :param ex_id:
        :param run_id:
        :return:
        """
        dir_ = os.path.join(self.root_dir, *self._dir(ex_id, run_id, split_id))

        with open(os.path.join(dir_, "metadata.json"), 'r') as f:
            metadata = self._metadata_serializer.deserialize(f.read())

        return None

    def put_results(self, experiment, run, split, results):
        """
        Write the results of a split to the backend

        :param experiment: Experiment ID
        :param run_id: Run ID of the current experiment run
        :param split_id: Split id
        :param results: results to be written to the backend

        :return: None
        """

        dir_ = os.path.join(self.root_dir, *self._dir(experiment.id, run.id, split.id))
        with open(os.path.join(dir_, "results.json"), 'w') as f:
            f.write(self._metadata_serializer.serialise(results))

    def put_metrics(self, experiment, run, split, metrics):
        """
        Writes the metrics of a split to the backend

        :param experiment: Experiment ID
        :param run: Run Id of the experiment
        :param split: Split ID
        :param metrics: dictionary containing all the required metrics to be written to the backend

        :return: None
        """
        dir_ = os.path.join(self.root_dir, *self._dir(experiment.id, run.id, split.id))
        with open(os.path.join(dir_, "metrics.json"), 'w') as f:
            f.write(self._metadata_serializer.serialise(metrics))

    def _do_print(self):
        return True

    def log(self, message):
        """
        This function logs all the messages to a file backend

        :param message: Message to be written to a file

        :return:
        """
        if self._file is None:
            self._file = open(os.path.join(self.root_dir, "log.txt"), "a")

        self._file.write(message + "\n")

    def log_experiment_progress(self, curr_value, limit, phase):
        """

        :param curr_value:
        :param limit:
        :param phase:
        :return:
        """
        if self._file is None:
            self._file = open(os.path.join(self.root_dir, "log.txt"), "a")

        self._file.write("EXPERIMENT PROGRESS: {curr_value}/{limit}. phase={phase} \n".format(phase=phase,
                                                                                              curr_value=curr_value,
                                                                                              limit=limit))

    def log_run_progress(self, curr_value, limit, phase):
        """

        :param curr_value:
        :param limit:
        :param phase:
        :return:
        """
        if self._file is None:
            self._file = open(os.path.join(self.root_dir, "log.txt"), "a")

        self._file.write("RUN PROGRESS: {curr_value}/{limit}. phase={phase} \n".format(phase=phase,
                                                                                       curr_value=curr_value,
                                                                                       limit=limit))

    def log_split_progress(self, curr_value, limit, phase):
        """

        :param curr_value:
        :param limit:
        :param phase:
        :return:
        """
        if self._file is None:
            self._file = open(os.path.join(self.root_dir, "log.txt"), "a")

        self._file.write("SPLIT PROGRESS: {curr_value}/{limit}. phase={phase} \n".format(phase=phase,
                                                                                         curr_value=curr_value,
                                                                                         limit=limit))

    def log_progress(self, message, curr_value, limit, phase):
        """

        :param message:
        :param curr_value:
        :param limit:
        :param phase:
        :return:
        """
        if self._file is None:
            self._file = open(os.path.join(self.root_dir, "log.txt"), "a")

        self._file.write("PROGRESS: {curr_value}/{limit}. phase={phase}. Message:{message} \n".
                         format(phase=phase, curr_value=curr_value, limit=limit,
                                message=message))

    def log_end_experiment(self):
        self._file.close()
        self._file = None

    def log_model(self, model, framework, modelname, finalmodel=False):
        """
        Logs an intermediate model to the backend
        :param model: Model to be logged
        :param framework: Framework of the model
        :param modelname: Name of the intermediate model
        :param finalmodel: Boolean value indicating whether the model is the final one or not
        :return:
        """
        if framework == 'pytorch':
            import torch
            import os
            path = os.path.join(self._split_dir,modelname)
            torch.save(model, path)



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

    def list(self, search_name=None, search_metadata=None) -> list:
        """
        List all data sets in the repository
        :param search_name: regular expression based search string for the title. Default None
        :param search_metadata: dict with regular expressions per metadata key. Default None
        """
        # todo apply the search metadata filter.
        dirs = _dir_list(self.root_dir, search_name)
        return [self.get(dir, metadata_only=True) for dir in dirs]


    def put(self, dataset: Dataset)-> None:
        """
        stores the provided dataset into the file backend under the directory `dataset.id`
        (file `data.bin` contains the binary and file `metadata.json` contains the metadata)
        :param dataset: dataset to put.
        :return:
        """
        _dir = _get_path(self.root_dir, str(dataset.id))
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

    @deprecated(reason="use put")
    def put_dataset(self, dataset):
        self.put(dataset)

    def get(self, id, metadata_only=False):
        """
        Fetches a data set with `name` and returns it (plus some metadata)

        :param id:
        :return: returns the dataset or the metadata if metadata_only is True
        """
        _dir = _get_path(self.root_dir, id)

        with open(os.path.join(_dir, "metadata.json"), 'r') as f:
            metadata = self._metadata_serializer.deserialize(f.read())
        attributes = metadata.pop("attributes")
        # print(type(metadata))
        ds = Dataset(id, **metadata)
        #sorted(attributes, key=lambda a: a["index"])
        #assert sum([int(a["index"]) for a in attributes]) == len(attributes) * (
        #    len(attributes) - 1) / 2  # check attribute correctness here
        attributes = [Attribute(a["name"], a["measurementLevel"], a["unit"], a["description"],
                               a["defaultTargetAttribute"], a["context"], a["index"])
                     for a in attributes]
        ds.set_data(None,attributes)
        if os.path.exists(os.path.join(_dir, "data.bin")):
            def __load_data():
                with open(os.path.join(_dir, "data.bin"), 'rb') as f:
                    data = self._data_serializer.deserialize(f.read())
                return data, attributes
            ds.set_data(__load_data)
        return ds

    def delete(self, id):
        """
        :param id:
        :return:
        """
        _dir = _get_path(self.root_dir, id)
        if os.path.exists(_dir):
            shutil.rmtree(_dir)


    @deprecated(reason="use get")
    def get_dataset(self, id, metadata_only=False):
        return self.get(id, metadata_only)

