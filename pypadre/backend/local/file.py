"""
  Module handling different repositories. A repository manages datasets and allows to load / store datasets.

  Currently we distinguish between a FileRepository and a HTTPRepository.
  In addition, the module defines serialiser for the individual binary data sets
"""
import copy
import os
import re
import shutil
import uuid

import numpy as np
from deprecated import deprecated

from pypadre.backend.serialiser import JSonSerializer, PickleSerializer
from pypadre.core.model.dataset.dataset import Dataset
from pypadre.core.model.dataset.attribute import Attribute
from pypadre.core import Experiment, Run, Split


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

        self._data_repository.put(experiment._dataset)

        with open(os.path.join(dir, "metadata.json"), 'w') as f:
            f.write(self._metadata_serializer.serialise(experiment.metadata))

        with open(os.path.join(dir, "workflow.bin"), 'wb') as f:
            f.write(self._binary_serializer.serialise(experiment.workflow))

        if experiment.requires_preprocessing:
            with open(os.path.join(dir, "preprocessing_workflow.bin"), 'wb') as f:
                f.write(self._binary_serializer.serialise(experiment.preprocessing_workflow))

    @deprecated("Use get_experiment instead")
    def get_local_experiment(self, id_, load_workflow=True):
        dir = os.path.join(self.root_dir, *self._dir(id_))

        with open(os.path.join(dir, "metadata.json"), 'r') as f:
            metadata = self._metadata_serializer.deserialize(f.read())

        with open(os.path.join(dir, "hyperparameter.json"), 'r') as f:
            hyperparameters = self._metadata_serializer.deserialize(f.read())

        workflow = None
        if load_workflow:
            with open(os.path.join(dir, "workflow.bin"), 'rb') as f:
                workflow = self._binary_serializer.deserialize(f.read())

        ex = Experiment(id_=id_, workflow=workflow,
                        dataset=self._data_repository.get_dataset(metadata["dataset_id"]))
        ex.set_hyperparameters(hyperparameters)
        return ex

    def get_experiment(self, id_):
        """Load experiment from local file system

        :param id_: Id or name of the experiment
        :return: Experiment instance
        """
        dir_ = os.path.join(self.root_dir, *self._dir(id_))
        with open(os.path.join(dir_, "workflow.bin"), 'rb') as f:
            workflow = self._binary_serializer.deserialize(f.read())
        with open(os.path.join(dir_, "experiment.json"), 'r') as f:
            configuration = self._metadata_serializer.deserialize(f.read())
        with open(os.path.join(dir_, "metadata.json"), 'r') as f:
            metadata = self._metadata_serializer.deserialize(f.read())
        with open(os.path.join(dir_, "preprocessing_workflow.bin"), 'rb') as f:
            preprocessing_workflow = self._binary_serializer.deserialize(f.read())
        experiment_params = copy.deepcopy(configuration)
        experiment_params[id_]["workflow"] = workflow.pipeline
        experiment_params[id_]["preprocessing"] = preprocessing_workflow
        dataset_name = self._data_repository.get_dataset_name_by_id(metadata["dataset_id"])
        experiment_params[id_]["dataset"] = self._data_repository.get(dataset_name)
        ex = Experiment(ex_id=id_, **experiment_params[id_])
        ex.experiment_configuration = configuration
        ex.metadata = metadata
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
            f.write(self._metadata_serializer.serialise(run.metadata))

        #Commented for pytorch integration

        with open(os.path.join(dir, "hyperparameter.json"), 'w') as f:
            params = experiment.hyperparameters()
            # This writes all data present within the params to the JSON file
            f.write(self._metadata_serializer.serialise(params))


        with open(os.path.join(dir, "workflow.bin"), 'wb') as f:
            f.write(self._binary_serializer.serialise(experiment.workflow))

    def get_run(self, ex_id, run_id):
        """Load Run with particular id from the experiment from local file system

        :param ex_id: Related experiment name for run
        :param run_id: Run name
        :return: Run instance
        """
        dir_ = os.path.join(self.root_dir, *self._dir(ex_id, run_id))
        with open(os.path.join(dir_, "metadata.json"), 'r') as f:
            metadata = self._metadata_serializer.deserialize(f.read())

        with open(os.path.join(dir_, "workflow.bin"), 'rb') as f:
            workflow = self._binary_serializer.deserialize(f.read())
        ex = self.get_experiment(ex_id)
        r = Run(ex, workflow, run_id=run_id, **dict(metadata))
        r.metadata = metadata
        return r

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
            split.id = uuid.uuid4()

        split_id = str(split.id)

        dir = os.path.join(self.root_dir, *self._dir(experiment.id, run.id, split_id))
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)
        with open(os.path.join(dir, "metadata.json"), 'w') as f:
            f.write(self._metadata_serializer.serialise(split.metadata))
        self._split_dir = dir

    def get_split(self, ex_id, run_id, split_id):
        """Load Split from local file system

        :param ex_id: Related experiment name for split
        :param run_id: Related run name for split
        :param split_id: Split name
        :param num: Split number
        :return: Split instance
        """
        dir_ = os.path.join(self.root_dir, *self._dir(ex_id, run_id, split_id))

        with open(os.path.join(dir_, "results.json"), 'r') as f:
            results = self._metadata_serializer.deserialize(f.read())
        with open(os.path.join(dir_, "metrics.json"), 'r') as f:
            metrics = self._metadata_serializer.deserialize(f.read())
        with open(os.path.join(dir_, "metadata.json"), 'r') as f:
            metadata = self._metadata_serializer.deserialize(f.read())

        train_idx = results.get("train_idx", None)
        test_idx = results.get("test_idx", None)
        val_idx = results.get("val_idx", None)

        train_idx = np.array(train_idx) if type(train_idx) is list else train_idx
        test_idx = np.array(test_idx) if type(test_idx) is list else test_idx
        val_idx = np.array(val_idx) if type(val_idx) is list else val_idx
        r = self.get_run(ex_id, run_id)
        num = results["split_num"]
        s = Split(r, num, train_idx, val_idx, test_idx, split_id=split_id, **r.metadata)
        s.run.results.append(results)
        s.run.metrics.append(metrics)
        s.metadata = metadata
        return s

    def put_results(self, experiment, run, split, results):
        """
        Write the results of a split to the backend

        :param experiment: Experiment ID
        :param run_id: Run ID of the current experiment run
        :param split_id: Split id
        :param results: results to be written to the backend

        :return: None
        """
        split_id = str(split.id)
        dir_ = os.path.join(self.root_dir, *self._dir(experiment.id, run.id, split_id))
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
        split_id = str(split.id)
        dir_ = os.path.join(self.root_dir, *self._dir(experiment.id, run.id, split_id))
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

    def update_metadata(self, data_dict, ex_id, run_id=None, split_id=None):
        """
        Update metadata property of either experiment, run or split

        :param data_dict: Dictionary containing key, value pairs for metadata e-g {"server_url": "url"}
        :param ex_id: Experiment id for which metadata should be updated
        :param run_id: Run id for which metadata should be updated
        :param split_id: Split id for which metadata should be updated
        """
        dir_ = os.path.join(self.root_dir, *self._dir(ex_id, run_id, split_id))
        with open(os.path.join(dir_, "metadata.json"), 'r+') as f:
            metadata = self._metadata_serializer.deserialize(f.read())
            metadata.update(data_dict)
            f.seek(0)
            f.write(self._metadata_serializer.serialise(metadata))

    def validate_and_save(self, experiment, run=None, split=None):
        saved = False
        if split is not None:
            dir_ = os.path.join(self.root_dir, *self._dir(experiment.id, run.id, split.id))
            if not os.path.exists(os.path.abspath(dir_)):
                self.put_split(experiment, run, split)
                saved = True
        elif run is not None:
            dir_ = os.path.join(self.root_dir, *self._dir(experiment.id, run.id))
            if not os.path.exists(os.path.abspath(dir_)):
                self.put_run(experiment, run)
                saved = True
        else:
            dir_ = os.path.join(self.root_dir, *self._dir(experiment.id))
            if not os.path.exists(os.path.abspath(dir_)):
                self.put_experiment(experiment)
                self.put_experiment_configuration(experiment)
                saved = True
        return saved


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
        return dirs #[self.get(dir, metadata_only=True) for dir in dirs]

    def get_dataset_name_id(self):
        """
        Lists all the dataset names along with the id of the datasets
        :return: List of tuples containing the dataset name and id
        """
        import json
        dataset_list = []

        # Get the names of all the datasets
        directories = os.listdir(self.root_dir)

        # Check the metadata of all the datasets and append them to the tuple
        for directory in directories:
            if os.path.exists(os.path.join(self.root_dir, directory, 'metadata.json')) and \
                    os.path.exists(os.path.join(self.root_dir, directory, 'data.bin')):

                # if the json file exists check if the tag id exists within the json
                with open(os.path.join(self.root_dir, directory, 'metadata.json'), 'r') as f:
                    metadata = json.loads(f.read())

                if metadata.get('id', None) is not None:
                    dataset_tuple = (directory, metadata.get('id', None))
                    dataset_list.append(dataset_tuple)

        return dataset_list

    def get_dataset_name_by_id(self, dataset_id):
        """
        Return dataset name for given dataset id

        :param dataset_id: Dataset id for which dataset name should searched
        :type dataset_id: str
        :return: String containing dataset name or empty string if its not found
        """
        dataset_name = ""
        for name, id_ in self.get_dataset_name_id():
            if id_ == dataset_id:
                dataset_name = name
                break
        return dataset_name

    def put(self, dataset: Dataset)-> None:
        """
        stores the provided dataset into the file backend under the directory `dataset.id`
        (file `data.bin` contains the binary and file `metadata.json` contains the metadata)
        :param dataset: dataset to put.
        :return:
        """
        _dir = _get_path(self.root_dir, str(dataset.name))
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

