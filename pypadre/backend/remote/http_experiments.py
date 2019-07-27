"""
Logic to upload experiment data to server goes here
"""
import copy
import io
import json
import re
import tempfile
import uuid
from itertools import groupby

import requests as req
import numpy as np
from requests_toolbelt import MultipartEncoder
from google.protobuf.internal.encoder import _VarintBytes

from pypadre.backend.http.protobuffer.protobuf import resultV1_pb2 as proto
from pypadre import experimentcreator
from pypadre.core import Experiment, Run, Split
from pypadre.backend.serialiser import PickleSerializer
from pypadre.eventhandler import trigger_event


class HttpBackendExperiments:
    """Experiment handler class to communicate to server"""
    def __init__(self, http_client, project_name="Default project"):
        """
        This initializes the Uploader class for given experiment.
        """
        self._http_client = http_client
        self._binary_serializer = PickleSerializer
        self.dataset_id = None
        self.experiment_id = None
        self.project_name = project_name

    def get_or_create_project(self, name):
        id_ = self.get_id_by_name(name, self._http_client.paths["projects"][1:])
        if id_ is None:
            id_ = self.create_project(name)
        return id_

    def get_or_create_dataset(self, ds):
        """Get or create new dataset

        If uid not given then check if dataset with this name already exists, if it exists then use it in experiment
        If dataset not found  then check if dataset with this id exists and if exists use it in experiment
        If dataset with given uid or same name does not exists then put this dataset to server
        """
        _id = ds.metadata.get("uid", None)
        get_url = self._http_client.get_base_url() + self._http_client.paths["dataset"](str(_id))
        dataset_id = None
        if self._http_client.online:
            try:
                if _id is None:  # Uid not given
                    dataset_id = self.get_id_by_name(ds.metadata.get("name"), self._http_client.paths["datasets"][1:])
                if dataset_id is None:
                    response = self._http_client.do_get(get_url)
                    dataset_id = json.loads(response.content)["uid"]

            except req.HttpError as e:
                trigger_event('EVENT_WARN', condition=False, source=self,
                              message="Dataset with id {%s} not found  " % str(_id))
                dataset_id = self._http_client.datasets.put(ds)
        return dataset_id

    def get_id_by_name(self, name, entity):
        """Get entity id by name

        :param name: Instance name of the entity to be searched on server
        :param entity: Name of entity e-g /projects, /datasets
        :returns: id of instance or None
        """
        id_ = None
        url = self.get_base_url() + self._http_client.paths["search"](entity) +"name:" + name
        if self._http_client.online:
            response = json.loads(self._http_client.do_get(url, **{}).content)
            if "_embedded" in response:
                first_entity = response["_embedded"][entity][0]
                if first_entity["name"] != name:  # Todo: After fixing url encoding for special chars remove this
                    return None
                id_ = first_entity["uid"]
        return id_

    def create_project(self, name):
        """Create project on server

        :param name: Name of the project
        :returns: Id of the instance or None
        """
        url = self.get_base_url() + self._http_client.paths["projects"]
        data = {"name": name, "owner": self._http_client.user}
        if self._http_client.online:
            response = self._http_client.do_post(url, **{"data": json.dumps(data)})
            return response.headers['Location'].split('/')[-1]
        return None

    def create_experiment(self, data):
        """Create experiment on server"""
        url = self.get_base_url() + self._http_client.paths["experiments"]
        location = ''
        if self._http_client.online:
            response = self._http_client.do_post(url, **{"data": json.dumps(data)})
            self.experiment_id = response.headers['Location'].split('/')[-1]
            location = response.headers['Location']
            data["metadata"]["server_url"] = location
            self._http_client.do_patch(location, **{"data": json.dumps({"metadata": data["metadata"]})})
        return location

    def put_experiment(self, experiment, append_runs=None):
        """
        Upload experiment to server with hyperparameters, configuration and metadata.

        Before uploading experiment makes sure related dataset and project already exists on server.

        :param experiment: Experiment instance
        :type experiment: <class 'padre.core.experiment.Experiment'>
        :returns: Url for newly created experiment on server
        :rtype: str

        Todo: Handle append_runs argument
        """

        dataset_dict = experiment.dataset
        self.dataset_id = self.get_or_create_dataset(dataset_dict)
        experiment_data = dict()
        experiment_data["name"] = experiment.metadata["name"]
        experiment_data["description"] = experiment.metadata["description"]
        experiment_data["algorithm"] = "http://padre.de/algorithm/1"
        experiment_data["executable"] = "http://padre.de/executable/1"
        experiment_data["sourceCode"] = "http://padre.de/executable/1"
        experiment_data["metadata"] = experiment.metadata
        experiment_data["published"] = True
        experiment_data["type"] = "http://www.padre-lab.eu/onto/Classification"
        experiment_data["uuid"] = str(uuid.uuid4())
        experiment_data["uid"] = 0
        experiment_data["links"] = [{
              "deprecation": "string",
              "href": "string",
              "hreflang": "string",
              "media": "string",
              "rel": "string",
              "templated": "true",
              "title": "string",
              "type": "string"
            }]
        experiment_data["projectId"] = self.get_or_create_project(self.project_name)
        experiment_data["datasetId"] = self.dataset_id
        experiment_data["pipeline"] = {"components": [
            {"description": experiment.metadata["description"],
             "hyperparameters": self.build_hyperparameters_list(experiment.hyperparameters()),
             "name": experiment.metadata["name"]}
        ]}
        experiment_data["configuration"] = experiment.experiment_configuration

        url = self.create_experiment(experiment_data)
        experiment.metadata["server_url"] = url
        return url

    def delete_experiment(self, ex):
        """
        Delete experiment from the server.

        :param ex: Id of the experiment or name of the experiment
        :type ex: str

        # todo: Implement delete by experiment name
        """
        if ex.isdigit() and self._http_client.online:
            url = self.get_base_url() + self._http_client.paths['experiment'](ex)
            return self._http_client.do_delete(url, **{})

    def get_experiment(self, ex):
        """Downloads the experiment given by ex, where ex is a string with the id or url of the
        experiment.

        Returns experiment with metadata, configuration and dataset downloaded from server while
        experiment creator is used to create test pipelines for workflow and preprocessing using experiment
        configuration.

        :param ex: Id or url of the experiment
        :returns: Returns experiment instance or none
        :rtype: <class 'padre.core.experiment.Experiment'> or None
        Todo:  Implement according to the following description.
                The experiment is downloaded from the server and stored in the local file store if the
                server version is newer than the local version or no local version exists.
                The function returns an experiment class, which is loaded from file.
        """
        experiment = None
        if self._http_client.online:
            if self._http_client.is_valid_url(ex):  # url of the experiment
                url = ex
                response = json.loads(self._http_client.do_get(url, **{}).content)
            elif ex.isdigit():
                url = self._http_client.get_base_url() + self._http_client.paths['experiment'](ex)
                response = json.loads(self._http_client.do_get(url, **{}).content)
            else:
                url = self._http_client.get_base_url() + self._http_client.paths['search']("experiments") + "name:" + ex
                response = json.loads(self._http_client.do_get(url, **{}).content)
                if "_embedded" in response:
                    response = response["_embedded"]["experiments"][0]  # Get first experiment
                else:
                    trigger_event('EVENT_ERROR', condition=False, source=self,
                                  message="Experiment with name {%s} not found  " % ex)
                    return experiment

            ds = self._http_client.datasets.get(str(response["dataset"]["uid"]))
            configuration = response["configuration"]
            keys = list(configuration.keys())
            if len(keys) > 0:  # If configuration not empty
                name = keys[0]
                conf = copy.deepcopy(configuration[name])
                experiment_creator = experimentcreator.ExperimentCreator()
                conf["dataset"] = ds
                conf["workflow"] = experiment_creator.create_test_pipeline(conf["workflow"])
                conf["preprocessing"] = experiment_creator.create_test_pipeline(conf["preprocessing"])
                experiment = Experiment(ex_id=conf["name"], **conf)
                experiment.metadata = response["metadata"]
                experiment.experiment_configuration = configuration
        return experiment

    def list_experiments(self, search=None, search_metadata=None, start=-1, count=999999999):
        """List of experiments from server.

        If search string is provided then search based on experiment name
        otherwise get list of all experiments

        :param search: Name of experiment
        :type search: str
        :param start: start index of sublist
        :param count: end index of sublist
        :returns: list of experiments containing experiment names
        :rtype: list

        Todo: We will later define a synatx to search also associated metadata (e.g. "description:search_string").

        """
        experiments = []
        start = max(start, 0)
        if search is not None:
            url = self.get_base_url() + self._http_client.paths["search"]("experiments") + "name?:" + search + "&size=" + str(count)
        else:
            url = self.get_base_url() + self._http_client.paths["experiments"] + "?size=" + str(count)
        if self._http_client.online:
            response = json.loads(self._http_client.do_get(url, **{}).content)
            if "_embedded" in response:
                experiments = response["_embedded"]["experiments"]
                experiments = [ex["name"] for ex in experiments]
        if start < len(experiments):
            experiments = experiments[start:]
        return experiments

    def put_run(self, experiment, run):
        """
        Put run information on server and also upload workflow for this new run on the server as binary.

        :param experiment: Experiment instance
        :type experiment: <class 'pypadre.core.experiment.Experiment'>
        :param run: Run instance
        :type run: <class 'pypadre.core.run.Run'>
        :return: Return url of run at the server.
        """
        location = ""
        experiment_id = experiment.metadata["server_url"].split("/")[-1]
        run_data = dict()
        run_data["clientAddress"] = self.get_base_url()
        run_data["uid"] = str(run.id)
        run_data["metadata"] = run.metadata
        run_data["hyperparameterValues"] = [{"component":
            {"description": experiment.metadata["description"],
             "hyperparameters": self.build_hyperparameters_list(experiment.hyperparameters()),
             "name": experiment.metadata["name"]
             }
        }]
        run_data["experimentId"] = experiment_id
        url = self.get_base_url() + self._http_client.paths["runs"]
        if self._http_client.online:
            response = self._http_client.do_post(url, **{"data": json.dumps(run_data)})
            location = response.headers["location"]
            run_id = location.split("/")[-1]
            run_model_url = self.get_base_url() + self._http_client.paths["run-models"](experiment_id, run_id)
            binary = self._binary_serializer.serialise(experiment._workflow)
            data = MultipartEncoder(fields={
                "file": ("fname",
                           io.BytesIO(binary),
                           "application/octet-stream")})
            headers = {"Content-Type": data.content_type}
            self._http_client.do_post(run_model_url, **{"data": data, "headers": headers})
            run.metadata["server_url"] = location
            self._http_client.do_patch(location, **{"data": json.dumps({"metadata": run.metadata})})

        return location

    def get_run(self, ex_id, run_id):
        """
        Get run from server including workflow.

        :param ex_id: Id of the experiment
        :param run_id: Id of the run
        :returns: Return run instance or None if client is not online
        :rtype: <class 'pypadre.core.run.Run'>

        Todo: Use run metadata instead of experiment metadata as run is identified by unique server_url
        """
        run_url = self._http_client.get_base_url() + self._http_client.paths['run'](ex_id, run_id)
        run_model_url = self._http_client.get_base_url() + self._http_client.paths["run-models"](ex_id, run_id)
        r = None
        if self._http_client.online:
            run_response = json.loads(self._http_client.do_get(run_url, **{}).content)
            model_response = self._http_client.do_get(run_model_url, **{})
            workflow = self._binary_serializer.deserialize(model_response.content)
            ex = self.get_experiment(ex_id)
            r = Run(ex, workflow, run_id=run_id, **ex.metadata)
        return r

    def get_experiment_run_idx(self, ex_id):
        """
        Get run ids for the given experiment id
        :param ex_id: Experiment id for which run ids should be returned
        :type ex_id: str
        :return: list of run ids for the given experiment
        """
        run_ids = []
        url = self._http_client.get_base_url() + self._http_client.paths['experiment-runs'](ex_id)
        if self._http_client.online:
            response = json.loads(self._http_client.do_get(url, **{}).content)
            if "_embedded" in response:
                for run in response["_embedded"]["runs"]:
                    run_ids.append(run["uid"])
        return run_ids

    def put_split(self, experiment, run, split):
        """
        Put split information on the server.

        Dataset splits of indices will be encoded before upload

        :param experiment:
        :type experiment: <class 'pypadre.core.experiment.Experiment'>
        :param run:
        :type run: <class 'pypadre.core.run.Run'>
        :param split:
        :type split: <class 'pypadre.core.split.Split'>
        :return: Return url of run-split at server
        """
        location = ""
        data = dict()
        r_id = run.metadata["server_url"].split("/")[-1]
        url = self.get_base_url() + self._http_client.paths["splits"]
        data["uid"] = str(split.id)
        data["clientAddress"] = self.get_base_url()
        data["runId"] = r_id
        data["split"] = self.encode_split(split)
        data["metadata"] = split.metadata
        data["metrics"] = {}
        if self._http_client.online:
            response = self._http_client.do_post(url, **{"data": json.dumps(data)})
            location = response.headers["location"]
            split.metadata["server_url"] = location
            self._http_client.do_patch(location, **{"data": json.dumps({"metadata": split.metadata})})
        split.metadata["server_url"] = location
        return location

    def get_split(self, ex_id, run_id, split_id):
        """
        Get split from the server with metadata, metrics and results.
        Dataset splits will be decoded into list of indices.

        :param ex_id: Experiment id
        :param run_id: Run id
        :param split_id: Split id
        :return: Split instance or None if http client is not online
        :rtype: <class 'pypadre.core.split.Split'>
        """
        s = None
        split_url = self.get_base_url() + self._http_client.paths["split"](split_id)
        if self._http_client.online:
            split_response = json.loads(self._http_client.do_get(split_url, **{}).content)
            decode_split = self.decode_split(split_response["split"])
            r = self.get_run(ex_id, run_id)
            s = Split(
                r, split_response["splitNum"], decode_split["train"], decode_split["val"], decode_split["test"],
                split_id=split_response["uid"], **r.metadata)

            s.run.metrics.append(split_response["metrics"])
            s.run.results.append(self.get_results(r.experiment, r, s, split_response))
        return s

    def get_run_split_idx(self, ex_id, run_id):
        """
        Get split ids for the given run and experiment ids

        :param ex_id: Experiment id for which split ids should be returned
        :type ex_id: str
        :param run_id: Run id for which split ids should be returned
        :type run_id: str
        :return: list of split ids for the given run and experiment
        """
        split_ids = []
        url = self._http_client.get_base_url() + self._http_client.paths['experiment-run-splits'](ex_id, run_id)
        if self._http_client.online:
            response = json.loads(self._http_client.do_get(url, **{}).content)
            if "_embedded" in response:
                for split in response["_embedded"]["runSplits"]:
                    split_ids.append(split["uid"])
        return split_ids

    def put_results(self, experiment, run, split, results):
        """
        Write results on temporary file-like object as protobuf and put it to the server.

        :param experiment:
        :type experiment: <class 'pypadre.experiment.Experiment'>
        :param run:
        :type run: <class 'pypadre.experiment.Run'>
        :param split:
        :type split: <class 'pypadre.experiment.Split'>
        :param results:
        :type results: <class 'dict'>
        :return: Returns http response
        :rtype: <class 'requests.models.Response'>
        """
        rs_id = split.metadata["server_url"].split("/")[-1]
        r_id = run.metadata["server_url"].split("/")[-1]
        e_id = experiment.metadata["server_url"].split("/")[-1]
        url = self.get_base_url() + self._http_client.paths["results"](e_id, r_id, rs_id)
        update_split_url = self.get_base_url() + self._http_client.paths["split"](rs_id)
        response = None
        if bool(results) and self._http_client.online:
            with tempfile.TemporaryFile() as temp_file:
                file = self.make_proto(results, temp_file)
                m = MultipartEncoder(
                    fields={"field0": ("fname", file, self.get_content_type(results["type"]))})
                response = self._http_client.do_post(url, **{"data": m, "headers": {"Content-Type": m.content_type}})
            patch_data = {
                "splitNum": results["split_num"],
                "trainingSampleCount": results["training_sample_count"],
                "testingSampleCount": results["testing_sample_count"]
            }
            self._http_client.do_patch(update_split_url, **{"data": json.dumps(patch_data)})
        return response

    def get_results(self, experiment, run, split, split_server_data):
        """
        Get split results from server.

        Results are formatted in original after downloading them from server.
        Json end point for split results is used to get results from server.

        :param experiment: Experiment which is downloaded from server
        :type experiment: <class 'pypadre.experiment.Experiment'>
        :param run: Run which is downloaded from server
        :type run: <class 'pypadre.experiment.Run'>
        :param split: Split which is downloaded from server
        :type split: <class 'pypadre.experiment.Split'>
        :param split_server_data: Split data from server
        :type split_server_data: dict
        :return: Dictionary containing results
        :rtype: dict
        """
        results = dict()
        experiment_id = experiment.id
        if not experiment_id.isdigit():
            experiment_id = experiment.metadata["server_url"].split("/")[-1]
        experiment_type = split_server_data["metrics"]["type"]
        content_type = self.get_content_type(experiment_type)
        results_url = self.get_base_url() + self._http_client.paths["results-json"](experiment_id, run.id, split.id)
        results_response = json.loads(
            self._http_client.do_get(results_url, **{"params": [("format", content_type)]}).content)
        results["type"] = experiment_type
        results["dataset"] = split_server_data["metrics"]["dataset"]
        results["split_num"] = split_server_data["splitNum"]
        results["training_sample_count"] = split_server_data["trainingSampleCount"]
        results["testing_sample_count"] = split_server_data["testingSampleCount"]
        if split.train_idx is not None:
            results["train_idx"] = split.train_idx.tolist()
        if split.test_idx is not None:
            results["test_idx"] = split.test_idx.tolist()
        if split.val_idx is not None:
            results["val_idx"] = split.val_idx.tolist()
        results["truth"] = list()
        results["predicted"] = list()
        results["probabilities"] = list()
        results["predictions"] = dict()
        for entry in results_response["rows"]:
            prediction = dict()
            truth = float("".join(entry["data"]["truth"]))
            predicted = float("".join(entry["data"]["predictions"]))
            probabilities = list(map(float, entry["data"]["score"]))
            prediction["truth"] = truth
            prediction["predicted"] = predicted
            prediction["probabilities"] = probabilities
            results["predictions"][entry["index"]] = prediction
            results["truth"].append(truth)
            results["predicted"].append(predicted)
            results["probabilities"].append(probabilities)
        return results

    def put_metrics(self, experiment, run, split, metrics):
        rs_id = split.metadata["server_url"].split("/")[-1]
        update_split_url = self.get_base_url() + self._http_client.paths["split"](rs_id)
        response = None
        if self._http_client.online:
            response = self._http_client.do_patch(update_split_url,
                                                  **{"data": json.dumps({"metrics": metrics})})
        return response

    def log(self, message):
        pass

    def log_experiment_progress(self, curr_value, limit, phase):
        """

        :param curr_value:
        :param limit:
        :param phase:
        :return:
        """
        pass

    def log_run_progress(self, curr_value, limit, phase):
        """

        :param curr_value:
        :param limit:
        :param phase:
        :return:
        """
        pass

    def log_split_progress(self, curr_value, limit, phase):
        """

        :param curr_value:
        :param limit:
        :param phase:
        :return:
        """
        pass

    def log_progress(self, message, curr_value, limit, phase):
        """

        :param message:
        :param curr_value:
        :param limit:
        :param phase:
        :return:
        """
        pass

    def log_end_experiment(self):
        pass

    def log_model(self, model, framework, modelname, finalmodel=False):
        """
        Logs an intermediate model to the backend
        :param model: Model to be logged
        :param framework: Framework of the model
        :param modelname: Name of the intermediate model
        :param finalmodel: Boolean value indicating whether the model is the final one or not
        :return:
        """
        pass

    def get_base_url(self):
        url = self._http_client.base
        if url[-1] == "/":
            url = url[0:-1]
        return url

    def get_id(self, http_response):
        return http_response.headers['Location'].split('/')[-1]

    def build_hyperparameters_list(self, obj):
        """
        Build a list of formatted hyperparamters as a dict for each parameter type.
        Passed obj dict can be as
        {"Step_0": {"hyper_parameters": {"models_parameters": ..., "optimisation_parameters": ...}}}

        :param obj: Dict containing experiment.hyperparamters()
        :type obj: dict
        :return: List of formatted hyperparameters
        :rtype: list
        """
        hyperparameters_list = []
        for k, v in obj.items():
            params = v["hyper_parameters"]
            for param_kind, attr_dict in params.items():
                for attr_name in attr_dict.keys():
                    data = dict()
                    data["description"] = attr_name
                    data["kind"] = self.map_to_parameter_kind(param_kind)
                    data["url"] = "dummy-padre.com/"
                    data["type"] = "RealNumber"
                    hyperparameters_list.append(data)
        return hyperparameters_list

    def map_to_parameter_kind(self, param):
        """
        Convert hyperparameter kind to compatible ParameterKind on server.

        :param param: hyperparameter kind
        :type param: str
        :return: Parameter type
        :rtype: str
        """
        params = {
            "model_parameters": "ModelParameter",
            "optimisation_parameters": "OptimizationParameter",
            "runtime_training_parameters": "RuntimeTrainingParameter",
            "runtime_testing_parameters": "RuntimeTestingParameter",
            "execution_parameters": "ExecutionParameter"
        }
        return params[param]

    def make_proto(self, results, file):
        """
        Make compatible protobuf object from results and write it on given file object

        :param results: Dictionary containing results of run-split
        :type results: <class 'dict'>
        :param file: file like object
        :type file: <class '_io.BufferedRandom'>
        :return: Protobuf written on file like object
        :rtype: <class '_io.BufferedRandom'>
        """
        for i, test_id in enumerate(results["test_idx"]):
            pb_meta = proto.ResultEntry()
            pb_meta.index = test_id
            result_prediction = results["predicted"][i]
            result_truth = results["truth"][i]
            self.add_value(pb_meta.prediction.add(), result_prediction)
            self.add_value(pb_meta.truth.add(), result_truth)

            if "probabilities" in results:
                result_probabilities = results["probabilities"][i]
                for p in result_probabilities:
                    self.add_value(pb_meta.score.add(), p)

            serialize = pb_meta.SerializeToString()
            file.write(_VarintBytes(len(serialize)))
            file.write(serialize)
            file.flush()
        file.seek(0)
        return file

    def add_value(self, pb_instance, value):
        """
        Add float or int attribute in the protobuf Value

        :param pb_instance: protobuf instance of type Value
        :type pb_instance: <class 'resultV1_pb2.Value'>
        :param value: Single value of each parameter
        :type value: <class 'float'> or <class 'int'> etc
        :return: None
        """
        if type(value) == float:
            pb_instance.double_t = value
        elif type(value) == int:
            pb_instance.int32_t = value

    def get_content_type(self, experiment_type):
        """
        Get compatible content type for results to upload

        :param experiment_type: regression, classification, transformation or dataset
        :type experiment_type: str
        :return: Content Type
        :rtype: str
        """
        if experiment_type == "regression":
            return "application/x.padre.regression.v1+protobuf"
        elif experiment_type == "classification":
            return "application/x.padre.classification.v1+protobuf"

    def encode_split(self, split):
        """Encode the train, test and validation sets into boolean representation of run length encoding.

        split.train_idx, split.test_idx and split.val_idx is np array or None

        :param split: Split instance
        :type split: <class 'pypadre.experiment.Split'>
        :returns: String as run length encoding
        """
        train_idx = split.train_idx
        test_idx  = split.test_idx
        val_idx = split.val_idx

        result = "train:"
        if train_idx is not None and train_idx.size > 0:
            train_bool_list = [False] * (np.amax(train_idx) + 1)
            for x in train_idx:
                train_bool_list[x] = True

            for b, g in groupby(train_bool_list):
                l = str(len(list(g)))
                if b:
                    result += "t" + l
                else:
                    result += "f" + l

        result += ",test:"
        if test_idx is not None and test_idx.size > 0:
            test_bool_list = [False] * (np.amax(test_idx) + 1)
            for x in test_idx:
                test_bool_list[x] = True

            for b, g in groupby(test_bool_list):
                l = str(len(list(g)))
                if b:
                    result += "t" + l
                else:
                    result += "f" + l
        result += ",val:"
        if val_idx is not None and val_idx.size > 0:
            val_bool_list = [False] * (np.amax(val_idx) + 1)
            for x in val_idx:
                val_bool_list[x] = True

            for b, g in groupby(val_bool_list):
                l = str(len(list(g)))
                if b:
                    result += "t" + l
                else:
                    result += "f" + l

        return result

    def decode_split(self, encoded_split):
        """
        Decode encoded split into list of indices for train, test and validation data

        :param encoded_split: Encoded string for example train:f1t1,test:f2t1,val:f3t1
        :type encoded_split: str
        :return: Dictionary containing decoded lists for each train, test and val sets
        """
        result = dict()
        pattern = re.compile(r"f\d{1,}|t\d{1,}")
        for section in encoded_split.split(","):
            name, value = section.split(":")
            res = self.decode_list(pattern.findall(value))
            result[name] = res
        return result

    def decode_list(self, list_of_encodings):
        """
        Decode list of encodings into list of indices

        :param list_of_encodings: for example: ["f2", "t1"]
        :return: Numpy array of dataset indices or None
        """
        result = []
        counter = 0
        for encoding in list_of_encodings:
            if encoding[0] == "f":
                counter += int(encoding[1:])
            if encoding[0] == "t":
                item = int(encoding[1:])
                result = result + list(range(counter, counter + item))
                counter += item
        if not result:
            return None
        return np.array(result)

    def put_experiment_configuration(self, experiment):
        """
        Writes the experiment configuration to the server.
        Also write experiment metadata to server which has updated server_url.

        :param experiment: Experiment to be written
        :return:
        """
        response = None
        data = dict()
        e_id = experiment.metadata["server_url"].split("/")[-1]
        experiment_url = self.get_base_url() + self._http_client.paths["experiment"](e_id)
        data["configuration"] = experiment.experiment_configuration
        if self._http_client.online:
            response = self._http_client.do_patch(experiment_url,
                                                  **{"data": json.dumps(data)})
        return response

    def validate_and_save(self, experiment, run=None, split=None, local_experiments=None):
        """
        Upload only new experiment, run or split to server.

        Upload only those experiment, run or split to server which are not already uploaded.
        Criteria to check for it is, if server_url attribute in metadata is empty then it means
        this experiment(or run or split) does not exists on the server. After uploading them
        update its server_url in metadata
        TODO: Check if experiment, run or split can downloaded from one server and uploaded to other

        :param put_fn: Callable function from http backend which can be either put_experiment,
            put_run or put_split.
        :type put_fn: <class 'method'>
        :param experiment: Experiment to be uploaded
        :type experiment: <class 'pypadre.core.experiment.Experiment'>
        :param run: Run to be uploaded
        :type run: <class 'pypadre.core.run.Run'>
        :param split: Split to be uploaded
        :type split: <class 'pypadre.core.split.Split'>
        :param local_experiments: Instance of local backend experiments
        :type local_experiments: <class 'pypadre.backend.file.ExperimentFileRepository'>
        :return: Boolean whether experiment, run or split is uploaded or not
        """
        server_url = ""
        if split is not None:
            if split.metadata["server_url"].strip() == "":
                server_url = self.put_split(experiment, run, split)
                if local_experiments is not None:
                    local_experiments.update_metadata({"server_url": server_url},
                                                       experiment.id,
                                                       run.id,
                                                       split.id)
        elif run is not None:
            if run.metadata["server_url"].strip() == "":
                server_url = self.put_run(experiment, run)
                if local_experiments is not None:
                    local_experiments.update_metadata({"server_url": server_url},
                                                       experiment.id, run.id)
        else:
            if experiment.metadata["server_url"].strip() == "":
                server_url = self.put_experiment(experiment)
                if local_experiments is not None:
                    local_experiments.update_metadata({"server_url": server_url},
                                                       experiment.id)
        return server_url


