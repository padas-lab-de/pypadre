"""
Logic to upload experiment data to server goes here
"""
import io
import json
import logging
import tempfile
import uuid
from itertools import groupby

import requests as req
from requests_toolbelt import MultipartEncoder
from google.protobuf.internal.encoder import _VarintBytes
from padre.backend.protobuffer.protobuf import resultV1_pb2 as proto

from padre import experimentcreator
from padre.backend.serialiser import PickleSerializer

logger = logging.getLogger('pypadre - http')


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
        id_ = self.get_id_by_name(name, self._http_client.paths["projects"])
        if id_ is None:
            id_ = self.create_project(name)
        return id_

    def get_or_create_dataset(self, ds):
        _id = ds.metadata.get("uid", None)
        get_url = self._http_client.get_base_url() + self._http_client.paths["dataset"](str(_id))
        dataset_id = None
        if self._http_client.has_token():
            try:
                response = self._http_client.do_get(get_url)
                dataset_id = json.loads(response.content)["uid"]
            except req.HTTPError as e:
                logger.warn("Dataset with id {%s} not found  " % str(_id))
                dataset_id = self._http_client.datasets.put(ds)
        return dataset_id

    def get_id_by_name(self, name, entity):
        """Get entity id by name

        :param name: Instance name of the entity to be searched on server
        :param entity: Name of entity e-g /projects, /datasets
        :returns: id of instance or None
        """
        id_ = None
        url = self.get_base_url() + entity + "?name=" + name
        if self._http_client.has_token():
            response = json.loads(self._http_client.do_get(url, **{}).content)
            if "_embedded" in response:
                id_ = response["_embedded"][entity[1:]][0]["uid"]
        return id_

    def create_project(self, name):
        """Create project on server

        :param name: Name of the project
        :returns: Id of the instance or None
        """
        url = self.get_base_url() + self._http_client.paths["projects"]
        data = {"name": name, "owner": self._http_client.user}
        if self._http_client.has_token():
            response = self._http_client.do_post(url, **{"data": json.dumps(data)})
            return response.headers['Location'].split('/')[-1]
        return None

    def create_experiment(self, data):
        """Create experiment on server"""
        url = self.get_base_url() + self._http_client.paths["experiments"]
        location = ''
        if isinstance(data, dict):
            data = json.dumps(data)
        if self._http_client.has_token():
            response = self._http_client.do_post(url, **{"data": data})
            self.experiment_id = response.headers['Location'].split('/')[-1]
            location = response.headers['Location']
        return location

    def put_experiment(self, experiment, append_runs=None):
        """
        Upload experiment to server
        :param experiment: Experiment instance
        :type experiment: <class 'padre.experiment.Experiment'>
        :return: None
        """

        dataset_dict = experiment.dataset
        self.dataset_id = self.get_or_create_dataset(dataset_dict)
        experiment_data = dict()
        experiment_data["name"] = experiment.metadata["name"]
        experiment_data["description"] = experiment.metadata["description"]
        experiment_data["algorithm"] = "http://padre.de/algorithm/1"
        experiment_data["executable"] = "http://padre.de/executable/1"
        experiment_data["sourceCode"] = "http://padre.de/executable/1"
        experiment_data["published"] = True
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
        if ex.isdigit() and self._http_client.has_token():
            url = self.get_base_url() + self._http_client.paths['experiment'](ex)
            return self._http_client.do_delete(url, **{})

    def get_experiment(self, ex):
        """
        Downloads the experiment given by ex, where ex is a string with the id or url of the
        experiment. The experiment is downloaded from the server and stored in the local file store
        if the server version is newer than the local version or no local version exists.
        The function returns an experiment class, which is loaded from file.

        :param ex: Id or url of the experiment
        :return:
        # todo: Return experiment instance according to above documentation
        """
        if self._http_client.has_token():
            if not ex.isdigit():  # url of the experiment
                url = self._http_client.base + ex
            else:
                url = self.get_base_url() + self._http_client.paths['experiment'](ex)
            response = json.loads(self._http_client.do_get(url, **{}).content)
            conf = response[list(response.keys())[0]]
            experiment_creator = experimentcreator.ExperimentCreator()
            experiment_creator.create_experiment(conf["name"],
                                                 conf["description"],
                                                 [conf["dataset"]],
                                                 conf["workflow"],
                                                 conf["params"])

            return experiment_creator
        return False

    def put_run(self, experiment, run):
        """
        Put run information on server and also upload workflow for this new run on the server as binary.

        :param experiment:
        :type experiment: <class 'padre.experiment.Experiment'>
        :param run:
        :type run: <class 'padre.experiment.Run'>
        :return: Return url of run at the server.
        """
        location = ""
        experiment_id = experiment.metadata["server_url"].split("/")[-1]
        run_data = dict()
        run_data["clientAddress"] = self.get_base_url()
        run_data["uid"] = str(uuid.uuid4())
        run_data["hyperparameterValues"] = [{"component":
            {"description": experiment.metadata["description"],
             "hyperparameters": self.build_hyperparameters_list(experiment.hyperparameters()),
             "name": experiment.metadata["name"]
             }
        }]
        run_data["experimentId"] = experiment_id
        url = self.get_base_url() + self._http_client.paths["runs"]
        if self._http_client.has_token():
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
        return location

    def put_split(self, experiment, run, split):
        """
        Put split information on the server.

        :param experiment:
        :type experiment: <class 'padre.experiment.Experiment'>
        :param run:
        :type run: <class 'padre.experiment.Run'>
        :param split:
        :type split: <class 'padre.experiment.Split'>
        :return: Return url of run-split at server
        """
        location = ""
        data = dict()
        r_id = run.metadata["server_url"].split("/")[-1]
        url = self.get_base_url() + self._http_client.paths["splits"]
        data["uid"] = str(uuid.uuid4())
        data["clientAddress"] = self.get_base_url()
        data["runId"] = r_id
        data["split"] = self.encode_split(split)
        data["metrics"] = {}
        if self._http_client.has_token():
            response = self._http_client.do_post(url, **{"data": json.dumps(data)})
            location = response.headers["location"]
        split.metadata["server_url"] = location
        return location

    def put_results(self, experiment, run, split, results):
        """
        Write results on temporary file-like object as protobuf and put it to the server.

        :param experiment:
        :type experiment: <class 'padre.experiment.Experiment'>
        :param run:
        :type run: <class 'padre.experiment.Run'>
        :param split:
        :type split: <class 'padre.experiment.Split'>
        :param results:
        :type results: <class 'dict'>
        :return: Returns http response
        :rtype: <class 'requests.models.Response'>
        """
        rs_id = split.metadata["server_url"].split("/")[-1]
        r_id = run.metadata["server_url"].split("/")[-1]
        e_id = experiment.metadata["server_url"].split("/")[-1]
        url = self.get_base_url() + self._http_client.paths["results"](e_id, r_id, rs_id)
        response = None
        if bool(results) and self._http_client.has_token():
            with tempfile.TemporaryFile() as temp_file:
                file = self.make_proto(results, temp_file)
                m = MultipartEncoder(
                    fields={"field0": ("fname", file, self.get_content_type(results["type"]))})
                response = self._http_client.do_post(url, **{"data": m, "headers": {"Content-Type": m.content_type}})
        return response

    def put_metrics(self, experiment, run, split, metrics):
        rs_id = split.metadata["server_url"].split("/")[-1]
        update_split_url = self.get_base_url() + self._http_client.paths["split"](rs_id)
        response = None
        if self._http_client.has_token():
            response = self._http_client.do_patch(update_split_url,
                                                  **{"data": json.dumps({"metrics": metrics})})
        return response

    def log(self, message):
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
            pb_instance.float_t = value
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
        """Encode the train and test sets into boolean representation of run length encoding.

        :param split: Split instance
        :type split: <class 'padre.experiment.Split'>
        :returns: String as run length encoding
        """
        train_idx = split.train_idx
        test_idx  = split.test_idx

        train_idx.sort()
        test_idx.sort()

        train_bool_list = [False] * (train_idx[-1] + 1)
        for x in train_idx:
            train_bool_list[x] = True

        result = "train:"
        for b, g in groupby(train_bool_list):
            l = str(len(list(g)))
            if b:
                result += "t" + l
            else:
                result += "f" + l

        test_bool_list = [False] * (test_idx[-1] + 1)
        for x in test_idx:
            test_bool_list[x] = True

        result += ",test:"
        for b, g in groupby(test_bool_list):
            l = str(len(list(g)))
            if b:
                result += "t" + l
            else:
                result += "f" + l

        return result




    def put_experiment_configuration(self, experiment):
        """
        Writes the experiment configuration to the HTTP Client
        :param experiment: Experiment to be written
        :return:
        """
        pass


