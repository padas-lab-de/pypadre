"""
Logic to upload experiment data to server goes here
"""
import json


class ExperimentUploader:
    """Experiment uploader to upload data to server"""
    def __init__(self, http_client, project_name="Test project"):
        """
        This initializes the Uploader class for given experiment.
        """
        self._http_client = http_client
        self.dataset_id = None
        self.experiment_id = None
        self.project_id = None
        self.create_project(project_name)

    def create_dataset(self, data):
        """Create data set"""
        url = self._http_client.base + self._http_client.paths["datasets"]
        if isinstance(data, dict):
            data = json.dumps(data)
        response = self._http_client.do_post(url, **{"data": data})
        self.dataset_id = response.headers['Location'].split('/')[-1]

    def create_project(self, name="Test Project"):
        """Create project on server"""
        url = self._http_client.base + self._http_client.paths["projects"]
        data = {"name": name, "owner": self._http_client.user}
        response = self._http_client.do_post(url, **{"data": json.dumps(data)})
        self.project_id = response.headers['Location'].split('/')[-1]

    def create_experiment(self, data):
        """Create experiment on server"""
        url = self._http_client.base + self._http_client.paths["experiments"]
        if isinstance(data, dict):
            data = json.dumps(data)
        response = self._http_client.do_post(url, **{"data": data})
        self.experiment_id = response.headers['Location'].split('/')[-1]
        return response.headers['Location']

    def put_experiment(self, experiment):
        """
        Upload experiment to server
        :param experiment: Experiment instance
        :type experiment: <class 'padre.experiment.Experiment'>
        :return: None
        """
        dataset_dict = experiment.dataset.metadata
        self.create_dataset(dataset_dict)

        experiment_data = experiment.metadata
        experiment_data["projectId"] = self.project_id
        experiment_data["datasetId"] = self.dataset_id
        experiment_data["pipeline"] = {"components": [
            {"description": "",
             "hyperparameters": [experiment.hyperparameters()]}
        ]}

        return self.create_experiment(experiment_data)




