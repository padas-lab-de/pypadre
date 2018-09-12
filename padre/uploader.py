"""
Logic to handle tokens and upload data to server goes here
"""
import json
import requests
from padre.constants import PADRE_API, HOST, USERNAME, PASSWORD


class Uploader:
    """Uploader helper to upload data to server"""
    def __init__(self, project_name):
        """
        This initializes the Uploader class.
        """
        self._csrf_token = self.get_csrf_token()
        self._bearer_token = self.get_access_token()

        self._headers = {
            'Authorization': self._bearer_token,
            'Content-Type': "application/json"
        }
        self.dataset_id = None
        self.project_id = None
        self.create_project(project_name)

    def get_csrf_token(self):
        """Get csrf token"""
        token = requests.get(PADRE_API).cookies.get("XSRF-TOKEN")
        return token

    def get_access_token(self):
        """Get access token"""
        data = {
            "username": USERNAME,
            "password": PASSWORD,
            "grant_type": "password"
        }
        url = PADRE_API + "/oauth/token?=" + self._csrf_token
        response = requests.post(url, data=data)
        token = "Bearer " + json.loads(response.content)['access_token']
        return token

    def create_dataset(self, data):
        """Create data set"""
        url = HOST + "/api/datasets"
        if isinstance(data, dict):
            data = json.dumps(data)
        response = requests.post(url, data=data, headers=self._headers)
        self.dataset_id = response.headers['Location'].split('/')[-1]

    def create_project(self, name):
        """Create project on server"""
        url = HOST + "/api/projects"
        data = {"name": name, "owner": USERNAME}
        response = requests.post(url, data=json.dumps(data), headers=self._headers)
        self.project_id = response.headers['Location'].split('/')[-1]

    def create_experiment(self, data):
        """Create experiment on server"""
        url = HOST + "/api/experiments"
        if isinstance(data, dict):
            data = json.dumps(data)
        return requests.post(url, data=data, headers=self._headers)

    def upload_experiment(self, experiment):
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

        self.create_experiment(experiment_data)



