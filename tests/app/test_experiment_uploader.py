"""
This file contains tests covering backend.experiment_uploader.ExperimentUploader class
All unnecessary function and http calls are mocked
"""
import json
import unittest

from mock import MagicMock, patch
from padre.backend.experiment_uploader import ExperimentUploader


class TestCreateDataSet(unittest.TestCase):
    """Test experiment_uploader.ExperimentUploader.create_dataset

    All unnecessary function call and http calls are mocked
    """
    def setUp(self):
        """Initializing for create_dataset test.

        All non related function calls and http calls will be mocked for this purpose.
        """
        self.test_dataset_id = '2'
        self.test_dataset_data = {'name': 'test data', 'description': 'test description'}
        self.http_client = MagicMock()
        self.http_client.has_token = MagicMock(return_value=True)
        mocked_post_response = MagicMock()
        mocked_post_response.headers = {'Location': 'api/datasets/' + self.test_dataset_id}
        self.http_client.do_post = MagicMock(return_value=mocked_post_response)


    @patch('padre.backend.experiment_uploader.ExperimentUploader.create_project')
    def test_create_dataset_01(self, mock_project):
        """Test ExperimentUploader.create_dataset function.

        Scenario: Correct id is set for data set.
        """
        obj = ExperimentUploader(self.http_client)
        mock_project.return_value = True
        obj.create_dataset(self.test_dataset_data)
        self.assertEqual(self.test_dataset_id,
                         obj.dataset_id,
                         'Data set not created successfully')

    @patch('padre.backend.experiment_uploader.ExperimentUploader.create_project')
    def test_create_dataset_02(self, mock_project):
        """Test ExperimentUploader.create_dataset function.

        Scenario: do_post called with correct args.
        """
        obj = ExperimentUploader(self.http_client)
        mock_project.return_value = True
        obj.create_dataset(self.test_dataset_data)
        self.assertEqual(self.test_dataset_data,
                         json.loads(self.http_client.do_post.call_args_list[0][1]['data']),
                         'Do post not called with expected data for create_dataset')

    def tearDown(self):
        pass


class TestPutExperiment(unittest.TestCase):
    """Test experiment_uploader.ExperimentUploader.put_experiment

    All unnecessary function calls and http calls are mocked
    """

    def setUp(self):
        """Initializing for create_dataset test.

        All non related function calls and http calls will be mocked for this purpose.
        """
        self.test_experiment_url = 'api/experiments/' + '3'
        self.http_client = MagicMock()
        self.http_client.user = 'test user'
        self.http_client.has_token = MagicMock(return_value=True)

        mocked_post_project = MagicMock()
        mocked_post_project.headers = {'Location': 'api/projects/' + '1'}
        mocked_post_dataset = MagicMock()
        mocked_post_dataset.headers = {'Location': 'api/datasets/' + '2'}
        mocked_post_experiment = MagicMock()
        mocked_post_experiment.headers = {'Location': self.test_experiment_url}

        self.http_client.do_post = MagicMock()
        self.http_client.do_post.side_effect = [
            mocked_post_project,
            mocked_post_dataset,
            mocked_post_experiment]

    def test_put_experiment(self):
        """Test ExperimentUploader.put_experiment function.

        Scenario: Put experiment should return url of newly created experiment.
        """
        obj = ExperimentUploader(self.http_client)
        ex = MagicMock()
        ex.dataset = MagicMock()
        ex.dataset.metadata = {'name': 'test name'}
        ex.metadata = {'name': 'test experiment'}
        ex.hyperparameters = MagicMock(return_value={'param1': 'value1'})

        result = obj.put_experiment(ex)
        self.assertEqual(self.test_experiment_url,
                         result,
                         'Put experiment does not return url of newly created experiment')

    def tearDown(self):
        pass