"""
This file contains tests covering backend.http_experiments.HttpBackendExperiments.validate_and_save function
"""
import unittest

from mock import MagicMock, call

from padre.backend.http_experiments import HttpBackendExperiments


class ValidateAndSave(unittest.TestCase):
    """Test HttpBackendExperiments.validate_and_save function"""
    def setUp(self):
        """Create mock objects"""
        self.returned_url = "http://test.com/api/object/id"
        self.http_client = MagicMock()
        self.http_client.has_token = MagicMock(return_value=False)
        self._remote_experiments = HttpBackendExperiments(self.http_client)
        self._remote_experiments.put_experiment = MagicMock(return_value=self.returned_url)
        self._remote_experiments.put_run = MagicMock(return_value=self.returned_url)
        self._remote_experiments.put_split = MagicMock(return_value=self.returned_url)
        self._local_experiments = MagicMock()
        self._local_experiments.update_metadata = MagicMock()

    def test_validate_and_save_01(self):
        """test HttpBackendExperiments.validate_and_save function for experiment

        Scenario: Upload experiment which is already uploaded
        """
        experiment = MagicMock()
        experiment.metadata = {"server_url": "http://test.com"}
        response = self._remote_experiments.validate_and_save(experiment, local_experiments=self._local_experiments)
        self.assertFalse(response, "Experiment which already exists on server is upload")

    def test_validate_and_save_02(self):
        """test HttpBackendExperiments.validate_and_save function for experiment

        Scenario: Upload experiment which is not already uploaded
        """

        experiment = MagicMock()
        experiment.metadata = {"server_url": ""}
        experiment_id = "3"
        experiment.id = experiment_id
        response = self._remote_experiments.validate_and_save(experiment, local_experiments=self._local_experiments)
        self.assertTrue(response, "Experiment which not exists on server is not upload")
        expected = [call({"server_url": self.returned_url}, experiment_id)]
        self.assertEqual(expected,
                         self._local_experiments.update_metadata.call_args_list,
                         "Update metadata not called with expected arguments")

    def test_validate_and_save_03(self):
        """test HttpBackendExperiments.validate_and_save function for run

        Scenario: Upload run which is already uploaded
        """
        experiment = MagicMock()
        run = MagicMock()
        run.metadata = {"server_url": "http://testrun.com"}
        response = self._remote_experiments.validate_and_save(
            experiment, run, local_experiments=self._local_experiments)
        self.assertFalse(response, "Run which already exists on server is upload")

    def test_validate_and_save_04(self):
        """test HttpBackendExperiments.validate_and_save function for run

        Scenario: Upload run which is not already uploaded
        """

        experiment = MagicMock()
        experiment.metadata = {"server_url": "test.com"}
        experiment_id = "3"
        experiment.id = experiment_id
        run = MagicMock()
        run.metadata = {"server_url": ""}
        run_id = "4"
        run.id = run_id
        response = self._remote_experiments.validate_and_save(
            experiment, run, local_experiments=self._local_experiments)
        self.assertTrue(response, "Run which not exists on server is not upload")
        expected = [call({"server_url": self.returned_url}, experiment_id, run_id)]
        self.assertEqual(expected,
                         self._local_experiments.update_metadata.call_args_list,
                         "Update metadata not called with expected arguments for run")

    def test_validate_and_save_05(self):
        """test HttpBackendExperiments.validate_and_save function for split

        Scenario: Upload split which is already uploaded
        """
        experiment = MagicMock()
        run = MagicMock()
        split = MagicMock()
        split.metadata = {"server_url": "http://testsplit.com"}
        response = self._remote_experiments.validate_and_save(
            experiment, run, split, local_experiments=self._local_experiments)
        self.assertFalse(response, "Run which already exists on server is upload")

    def test_validate_and_save_06(self):
        """test HttpBackendExperiments.validate_and_save function for split

        Scenario: Upload split which is not already uploaded
        """

        experiment = MagicMock()
        experiment.metadata = {"server_url": "test.com"}
        experiment_id = "3"
        experiment.id = experiment_id
        run = MagicMock()
        run.metadata = {"server_url": "testrun.com"}
        run_id = "4"
        run.id = run_id
        split = MagicMock()
        split.metadata = {"server_url": ""}
        split_id = "5"
        split.id = split_id
        response = self._remote_experiments.validate_and_save(
            experiment, run, split, local_experiments=self._local_experiments)
        self.assertTrue(response, "Split which not exists on server is not upload")
        expected = [call({"server_url": self.returned_url}, experiment_id, run_id, split_id)]
        self.assertEqual(expected,
                         self._local_experiments.update_metadata.call_args_list,
                         "Update metadata not called with expected arguments for split")

    def tearDown(self):
        pass
