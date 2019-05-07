"""
This file contains tests covering app.padre_app.ExperimentApp.validate_and_upload function
"""
import copy
import unittest

from mock import MagicMock, call

from padre.app import pypadre


class ValidateAndUpload(unittest.TestCase):
    """Test ExperimentApp.validate_and_upload function"""
    def setUp(self):
        """Create mock objects"""
        self.returned_url = "http://test.com/api/object/id"
        self.put_fn = MagicMock(return_value=self.returned_url)
        self.pypadre = copy.deepcopy(pypadre)
        self.pypadre.local_backend.experiments.update_metadata = MagicMock()

    def test_validate_and_upload_01(self):
        """test app.padre_app.Experiment.validate_and_update function for experiment

        Scenario: Upload experiment which is already uploaded
        """
        experiment = MagicMock()
        experiment.metadata = {"server_url": "http://test.com"}
        response = self.pypadre.experiments.validate_and_upload(self.put_fn, experiment)
        self.assertFalse(response, "Experiment which already exists on server is upload")

    def test_validate_and_upload_02(self):
        """test app.padre_app.Experiment.validate_and_update function for experiment

        Scenario: Upload experiment which is not already uploaded
        """

        experiment = MagicMock()
        experiment.metadata = {"server_url": ""}
        experiment_id = "3"
        experiment.id = experiment_id
        response = self.pypadre.experiments.validate_and_upload(self.put_fn, experiment)
        self.assertTrue(response, "Experiment which not exists on server is not upload")
        expected = [call({"server_url": self.returned_url}, experiment_id)]
        self.assertEqual(expected,
                         self.pypadre.local_backend.experiments.update_metadata.call_args_list,
                         "Update metadata not called with expected arguments")

    def test_validate_and_upload_03(self):
        """test app.padre_app.Experiment.validate_and_update function for run

        Scenario: Upload run which is already uploaded
        """
        experiment = MagicMock()
        run = MagicMock()
        run.metadata = {"server_url": "http://testrun.com"}
        response = self.pypadre.experiments.validate_and_upload(self.put_fn, experiment, run)
        self.assertFalse(response, "Run which already exists on server is upload")

    def test_validate_and_upload_04(self):
        """test app.padre_app.Experiment.validate_and_update function for run

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
        response = self.pypadre.experiments.validate_and_upload(self.put_fn, experiment, run)
        self.assertTrue(response, "Run which not exists on server is not upload")
        expected = [call({"server_url": self.returned_url}, experiment_id, run_id)]
        self.assertEqual(expected,
                         self.pypadre.local_backend.experiments.update_metadata.call_args_list,
                         "Update metadata not called with expected arguments for run")

    def test_validate_and_upload_05(self):
        """test app.padre_app.Experiment.validate_and_update function for split

        Scenario: Upload split which is already uploaded
        """
        experiment = MagicMock()
        run = MagicMock()
        split = MagicMock()
        split.metadata = {"server_url": "http://testsplit.com"}
        response = self.pypadre.experiments.validate_and_upload(self.put_fn, experiment, run)
        self.assertFalse(response, "Run which already exists on server is upload")

    def test_validate_and_upload_06(self):
        """test app.padre_app.Experiment.validate_and_update function for split

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
        response = self.pypadre.experiments.validate_and_upload(self.put_fn, experiment, run, split)
        self.assertTrue(response, "Split which not exists on server is not upload")
        expected = [call({"server_url": self.returned_url}, experiment_id, run_id, split_id)]
        self.assertEqual(expected,
                         self.pypadre.local_backend.experiments.update_metadata.call_args_list,
                         "Update metadata not called with expected arguments for split")

    def tearDown(self):
        """Remove pypadre object after test"""
        del self.pypadre
