from typing import List

from pypadre.pod.app.base_app import BaseChildApp
from pypadre.pod.repository.i_repository import IExperimentRepository
from pypadre.pod.service.experiment_service import ExperimentService


class ExperimentApp(BaseChildApp):
    """
    Class providing commands for managing datasets.
    """
    def __init__(self, parent, backends: List[IExperimentRepository], **kwargs):
        super().__init__(parent, service=ExperimentService(backends=backends), **kwargs)

    def execute(self, id):
        return self.service.execute(id)

    def create(self, *args, **kwargs):
        experiment = self.service.create(*args, **kwargs)
        self.put(experiment)
        return experiment

    # def delete_experiments(self, search):
    #     """
    #        lists the experiments and returns a list of experiment names matching the criterions
    #        :param search: str to search experiment name only or
    #        dict object with format {field : regexp<String>} pattern to search in particular fields using a regexp.
    #        None for all experiments
    #     """
    #     if isinstance(search, dict):
    #         s = copy.deepcopy(search)
    #         file_name = s.pop("name")
    #     else:
    #         file_name = search
    #         s = None
    #
    #     self._parent.local_backend.experiments.delete_experiments(search_id=file_name, search_metadata=s)
    #
    # def list_experiments(self, search=None, start=-1, count=999999999, ):
    #     """
    #     lists the experiments and returns a list of experiment names matching the criterions
    #     :param search: str to search experiment name only or
    #     dict object with format {field : regexp<String>} pattern to search in particular fields using a regexp.
    #     None for all experiments
    #     :param start: start in the list to be returned
    #     :param count: number of elements in the list to be returned
    #     :return:
    #     """
    #     if search is not None:
    #         if isinstance(search, dict):
    #             s = copy.deepcopy(search)
    #             file_name = s.pop("name")
    #         else:
    #             file_name = search
    #             s = None
    #         return _sub_list(self._parent.local_backend.experiments.list_experiments(search_id=file_name,
    #                                                                                  search_metadata=s)
    #                          , start, count)
    #     else:
    #         return _sub_list(self._parent.local_backend.experiments.list_experiments(), start, count)
    #
    # def list_runs(self, ex_id, start=-1, count=999999999, search=None):
    #     return _sub_list(self._parent.local_backend.experiments.list_runs(ex_id), start, count)
    #
    # def run(self, **ex_params):
    #     """
    #     runs an experiment either with the given parameters or, if there is a parameter decorated=True, runs all
    #     decorated experiments.
    #     Befor running the experiments, the backend for storing results is configured as file_repository.experiments
    #     :param ex_params: kwargs for an experiment or decorated=True
    #     :return:
    #     """
    #     if "decorated" in ex_params and ex_params["decorated"]:
    #         from pypadre.decorators import run
    #         return run()
    #     else:
    #         p = ex_params.copy()
    #         ex = Experiment(**p)
    #         ex.run()
    #         return ex
    #
    # def pull(self, ex_id):
    #     """
    #     Download experiment, run and split from server if it does not exists on local directory
    #     Download all runs, splits, results and metrics from the server associated with the experiment.
    #     Downloaded experiment will be saved in the local file system.
    #
    #     :param ex_id: Can be experiment name or experiment id or experiment url.
    #     :type ex_id: int or str
    #     :return: Experiment
    #     Todo: In case ex_id is name of experiment and if two experiments with this name exists on the server first one will be downloaded
    #     """
    #     remote_experiments_ = self._parent.remote_backend.experiments
    #     local_experiments_ = self._parent.local_backend.experiments
    #     ex = remote_experiments_.get_experiment(ex_id)
    #     if not ex_id.isdigit():
    #         ex_id = ex.metadata["server_url"].split("/")[-1]
    #     local_experiments_.validate_and_save(ex)
    #     for run_id in remote_experiments_.get_experiment_run_idx(ex_id):
    #         r = remote_experiments_.get_run(ex_id, run_id)
    #         local_experiments_.validate_and_save(ex, r)
    #         for split_id in remote_experiments_.get_run_split_idx(ex_id, run_id):
    #             s = remote_experiments_.get_split(ex_id, run_id, split_id)
    #             if local_experiments_.validate_and_save(ex, r, s):
    #                 local_experiments_.put_results(ex, r, s, s.run.results[0])
    #                 local_experiments_.put_metrics(ex, r, s, s.run.metrics[0])
    #     return ex
    #
    # def push(self, experiment_name):
    #     """Upload given experiment with all runs and splits.
    #
    #     Upload all runs, splits, results and metrics to the server associated with the experiment.
    #     If any experiment, run or split is already uploaded then it will not be uploaded second time.
    #
    #     To check uniqueness on server: Before uploading check if its server_url in metadata is not empty if its
    #     empty then upload it to server and after uploading experiment, run or split update its url in the metadata
    #     so that its not uploaded second time.
    #
    #     :param experiment_name: Name of the experiment on local system
    #     :type experiment_name: str
    #     :return: Experiment
    #     """
    #     experiment_path = os.path.join(self._parent.local_backend.root_dir, "experiments",
    #                                    experiment_name + ".ex")
    #     assert_condition(
    #         condition=experiment_name.strip() != "" and os.path.exists(os.path.abspath(experiment_path)),
    #         source=self,
    #         message='Experiment not found')
    #     remote_experiments_ = self._parent.remote_backend.experiments
    #     local_experiments_ = self._parent.local_backend.experiments
    #     ex = local_experiments_.get_experiment(experiment_name)
    #     ex.metadata["server_url"] = remote_experiments_.validate_and_save(ex, local_experiments=local_experiments_)
    #
    #     list_of_runs = filter(lambda x: x.endswith(".run"), os.listdir(experiment_path))
    #     for run_name in list_of_runs:  # Upload all runs for this experiment
    #         run_path = os.path.join(experiment_path, run_name)
    #         r = local_experiments_.get_run(experiment_name,
    #                                        run_name.split(".")[0])
    #         r.metadata["server_url"] = remote_experiments_.validate_and_save(ex, r, local_experiments=local_experiments_)
    #
    #         list_of_splits = filter(lambda x: x.endswith(".split"), os.listdir(run_path))
    #         for split_name in list_of_splits:  # Upload all splits for this run
    #             s = local_experiments_.get_split(experiment_name,
    #                                              run_name.split(".")[0],
    #                                              split_name.split(".")[0])
    #             if remote_experiments_.validate_and_save(ex, r, s, local_experiments=local_experiments_):
    #                 remote_experiments_.put_results(ex, r, s, s.run.results[0])
    #                 remote_experiments_.put_metrics(ex, r, s, s.run.metrics[0])
    #     return ex
    #
    # def sync(self, name):
    #     """Todo: Implement after discussion on issue#67"""
    #     pass


