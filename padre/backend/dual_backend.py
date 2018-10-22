from padre.experiment import Experiment


class DualBackend:

    def __init__(self, file_backend, http_backend):
        self._file_backend = file_backend
        self._http_backend = http_backend
        self._http_experiments = http_backend.experiments
        self._file_datasets = file_backend.datasets
        self._file_experiments = file_backend.experiments

    def list_datasets(self, search_id=None, search_metadata=None):
        """List data sets from both file backend and http backend."""
        return self._file_datasets.list_datasets(search_id, search_metadata)

    def put_dataset(self, dataset):
        """Put data set on both and http and file backend."""
        return self._file_datasets.put_dataset(dataset)

    def get_dataset(self, id, metadata_only=False):
        """Get data set from both http and file backend."""
        return self._file_datasets.get_dataset(id, metadata_only)

    def delete_dataset(self, id):
        """Delete data set from both http and file backend."""
        raise NotImplemented

    # functions for experiment management. See experiment backend and issue #45
    def list_experiments(self, search_id=".*", search_metadata=None):
        return self._file_experiments.list_experiments(search_id, search_metadata)

    def delete_experiments(self, experiment=".*", mode="all", search_metadata=None):
        if isinstance(experiment, Experiment):
            experiment = experiment.metadata["name"]
        if mode == "local":
            self._file_experiments.delete_experiments(experiment, search_metadata)
        if mode == "remote":
            self._http_experiments.delete_experiment(experiment)
        if mode == "all":
            self._file_experiments.delete_experiments(experiment, search_metadata)
            self._http_experiments.delete_experiment(experiment)

    def put_experiment(self, experiment, append_runs=False, allow_overwrite=False):
        if isinstance(experiment, Experiment):
            url = self._http_experiments.put_experiment(experiment)
            experiment.metadata["server_url"] = url
            self._file_experiments.put_experiment(experiment, append_runs)
        elif isinstance(experiment, str):
            experiment = self._file_experiments.get_experiment(experiment)
            self._http_experiments.put_experiment(experiment)
        return experiment

    def get_experiment(self, id_, load_workflow=True):
        return self._file_experiments.get_experiment(id_, load_workflow)

    def put_run(self, experiment, run):
        return self._file_experiments.put_run(experiment, run)

    def put_split(self, experiment, run, split):
        return self._file_experiments.put_split(experiment, run, split)

    def put_result(self, experiment, run, split, results):
        return self._file_experiments.put_results(experiment, run, split, results)

    def put_metrics(self, experiment, run, split, metrics):
        return self._file_experiments.put_metrics(experiment, run, split, metrics)

  # todo implement all functions currently needed by the experiment class (when the backend is set)
