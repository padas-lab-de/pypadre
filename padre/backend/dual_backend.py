from padre.experiment import Experiment


class DualBackend:
    """
    For realising transparent file and http backend this class is implemented that
    combines both file and http backend. Transparent thereby means, that the file backend
    is used as cache or that we can sync file cache easily with the http server.

    Functionality
    ============
    Implement all functions from file and http backends here and then function calls will be
    delegated to appropriate backend.
    """
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

    def get_dataset(self, dataset_id, metadata_only=False):
        """Get data set from both http and file backend."""
        return self._file_datasets.get_dataset(dataset_id, metadata_only)

    def delete_dataset(self, id):
        """Delete data set from both http and file backend."""
        raise NotImplemented

    # functions for experiment management. See experiment backend and issue #45
    def list_experiments(self, search_id=".*", search_metadata=None, start=-1, count=999999999,
                         remote=True):
        """
        Lists the count experiments according to the provided search string starting from
        entry number start. If remote is True, we also search the http backend. search is a
        string that searches by default in the title of the experiment. We will later define
        a syntax to search also associated metadata (e.g. "description:search_string").
        The function returns a list of strings with experiment names / experiment identifies
        that could be used in get_experiment.

        :param search_id:
        :param search_metadata:
        :param start:
        :param count:
        :param remote:
        :return: List of experiments
        # todo: Implement for http backend

        """
        return self._file_experiments.list_experiments(search_id, search_metadata)

    def delete_experiments(self, experiment=".*", mode="all", search_metadata=None):
        """
        Deletes the experiment given by ex (either as class or as string) from the local file cache
        or the remote server.

        :param experiment: Experiment instance or string
        :param mode: mode can be 'local|remote|all'
        :param search_metadata:
        :return: None
        """
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
        """
         Uploads the experiment given by ex to the server using the configured backends
         if the experiment did not change or is not available on the server. ex can be either an
         instance of the experiment class or a string with the name of the experiment in the file
         backend. If it is an instance of the experiment class, the experiment class is first
         stored to disk using the file backend. If it is a string with the name, the experiment
         is loaded from the file backend and stored to the serer. The experiment in the file
         backend should receive an url in its metadata file to indicate that it has been uploaded
         to the server. An experiment should only be put to the server or the file if the
         experiment did change. But for a first version we simply look at timestamps.
         .

        :param experiment: Either instance of experiment or name of the experiment
        :param append_runs:
        :param allow_overwrite:
        :return: The function returns the experiment class that has been put to the server
        """
        if isinstance(experiment, Experiment):
            url = self._http_experiments.put_experiment(experiment)
            experiment.metadata["server_url"] = url
            self._file_experiments.put_experiment(experiment, append_runs)
        elif isinstance(experiment, str):
            experiment = self._file_experiments.get_experiment(experiment)
            self._http_experiments.put_experiment(experiment)
        return experiment

    def get_experiment(self, ex, load_workflow=True):
        """
        Get experiment from server if not found then find it from local file system

        :param ex: id or url of the experiment
        :param load_workflow:
        :return:

        todo: Implement for http backend
        """
        result = self._http_experiments.get_experiment(ex)
        if isinstance(result, Experiment):
            return result
        return self._file_experiments.get_experiment(ex, load_workflow)

    def put_run(self, experiment, run):
        server_url = self._http_experiments.put_run(experiment, run)
        run.metadata["server_url"] = server_url
        self._file_experiments.put_run(experiment, run)


    def put_split(self, experiment, run, split):
        server_url = self._http_experiments.put_split(experiment, run, split)
        split.metadata["server_url"] = server_url
        self._file_experiments.put_split(experiment, run, split)


    def put_results(self, experiment, run, split, results):
        self._http_experiments.put_results(experiment, run, split, results)
        self._file_experiments.put_results(experiment, run, split, results)

    def put_metrics(self, experiment, run, split, metrics):
        return self._file_experiments.put_metrics(experiment, run, split, metrics)

  # todo implement all functions currently needed by the experiment class (when the backend is set)
