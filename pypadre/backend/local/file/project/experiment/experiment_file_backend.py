import copy
import os

from pypadre import Experiment
from pypadre.backend.interfaces.backend.generic.i_base_file_backend import File
from pypadre.backend.interfaces.backend.i_experiment_backend import IExperimentBackend
from pypadre.backend.local.file.project.experiment.execution.execution_file_backend import PadreExecutionFileBackend
from pypadre.backend.serialiser import JSonSerializer, PickleSerializer


class PadreExperimentFileBackend(IExperimentBackend):

    def __init__(self, parent):
        name = 'experiments'
        placeholder = '{EXPERIMENT_ID}'
        super().__init__(parent, name=name)
        self.root_dir = os.path.join(self._parent.root_dir, name, placeholder)
        self._execution = PadreExecutionFileBackend(self)

    META_FILE = File("metadata.json", JSonSerializer)
    CONFIG_FILE = File("experiment.json", JSonSerializer)
    WORKFLOW_FILE = File("workflow.json", PickleSerializer)
    PREPROCESS_WORKFLOW_FILE = File("preprocessing_workflow.json", PickleSerializer)

    @property
    def execution(self):
        return self._execution

    def put_config(self, experiment):
        pass

    def to_folder_name(self, experiment):
        return experiment.id

    def get_by_name(self, name):
        """
        Shortcut because we know name is the folder name. We don't have to search in metadata.json
        :param name: Name of the dataset
        :return:
        """
        return self.get_by_dir(self.get_dir(name))

    def get_by_dir(self, directory):
        metadata = self.get_file(directory, self.META_FILE)
        config = self.get_file(directory, self.CONFIG_FILE)
        workflow = self.get_file(directory, self.WORKFLOW_FILE)
        preprocess_workflow = self.get_file(directory, self.PREPROCESS_WORKFLOW_FILE)

        # TODO only pass metadata / config etc to experiment creator. We shouldn't think about the structure of experiments here
        ex = Experiment(ex_id=id_, **experiment_params[id_])
        return ex

    def put_progress(self, experiment, **kwargs):
        # TODO kwargs
        self.log("EXPERIMENT PROGRESS: {curr_value}/{limit}. phase={phase} \n".format(**kwargs))

    def put(self, experiment, allow_overwrite=True):
        directory = self.get_dir(self.to_folder_name(experiment))

        if os.path.exists(directory) and not allow_overwrite:
            raise ValueError("Experiment %s already exists." +
                             "Overwriting not explicitly allowed. Set allow_overwrite=True".format(experiment.name))

        self.write_file(directory, self.META_FILE, experiment.metadata)
        self.write_file(directory, self.WORKFLOW_FILE, experiment.workflow)

        # TODO when to write experiment.json???
        # TODO: Experiment.json should be written within the execution folder as any change
        #  to the experiment configuration would spawn a new directory

        if experiment.requires_preprocessing:
            self.write_file(directory, self.PREPROCESS_WORKFLOW_FILE, experiment.preprocessing_workflow)

        # Git operation of creating a repository
        self._create_repo(path=directory, bare=False)

        # TODO: Add experiment as a submodule to the project repo

    def add_and_commit(self, path):
        repo = self.get_repo(path=path)
        self._add_untracked_files(repo=repo)
        self._commit(repo, message=self._DEFAULT_GIT_MSG)
