import copy
import os

from pypadre import Experiment
from pypadre.backend.interfaces.backend.generic.i_base_file_backend import File
from pypadre.backend.interfaces.backend.i_experiment_backend import IExperimentBackend
from pypadre.backend.local.file.interfaces.i_base_binary_file_backend import IBaseBinaryFileBackend
from pypadre.backend.local.file.project.experiment.execution.execution_file_backend import PadreExecutionFileBackend
from pypadre.backend.serialiser import JSonSerializer, PickleSerializer


class PadreExperimentFileBackend(IExperimentBackend, IBaseBinaryFileBackend):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "experiments")
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

    def put_progress(self, experiment):
        self.log("EXPERIMENT PROGRESS: {curr_value}/{limit}. phase={phase} \n".format(phase=phase, curr_value=curr_value, limit=limit))

    def put(self, experiment, allow_overwrite=True):
        directory = self.get_dir(self.to_folder_name(experiment))

        if os.path.exists(directory) and not allow_overwrite:
            raise ValueError("Experiment %s already exists." +
                             "Overwriting not explicitly allowed. Set allow_overwrite=True".format(experiment.name))

        self.write_file(directory, self.META_FILE, experiment.metadata)
        self.write_file(directory, self.WORKFLOW_FILE, experiment.workflow)

        # TODO when to write experiment.json???

        if experiment.requires_preprocessing:
            self.write_file(directory, self.PREPROCESS_WORKFLOW_FILE, experiment.preprocessing_workflow)
