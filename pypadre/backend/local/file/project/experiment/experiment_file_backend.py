import os

from pypadre.backend.interfaces.backend.i_experiment_backend import IExperimentBackend
from pypadre.backend.local.file.interfaces.i_base_binary_file_backend import IBaseBinaryFileBackend
from pypadre.backend.local.file.project.experiment.execution.execution_file_backend import PadreExecutionFileBackend


class PadreExperimentFileBackend(IExperimentBackend, IBaseBinaryFileBackend):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "experiments")
        self._execution = PadreExecutionFileBackend(self)

    def put_config(self, obj):
        pass

    @property
    def execution(self):
        return self._execution

    def _init(self):
        pass

    def _commit(self):
        pass

    def to_folder_name(self, obj):
        return obj.id

    def get_by_name(self, name):
        """
        Shortcut because we know name is the folder name. We don't have to search in metadata.json
        :param name: Name of the dataset
        :return:
        """
        return self.get_by_dir(self.get_dir(name))

    def get_by_dir(self, directory):
        self.get_meta_file(directory)
        #TODO parse to experiment object
        pass

    def log(self, msg):
        # TODO log something
        pass

    def put_progress(self, obj):
        # TODO implement
        pass

    def put(self, obj, allow_overwrite=True):
        folder_name = self.to_folder_name(obj)
        directory = self.get_dir(folder_name)

        if os.path.exists(directory) and not allow_overwrite:
            raise ValueError("Experiment %s already exists." +
                             "Overwriting not explicitly allowed. Set allow_overwrite=True")

        super().put(obj)

        if obj.requires_preprocessing:
            with open(os.path.join(directory, "preprocessing_workflow.bin"), 'wb') as f:
                f.write(self._binary_serializer.serialise(experiment.preprocessing_workflow))