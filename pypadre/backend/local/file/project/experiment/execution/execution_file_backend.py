import os

from pypadre.backend.interfaces.backend.generic.i_base_file_backend import File
from pypadre.backend.interfaces.backend.i_execution_backend import IExecutionBackend
from pypadre.backend.local.file.project.experiment.execution.run.run_file_backend import PadreRunFileBackend
from pypadre.backend.serialiser import JSonSerializer
from pypadre.core.model.execution import Execution


class PadreExecutionFileBackend(IExecutionBackend):

    @staticmethod
    def _placeholder():
        return '{EXECUTION_ID}'

    @staticmethod
    def _get_parent_of(obj: Execution):
        return obj.experiment

    CONFIG_FILE = File("experiment.json", JSonSerializer)
    NAME = 'executions'
    META_FILE = File("metadata.json", JSonSerializer)

    def __init__(self, parent):
        super().__init__(parent, name=self.NAME)
        self.root_dir = os.path.join(self._parent.root_dir, self._parent._placeholder(), self.NAME)
        self._run = PadreRunFileBackend(self)

    @property
    def run(self):
        return self._run

    def to_folder_name(self, obj):
        return obj.name

    def get(self, uid):
        """
        Shortcut because we know the uid is the folder name
        :param uid: Uid of the execution
        :return:
        """
        # TODO might be changed. Execution get folder name or id by git commit hash?
        return super().get(uid)

    def get_by_dir(self, directory):
        metadata = self.get_file(directory, self.META_FILE)
        #TODO parse to execution object
        pass

    def put(self, execution, allow_overwrite=True):
        directory = self.to_directory(execution)

        if os.path.exists(directory) and not allow_overwrite:
            raise ValueError("Experiment %s already exists." +
                             "Overwriting not explicitly allowed. Set allow_overwrite=True".format(
                                 self.to_folder_name(execution)))

        elif not os.path.exists(directory):
            os.makedirs(directory)

        self.write_file(directory, self.META_FILE, execution.metadata)

        # The code for each execution changes. So it is necessary to write the experiment.json file too.
        self.write_file(directory, self.CONFIG_FILE, execution.config)

    def patch(self, execution, allow_overwrite=True):
        directory = self.to_directory(execution)

        if os.path.exists(directory):
            metadata = self.get_file(directory, self.META_FILE)
            # Patch the existing metadata with the metadata of the project
            execution.merge_metadata(metadata=metadata)

        elif not os.path.exists(directory):
            os.makedirs(directory)

        self.write_file(directory, self.META_FILE, execution.metadata)

        # The configuration for the execution MUST be identical, else throw an error.
        # So there is no need to rewrite the config file. Any change should create a new execution
        self.write_file(directory, self.CONFIG_FILE, execution.config)
