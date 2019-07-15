import os

from pypadre.backend.interfaces.backend.i_execution_backend import IExecutionBackend
from pypadre.backend.local.file.project.experiment.execution.run.run_file_backend import PadreRunFileBackend


class PadreExecutionFileBackend(IExecutionBackend):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "executions")
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
        return self.get_by_dir(self.get_dir(uid))

    def get_by_dir(self, directory):
        self.get_meta_file(directory);
        #TODO parse to execution object
        pass

    def put(self, obj):
        pass