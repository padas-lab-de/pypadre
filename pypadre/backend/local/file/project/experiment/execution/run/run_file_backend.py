import os
import shutil
import uuid

from pypadre.backend.interfaces.backend.generic.i_base_file_backend import File
from pypadre.backend.interfaces.backend.i_run_backend import IRunBackend
from pypadre.backend.local.file.project.experiment.execution.run.split.split_file_backend import PadreSplitFileBackend
from pypadre.backend.serialiser import JSonSerializer, PickleSerializer
from pypadre.core import Run


class PadreRunFileBackend(IRunBackend):
    NAME = "runs"
    PLACEHOLDER = '{RUN_ID}'

    def __init__(self, parent):
        super().__init__(parent, name=self.NAME)
        self.root_dir = os.path.join(self._parent.root_dir, self._parent.PLACEHOLDER, self.NAME)
        self._split = PadreSplitFileBackend(self)

    META_FILE = File("metadata.json", JSonSerializer)
    HYPERPARAMETER_FILE = File("hyperparameter.json", JSonSerializer)
    WORKFLOW_FILE = File("workflow.json", PickleSerializer)

    @property
    def split(self):
        return self._split

    def to_folder_name(self, run):
        return run.id

    def get_by_dir(self, directory):
        metadata = self.get_file(directory, self.META_FILE)
        hyperparameter = self.get_file(directory, self.HYPERPARAMETER_FILE)
        workflow = self.get_file(directory, self.WORKFLOW_FILE)

        # TODO create Run
        execution = self.parent.get_by_dir(directory)
        run = Run(execution, workflow, **metadata)
        # TODO what to do with hyperparameters?
        return run

    def log(self, msg):
        pass

    def put_progress(self, run, **kwargs):
        self.log(
            "RUN PROGRESS: {curr_value}/{limit}. phase={phase} \n".format(**kwargs))

    def get(self, uid):
        """
        Shortcut because we know the id is the folder name. We don't have to search in metadata.json
        :param uid: Uid of the run
        :return:
        """

        directory = self.get_dir(uid)
        return self.get_by_dir(directory)

    def put(self, run, allow_overwrite=False):
        """
        Stores a run of an experiment to the file repository.
        :param allow_overwrite: allow overwrite of experiment
        :param run: run to put
        :return:
        """
        if run.id is None:  # this is a new experiment
            run.id = uuid.uuid4()

        directory = self.get_dir(self.to_folder_name(run))

        if os.path.exists(directory) and not allow_overwrite:
            raise ValueError("Run %s already exists." +
                             "Overwriting not explicitly allowed. Set allow_overwrite=True".format(run.id))
        else:
            shutil.rmtree(directory)
        os.mkdir(directory)

        self.write_file(directory, self.META_FILE, run.metadata)
        self.write_file(directory, self.HYPERPARAMETER_FILE, run.experiment.hyperparameters())
        self.write_file(directory, self.WORKFLOW_FILE, run.workflow)
