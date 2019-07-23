import os
import shutil
import uuid

from pypadre.backend.interfaces.backend.generic.i_base_file_backend import File
from pypadre.backend.interfaces.backend.i_run_backend import IRunBackend
from pypadre.backend.local.file.project.experiment.execution.run.split.split_file_backend import PadreSplitFileBackend
from pypadre.backend.serialiser import JSonSerializer, PickleSerializer
from pypadre.core import Run


class PadreRunFileBackend(IRunBackend):

    def __init__(self, parent):
        super().__init__(parent)
        self.root_dir = os.path.join(self._parent.root_dir, "runs")
        self._split = PadreSplitFileBackend(self)

    META_FILE = File("metadata.json", JSonSerializer)
    HYPERPARAMETER_FILE = File("hyperparameter.json", JSonSerializer)
    WORKFLOW_FILE = File("workflow.json", PickleSerializer)

    @property
    def split(self):
        return self._split

    def put_progress(self, run):
        self.log(
            "RUN PROGRESS: {curr_value}/{limit}. phase={phase} \n".format(phase=phase, curr_value=curr_value,
                                                                                 limit=limit))

    def get(self, uid):
        """
        Shortcut because we know the id is the folder name. We don't have to search in metadata.json
        :param uid: Uid of the run
        :return:
        """

        directory = self.get_dir(uid)
        metadata = self.get_file(directory, self.META_FILE)
        hyperparameter = self.get_file(directory, self.HYPERPARAMETER_FILE)
        workflow = self.get_file(directory, self.WORKFLOW_FILE)

        # TODO create Run
        run = Run()
        return run

    def put(self, run):
        """
        Stores a run of an experiment to the file repository.
        :param run: run to put
        :return:
        """
        if run.id is None:  # this is a new experiment
            run.id = uuid.uuid4()

        directory = self.get_dir(self.to_folder_name(run))

        if os.path.exists(directory) and not allow_overwrite:
            raise ValueError("Experiment %s already exists." +
                             "Overwriting not explicitly allowed. Set allow_overwrite=True".format(experiment.name))
        else:
            shutil.rmtree(directory)

        os.mkdir(directory)
        self.write_file(directory, self.META_FILE, run.metadata)
        # TODO get hyperparameters ?
        self.write_file(directory, self.HYPERPARAMETER_FILE, run.hyperparameters)
        self.write_file(directory, self.WORKFLOW_FILE, run.workflow)
