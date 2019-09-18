from pypadre.core import Run
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.generic.i_file_repository import File, IChildFileRepository
from pypadre.pod.repository.i_repository import IRunRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer, PickleSerializer

NAME = "runs"

META_FILE = File("metadata.json", JSonSerializer)
HYPERPARAMETER_FILE = File("hyperparameter.json", JSonSerializer)
WORKFLOW_FILE = File("workflow.json", PickleSerializer)


class RunFileRepository(IChildFileRepository, IRunRepository):

    @staticmethod
    def placeholder():
        return '{RUN_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(parent=backend.execution, name=NAME, backend=backend)

    def get_by_dir(self, directory):
        metadata = self.get_file(directory, META_FILE)
        hyperparameter = self.get_file(directory, HYPERPARAMETER_FILE)
        workflow = self.get_file(directory, WORKFLOW_FILE)

        # TODO what to do with hyperparameters?
        execution = self.parent.get_by_dir(self.get_parent_dir(directory))
        run = Run(execution=execution, workflow=workflow, **metadata)
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

        return super().get(uid)

    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        run = obj
        self.write_file(directory, META_FILE, run.metadata)
        self.write_file(directory, HYPERPARAMETER_FILE, run.experiment.hyperparameters())
        self.write_file(directory, WORKFLOW_FILE, run.workflow, "wb")
