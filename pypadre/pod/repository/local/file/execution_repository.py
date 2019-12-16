import glob
import os

from pypadre.core.model.execution import Execution
from pypadre.core.model.experiment import Experiment
from pypadre.core.model.generic.custom_code import CodeManagedMixin
from pypadre.core.model.generic.lazy_loader import SimpleLazyObject
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IExecutionRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File, IChildFileRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer, DillSerializer

NAME = 'executions'

# CONFIG_FILE = File("experiment.json", JSonSerializer)
META_FILE = File("metadata.json", JSonSerializer)
WORKFLOW_FILE = File("workflow.pickle", DillSerializer)


class ExecutionFileRepository(IChildFileRepository, IExecutionRepository):

    @staticmethod
    def placeholder():
        return '{EXECUTION_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(parent=backend.experiment, name=NAME, backend=backend)

    def to_folder_name(self, execution):
        return str(execution.id)

    def get(self, uid):
        """
        Shortcut because we know the uid is the folder name
        :param uid: Uid of the execution
        :return:
        """
        # TODO: Execution folder name is the hash. Get by uid will require looking into the metadata
        return super().get(uid)

    def _get_by_dir(self, directory):

        path = glob.glob(os.path.join(self._replace_placeholders_with_wildcard(self.root_dir), directory))[0]

        metadata = self.get_file(directory, META_FILE)
        experiment = SimpleLazyObject(load_fn=lambda: self.backend.experiment.get(metadata.get(Execution.EXPERIMENT_ID)),
                                           id=metadata.get(Execution.EXPERIMENT_ID), clz=Experiment)
        reference = self.backend.code.get(metadata.get(CodeManagedMixin.DEFINED_IN))
        pipeline = self.get_file(path, WORKFLOW_FILE)
        # runs = self.backend.run.list({'execution_id': metadata.get('id')})
        return Execution(experiment=experiment, metadata=metadata, reference=reference, pipeline=pipeline)

    def _put(self, obj, *args, directory: str, merge=False, **kwargs):
        execution = obj
        self.write_file(directory, META_FILE, execution.metadata)
        self.write_file(directory, WORKFLOW_FILE, execution.pipeline, 'wb')

        # The code for each execution changes. So it is necessary to write the experiment.json file too.
        # self.write_file(directory, CONFIG_FILE, execution.config)
