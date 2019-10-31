import os
import re

from cachetools import LRUCache, cached

from pypadre.core.model.experiment import Experiment
from pypadre.core.util.utils import remove_cached
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IExperimentRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File, IChildFileRepository
from pypadre.pod.repository.local.file.generic.i_git_repository import IGitRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer, DillSerializer
# CONFIG_FILE = File("experiment.json", JSonSerializer)
from pypadre.pod.util.git_util import create_repo, add_and_commit, git_hash

WORKFLOW_FILE = File("workflow.pickle", DillSerializer)
META_FILE = File("metadata.json", JSonSerializer)

NAME = 'experiments'
cache = LRUCache(maxsize=16)


class ExperimentFileRepository(IChildFileRepository, IGitRepository, IExperimentRepository):

    @staticmethod
    def placeholder():
        return '{EXPERIMENT_ID}'

    def __init__(self, backend: IPadreBackend, **kwargs):
        super().__init__(parent=backend.project, name=NAME, backend=backend, **kwargs)

    # TODO caching could also be done on service layer
    # TODO write tests for caching
    @cached(cache)
    def get(self, uid):
        return super().get(uid)

    def delete(self, id):
        remove_cached(cache, id)
        return super().delete(id)

    def to_folder_name(self, experiment):
        return experiment.name

    def get_by_name(self, name):
        """
        Shortcut because we know name is the folder name. We don't have to search in metadata.json
        :param name: Name of the dataset
        :return:
        """
        return self.list({'folder': re.escape(name)})

    def get_by_dir(self, directory):
        import glob

        path = glob.glob(os.path.join(self._replace_placeholders_with_wildcard(self.root_dir), directory))[0]

        metadata = self.get_file(path, META_FILE)
        # config = self.get_file(path, CONFIG_FILE)
        pipeline = self.get_file(path, WORKFLOW_FILE)
        # preprocess_workflow = self.get_file(path, PREPROCESS_WORKFLOW_FILE)

        project = self.backend.project.get(metadata.get(Experiment.PROJECT_ID))
        dataset = self.backend.dataset.get(metadata.get(Experiment.DATASET_ID))

        # TODO only pass metadata / config etc to experiment creator. We shouldn't think about the structure of experiments here

        ex = Experiment(name=metadata.get("name"), description=metadata.get("description"), project=project,
                        dataset=dataset, metadata=metadata, pipeline=pipeline)
        return ex

    def put_progress(self, experiment, **kwargs):
        self.log("EXPERIMENT PROGRESS: {curr_value}/{limit}. phase={phase} \n".format(**kwargs))

    def _put(self, experiment: Experiment, *args, directory, merge=False, **kwargs):

        # update experiment
        if merge:
            metadata = self.get_file(directory, META_FILE)
            if metadata is not None:
                # TODO this merge function should merge our changes into the already existing data and not the other
                # way around
                experiment.merge_metadata(metadata=metadata)

        self.write_file(directory, META_FILE, experiment.metadata)
        self.write_file(directory, WORKFLOW_FILE, experiment.pipeline, 'wb')
        remove_cached(cache, experiment.id)
