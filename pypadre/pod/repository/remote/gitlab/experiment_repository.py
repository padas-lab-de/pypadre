import os
import re

from cachetools import LRUCache, cached

from pypadre.core.model.execution import Execution
from pypadre.core.model.experiment import Experiment
from pypadre.core.util.utils import remove_cached
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import IExperimentRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File, IChildFileRepository
from pypadre.pod.repository.remote.gitlab.generic.gitlab import GitLabRepository
from pypadre.pod.repository.serializer.serialiser import DillSerializer, JSonSerializer, YamlSerializer
from pypadre.pod.util.git_util import add_and_commit

WORKFLOW_FILE = File("workflow.pickle", DillSerializer)
META_FILE = File("metadata.json", JSonSerializer)
MANIFEST_FILE = File("manifest.yml", YamlSerializer)
NAME = 'experiments'

cache = LRUCache(maxsize=16)


class ExperimentGitlabRepository(IChildFileRepository, GitLabRepository, IExperimentRepository):

    @staticmethod
    def placeholder():
        return '{EXPERIMENT_ID}'

    def __init__(self, backend:IPadreBackend, **kwargs):
        # self._parent = backend.project
        # if self._parent.has_placeholder():
        #     root_dir = os.path.join(self._parent.root_dir, self._parent.placeholder(), NAME)
        # else:
        #     root_dir = os.path.join(self._parent.root_dir, NAME)
        super().__init__(parent=backend.project,name=NAME, gitlab_url=backend.url, token=backend.token
                         , backend=backend,**kwargs)

        self._group = self.get_group(name=NAME)

    def get_by_name(self, name):
        """
        Shortcut because we know name is the folder name. We don't have to search in metadata.json
        :param name: Name of the dataset
        :return:
        """
        return self.list({'folder': re.escape(name)})

    def _get_by_dir(self, directory):
        import glob

        path = glob.glob(os.path.join(self._replace_placeholders_with_wildcard(self.root_dir), directory))[0]

        metadata = self.get_file(path, META_FILE)
        pipeline = self.get_file(path, WORKFLOW_FILE)

        project = self.backend.project.get(metadata.get(Experiment.PROJECT_ID))
        dataset = self.backend.dataset.get(metadata.get(Experiment.DATASET_ID))

        # TODO only pass metadata / config etc to experiment creator. We shouldn't think about the structure of experiments here

        ex = Experiment(name=metadata.get("name"), description=metadata.get("description"), project=project,
                        dataset=dataset, metadata=metadata, pipeline=pipeline)
        return ex

    def to_folder_name(self, experiment):
        return experiment.name

    def _get_by_repo(self,repo,path=''):
        if repo is None:
            return None

        metadata = self.get_file(repo,META_FILE)
        pipeline = self.get_file(repo,WORKFLOW_FILE)

        project = self.backend.project.get(metadata.get(Experiment.PROJECT_ID))
        dataset = self.backend.dataset.get(metadata.get(Experiment.DATASET_ID))

        ex = Experiment(name= metadata.get("name"), description=metadata.get("description"), project=project,
                        dataset=dataset, metadata=metadata, pipeline=pipeline)
        return ex

    def put_progress(self, experiment, **kwargs):
        self.log("EXPERIMENT PROGRESS: {curr_value}/{limit}. phase={phase} \n".format(**kwargs))

    def update(self,experiment: Experiment, commit_message:str):
        add_and_commit(self.to_directory(experiment),message=commit_message)
        self.push_changes(commit_counter=3)
        #TODO

    def _put(self, experiment: Experiment, *args, directory, merge=False,**kwargs):

        # update experiment
        if merge:
            metadata = self.get_file(directory, META_FILE)
            if metadata is not None:
                # TODO this merge function should merge our changes into the already existing data and not the other
                # way around
                experiment.merge_metadata(metadata=metadata)
        if self.remote is not None:
            # TODO add a counter (of commits) or a timer for each push
            add_and_commit(directory,message="Adding unstaged changes in the repo")
            self.push_changes()
        else:
            self.write_file(directory, META_FILE, experiment.metadata)
            self.write_file(directory, WORKFLOW_FILE, experiment.pipeline, 'wb')
            add_and_commit(directory,message="Adding the metadata and the workflow of the experiment")
            self.parent.update(experiment.parent, src=experiment.name, url=self.get_repo_url(),
                               commit_message="Adding the experiment named {} repository to the tsrc file.".format(experiment.name))

        remove_cached(cache, experiment.id)