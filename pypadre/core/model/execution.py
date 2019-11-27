import pyhash

from pypadre.core.base import ChildMixin
from pypadre.core.model.computation.run import Run
from pypadre.core.model.generic.custom_code import CodeManagedMixin
from pypadre.core.model.generic.i_executable_mixin import ValidateableExecutableMixin
from pypadre.core.model.generic.i_model_mixins import ProgressableMixin
from pypadre.core.model.generic.i_storable_mixin import StoreableMixin
from pypadre.core.util.utils import persistent_hash
from pypadre.core.validation.json_validation import make_model
from pypadre.pod.util.git_util import git_diff

execution_model = make_model(schema_resource_name='execution.json')


class Execution(CodeManagedMixin, StoreableMixin, ProgressableMixin, ValidateableExecutableMixin, ChildMixin):
    """
    A execution should save data about the running env and the version of the code on which it was run .
    An execution is linked to the version of the code being executed. The execution directory is the hash of the commit
    of the source code file.
    """
    EXPERIMENT_ID = "experiment_id"
    EXPERIMENT_NAME = "experiment_name"

    _runs = []

    @classmethod
    def _tablefy_register_columns(cls):
        super()._tablefy_register_columns()
        cls.tablefy_register_columns({'hash': 'hash'})

    def __init__(self, experiment, runs=None, pipeline=None, **kwargs):
        # Add defaults
        defaults = {}

        # Merge defaults
        metadata = {**defaults, **kwargs.pop("metadata", {}), **{self.EXPERIMENT_ID: experiment.id,
                                                                 self.EXPERIMENT_NAME: experiment.name}}

        metadata = {
            **{"id": str(kwargs.get("reference").id) + "-" + str(persistent_hash(experiment.id, algorithm=pyhash.city_64()))},
            **metadata}
        super().__init__(parent=experiment, model_clz=execution_model, metadata=metadata, **kwargs)

        if runs is not None:
            self._runs = runs

        self._pipeline = pipeline

    def _execute_helper(self, *args, **kwargs):
        self.send_put()
        run = Run(execution=self)
        self._runs.append(run)
        return run.execute(data=self.dataset, execution=self, **kwargs)

    @property
    def hash(self):
        return self.reference.id

    @property
    def experiment(self):
        return self.parent

    @property
    def dataset(self):
        return self.experiment.dataset

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def run(self):
        return self._runs

    @property
    def experiment_id(self):
        return self.metadata.get(self.EXPERIMENT_ID, None)

    def compare(self, execution):
        if self.reference.repo_type == self.reference.repository_identifier._RepositoryType.git:
            _version = self.reference.repository_identifier.version()
            __version = execution.reference.repository_identifier.version()
            path_to_ref = self.reference.repository_identifier.path
            return git_diff(_version, __version, path=path_to_ref)
        else:
            raise NotImplementedError()
