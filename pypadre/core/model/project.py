from pypadre.core.base import MetadataMixin
from pypadre.core.model.generic.custom_code import CodeManagedMixin
from pypadre.core.model.generic.i_executable_mixin import ValidateableExecutableMixin
from pypadre.core.model.generic.i_model_mixins import ProgressableMixin
from pypadre.core.model.generic.i_storable_mixin import StoreableMixin
from pypadre.core.validation.json_validation import make_model

project_model = make_model(schema_resource_name='project.json')


class Project(CodeManagedMixin, StoreableMixin, ProgressableMixin, ValidateableExecutableMixin, MetadataMixin):
    """ A project should group experiments """

    def __init__(self, name="Default project", description="Default project description", experiments=None, sub_projects=None, **kwargs):
        # Add defaults
        defaults = {}

        # Merge defaults
        metadata = {**defaults, **kwargs.pop("metadata", {}), **{"id": name, "name": name, "description": description}}

        super().__init__(model_clz=project_model, metadata=metadata, **kwargs)

        if experiments is None:
            experiments = []
        if sub_projects is None:
            sub_projects = []
        self._experiments = experiments
        self._sub_projects = sub_projects

    def get(self, key):
        if key == 'id':
            return self.name

        else:
            return self.__dict__.get(key, None)

    def _execute_helper(self, experiment_pipeline_parameters: dict, **kwargs):
        return {
            experiment: experiment.execute(pipeline_parameters=experiment_pipeline_parameters.get(experiment.id),
                                           **kwargs)
            for experiment in self._experiments}
