from pypadre.core.base import MetadataEntity
from pypadre.core.model.generic.custom_code import ICodeManagedObject
from pypadre.core.model.generic.i_executable_mixin import IExecuteable
from pypadre.core.model.generic.i_model_mixins import IStoreable, IProgressable
from pypadre.core.printing.tablefyable import Tablefyable


class Project(ICodeManagedObject, IStoreable, IExecuteable, IProgressable, MetadataEntity, Tablefyable):
    """ A project should group experiments """

    @classmethod
    def _tablefy_register_columns(cls):
        # TODO fill with properties to extract for table
        cls.tablefy_register_columns({})

    @classmethod
    def from_schema(cls, *, metadata: ProjectSchema):
        return cls(name=metadata.name, metadata=metadata.description)

    def __init__(self, name, description, experiments=None, sub_projects=None, **kwargs):
        # Add defaults
        defaults = {"name": "default project name", "description": "This is the default project."}

        # Merge defaults
        metadata = {**defaults, **kwargs.pop("metadata", {}), **{"name": name, "description": description}}

        super().__init__(schema_resource_name='project.json', metadata=metadata, **kwargs)

        if experiments is None:
            experiments = []
        if sub_projects is None:
            sub_projects = []
        self._experiments = experiments
        self._sub_projects = sub_projects

    def handle_failure(self, e):
        pass

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
