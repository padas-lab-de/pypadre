from pypadre.core.base import MetadataEntity
from pypadre.core.printing.tablefyable import Tablefyable


class Project(MetadataEntity, Tablefyable):
    """ A project should group experiments """

    @classmethod
    def _tablefy_register_columns(cls):
        # TODO fill with properties to extract for table
        cls.tablefy_register_columns({})

    def __init__(self, **options):
        super().__init__(schema_resource_name='project.json', metadata=options, **options)

        self._experiments = []
        self._sub_projects = []

    def handle_failure(self, e):
        pass

    def get(self, key):
        if key == 'id':
            return self.name

        else:
            return self.__dict__.get(key, None)

    def execute(self, **kwargs):
        # TODO args per experiment
        return {experiment: experiment.execute(**kwargs) for experiment in self._experiments}
