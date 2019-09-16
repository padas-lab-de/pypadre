from importlib import resources

from jsonpickle import json

from pypadre.pod.base import MetadataEntity
from pypadre.pod.printing.tablefyable import Tablefyable
from pypadre.pod.validation import Validateable


class Project(Validateable, MetadataEntity, Tablefyable):
    """ A project should group experiments """

    @classmethod
    def tablefy_register_columns(cls):
        # TODO fill with properties to extract for table
        cls._tablefy_register_columns({})

    def __init__(self, **options):
        # Validate input types
        # TODO alternative See https://rhettinger.wordpress.com/2011/05/26/super-considered-super/
        Validateable.__init__(self, schema_resource_name='project.json', **options)
        MetadataEntity.__init__(self, **options)
        Tablefyable.__init__(self)

        self._experiments = []
        self._sub_projects = []

    def handle_failure(self, e):
        pass

    def get(self, key):
        if key == 'id':
            return self.name

        else:
            return self.__dict__.get(key, None)
