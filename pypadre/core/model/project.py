from importlib import resources

from jsonpickle import json

from pypadre.base import MetadataEntity
from pypadre.validation import Validateable


class Project(Validateable, MetadataEntity):
    """ A project should group experiments """

    def handle_failure(self, e):
        pass

    _id = None
    _metadata = None

    def __init__(self, **options):
        # Validate input types

        with resources.open_text('pypadre.res.schema', 'project.json') as f:
            schema = json.loads(f.read())

        super(Project, self).__init__(id_=options.pop("id", None), schema=schema, **options)

        self._experiments = []
        self._sub_projects = []

    def get(self, key):
        if key == 'id':
            return self.name

        else:
            return self.__dict__.get(key, None)
