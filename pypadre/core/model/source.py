from collections import OrderedDict

from pypadre.pod.base import MetadataEntity
from pypadre.pod.eventhandler import assert_condition, trigger_event
from pypadre.pod.printing.tablefyable import Tablefyable


class Source(MetadataEntity):

    def validate(self):
        pass

    path = None

    def __init__(self, **options):

        self.path = options.get('path', None)


    @property
    def path(self):
        return self._path
