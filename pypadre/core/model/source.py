from pypadre.core.base import MetadataEntity


class Source(MetadataEntity):

    def validate(self, **kwargs):
        super().validate(**kwargs)

    path = None

    def __init__(self, **options):

        self.path = options.get('path', None)


    @property
    def path(self):
        return self._path
