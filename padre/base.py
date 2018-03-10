"""
Modul containing basic padre datastructures
"""

class MetadataEntity:

    def __init__(self, **metadata):
        self._metadata = dict(metadata)

    @property
    def createdAt(self):
        if "createdAt" in self._metadata:
            return self._metadata["createdAt"]
        else:
            return None

    @property
    def updatedAt(self):
        if "updatedAt" in self._metadata:
            return self._metadata["updatedAt"]
        else:
            return None

    @property
    def lastModifiedBy(self):
        if "lastModifiedBy" in self._metadata:
            return self._metadata["lastModifiedBy"]
        else:
            return None

    @property
    def createdBy(self):
        if "createdBy" in self._metadata:
            return self._metadata["createdBy"]
        else:
            return None
