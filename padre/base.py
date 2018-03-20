"""
Modul containing basic padre datastructures
"""
import sys



class PadreLogger:
    """
    Base class for logging output, warnings and errors
    """

    def warn(self, condition, source, message):
        if not condition:
            sys.stderr.write(str(source)+":\t"+message+"\n")


    def error(self, condition, source, message):
        if not condition:
            raise ValueError(str(source)+":\t"+message)



default_logger = PadreLogger()

class MetadataEntity:
    """
    Base object for entities that manage metadata
    """


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
