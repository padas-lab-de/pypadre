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
            sys.stderr.write(str(source) + ":\t" + message + "\n")

    def error(self, condition, source, message):
        if not condition:
            raise ValueError(str(source)+":\t"+message)

    def log(self, source, message, padding=""):
        sys.stdout.write(padding+str(source) + ":\t" + message + "\n")

default_logger = PadreLogger()
""


class MetadataEntity:
    """
    Base object for entities that manage metadata. A MetadataEntity manages and id and a dict of metadata.
    The metadata should contain all necessary non-binary data to describe an entity.
    """
    def __init__(self, id_=None, **metadata):
        self._id = id_
        self._metadata = dict(metadata)

    @property
    def id(self):
        """
        returns the unique id of the data set. Data sets will be managed on the basis of this id
        :return: string
        """
        return self._id

    @id.setter
    def id(self, _id):
        """
        used for updating the id after the undlerying repository has assigned one
        :param _id: id, ideally an url
        :return:
        """
        self._id = _id

    @property
    def name(self):
        """
        returns the name of this object, which is expected in field "name" of the metadata. If this field does not
        exist, the id is returned
        :return:
        """
        if self._metadata and "name" in self._metadata:
            return self._metadata["name"]
        else:
            return self._id

    @name.setter
    def name(self, name):
        self._metadata["name"] = name

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

    @property
    def metadata(self):
        return self._metadata