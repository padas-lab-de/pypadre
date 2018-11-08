"""
Modul containing basic padre datastructures
"""
import json
import sys
from datetime import datetime


class PadreLogger:
    """
    Base class for logging output, warnings and errors
    """
    _file = None

    def warn(self, condition, source, message):
        if not condition:
            sys.stderr.write(str(source) + ":\t" + message + "\n")
            if self._file is not None:
                pass
                self._file.write("WARN:" + str(datetime.now())[:-3] + " " + str(source) + ":\t" + message + "\n")

    def error(self, condition, source, message):
        if not condition:
            if self._file is not None:
                #._file.write("ERROR:" + str(datetime.now())[:-3] + " " + str(source) + ":\t" + message + "\n")
                self._file.close()
                self._file = None

            raise ValueError(str(source)+":\t"+message)

    def log(self, source, message, padding=""):

        if self._file is not None:
            self._file.write("INFO:" + str(datetime.now())[:-3] + " " + padding+str(source) + ":\t" + message + "\n")
        sys.stdout.write(padding+str(source) + ":\t" + message + "\n")

    def open_log_file(self, path=None):

        import os

        if path is None:
            return None

        if os.path.isdir(path):
            if self._file is not None:
                self._file.close()

            self._file = open(os.path.join(path, "log.txt"), "a")

        else:
            if self._file is not None:
                self._file.close()

            self._file = None

    def close_log_file(self):

        if self._file is not None:
            self._file.close()
            self._file = None

    def log_start_experiment(self, experiment):
        """
        This function handles the start of an experiment.

        :param experiment: Experiment object to be logged

        :return:
        """
        pass

default_logger = PadreLogger()
""

class MetadataEntity:
    """
    Base object for entities that manage metadata. A MetadataEntity manages and id and a dict of metadata.
    The metadata should contain all necessary non-binary data to describe an entity.
    """
    def __init__(self, id_=None, **metadata):
        self._metadata = dict(metadata)

        if id_==None:
            if metadata.__contains__("openml_id"):
                self._id=metadata["openml_id"]
            else:
                self._id=None
        else:
            self._id = id_


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