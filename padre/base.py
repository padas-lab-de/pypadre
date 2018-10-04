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

                self._file.write("WARN:" + str(datetime.now())[:-3] + " " + str(source) + ":\t" + message + "\n")

    def error(self, condition, source, message):
        if not condition:
            if self._file is not None:
                self._file.write("ERROR:" + str(datetime.now())[:-3] + " " + str(source) + ":\t" + message + "\n")
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




default_logger = PadreLogger()
""


class ResultLogger:
    """
    Base class for logging the results. Currently, it is assumed to be a separate entity to
    PadreLogger.
    """
    _log_dir = ''

    def log_result(self, result):
        """
        This function logs the result to a file within the run folder
        :param result: A JSON Serializable object
        :return: None
        """
        import os
        with open(os.path.join(self._log_dir, "results.json"), 'a') as f:
            f.write(json.dumps(result))

    def set_log_directory(self, dir):
        """
        This function sets the path of the log
        :param dir: The path to the directory where the results are to be written
        :return: None
        """
        if dir is not None:
            self._log_dir = dir

    def log_metrics(self,  metrics):
        """
        This function logs the classification metrics like
        precision, recall, accuracy etc
        :param metrics: The JSON serializable object containing the different metrics of that split
        :return: None
        """
        import os
        with open(os.path.join(self._log_dir, "metrics.json"), 'a') as f:
            f.write(json.dumps(metrics))


result_logger = ResultLogger()


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