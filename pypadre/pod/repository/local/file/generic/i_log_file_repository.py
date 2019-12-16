import os
from abc import abstractmethod, ABCMeta
from datetime import datetime

from pypadre.pod.repository.generic.i_repository_mixins import ILogRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import IFileRepository

FILE_NAME = "padre.log"


class ILogFileRepository(IFileRepository, ILogRepository):
    """ This is the abstract implementation of a file backend which adds logging functionality"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **kwargs):
        self._file = None
        super().__init__(**kwargs)

    def __del__(self):
        if self._file is not None:
            self._file.close()
    """
    def log(self, msg):
        self._write(msg)

    def _write(self, message):
        # TODO close file
        if self._file is None:
            self._file = open(os.path.join(self.root_dir, FILE_NAME), "a")
        self._file.write(message)
    """

    @property
    def time(self):
        return str(datetime.now())

    def log_info(self, message="", **kwargs):
        self.log(message=self.time + ": " + "INFO: " + message + "\n", **kwargs)

    def log_warn(self, message="", **kwargs):
        self.log(message=self.time + ": " + "WARN: " + message + "\n", **kwargs)

    def log_error(self, message="", **kwargs):
        self.log(message=self.time + ": " + "ERROR: " + message + "\n", **kwargs)

    def log(self, message, **kwargs):
        if self._file is None:
            path = os.path.join(self.root_dir, "padre.log")
            if not os.path.exists(self.root_dir):
                os.makedirs(self.root_dir)

            self._file = open(path, "a")
        self._file.write(message)
        self._file.flush()

