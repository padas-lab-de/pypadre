import os
from abc import abstractmethod, ABCMeta

from pypadre.pod.backend.interfaces.backend.generic.i_base_file_backend import FileBackend
from pypadre.pod.backend.interfaces.backend.generic.i_log_backend import ILogBackend


class LogFileBackend(FileBackend, ILogBackend):
    """ This is the abstract implementation of a file backend which adds logging functionality"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, parent, name, **kwargs):
        super().__init__(parent=parent, name=name)

    FILE_NAME = "log.txt"

    def log(self, msg):
        self._write(msg)

    def _write(self, message):
        if self._file is None:
            self._file = open(os.path.join(self.root_dir, self.FILE_NAME), "a")
        self._file.write(message)

