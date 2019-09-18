import os
from abc import abstractmethod, ABCMeta

from pypadre.pod.repository.generic.i_file_repository import IFileRepository
from pypadre.pod.repository.generic.i_repository_mixins import ILogRepository


class ILogFileRepository(IFileRepository, ILogRepository):
    """ This is the abstract implementation of a file backend which adds logging functionality"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, parent, name, **kwargs):
        super().__init__(root_dir=, backend=, parent=parent, name=name)

    FILE_NAME = "log.txt"

    def log(self, msg):
        self._write(msg)

    def _write(self, message):
        if self._file is None:
            self._file = open(os.path.join(self.root_dir, self.FILE_NAME), "a")
        self._file.write(message)

