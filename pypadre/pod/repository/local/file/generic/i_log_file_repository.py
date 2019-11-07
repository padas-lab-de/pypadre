import os
from abc import abstractmethod, ABCMeta

from pypadre.pod.repository.generic.i_repository_mixins import ILogRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import IFileRepository

FILE_NAME = "padre.log"


class ILogFileRepository(IFileRepository, ILogRepository):
    """ This is the abstract implementation of a file backend which adds logging functionality"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log(self, msg):
        self._write(msg)

    def _write(self, message):
        # TODO close file
        if self._file is None:
            self._file = open(os.path.join(self.root_dir, FILE_NAME), "a")
        self._file.write(message)

