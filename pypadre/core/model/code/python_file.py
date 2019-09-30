from _py_abc import ABCMeta
from abc import abstractmethod

from pypadre.core.base import MetadataEntity
from pypadre.core.model.code.code_file import CodeFile


class PythonFile(CodeFile):

    def call(self):
        # TODO subprocess.Popen(), os.system() or execfile()
        pass
