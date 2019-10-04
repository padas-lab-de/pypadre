from _py_abc import ABCMeta
from abc import abstractmethod

from pypadre.core.base import MetadataEntity
from pypadre.core.model.code.code_file import CodeFile


class PythonFile(CodeFile):

    def __init__(self, **kwargs):
        # TODO Add defaults
        defaults = {}

        # Merge defaults TODO some fn metadata extracted from the fn
        metadata = {**defaults, **kwargs.pop("metadata", {})}
        self._path = kwargs.pop('path', None)
        self._function = kwargs.pop('function', None)

        super().__init__(metadata=metadata, **kwargs)

    def call(self, args):
        # TODO subprocess.Popen(), os.system() or execfile()
        # Add code to specify entry point

        import sys
        import os

        # append the directory to the system path for importing the file
        directory_path = os.path.dirname(self.path)
        sys.path.append(directory_path)

        # Get the filename with out the extension for importing
        package_name = str(os.path.basename(self.path)).split(sep='.')[0]
        func = getattr(__import__(package_name), self._function)
        return func(args)
