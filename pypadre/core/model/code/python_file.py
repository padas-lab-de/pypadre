# from pypadre.core.model.code.code_file import CodeFile
#
#
# class PythonFile(CodeFile):
#
#     def __init__(self, **kwargs):
#         # TODO Add defaults
#         defaults = {}
#
#         # TODO overwrite with imports?
#         # TODO Constants into ontology stuff
#         # Merge defaults TODO some file metadata extracted from the path
#         metadata = {**defaults, **kwargs.pop("metadata", {})}
#         self._path = kwargs.pop('path', None)
#         self._function = kwargs.pop('function', None)
#
#         super().__init__(metadata=metadata, **kwargs)
#
#     @property
#     def path(self):
#         return self._path
#
#     @property
#     def function(self):
#         return self._function
#
#     def call(self, args):
#
#         # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
#         # https://stackoverflow.com/questions/3606202/how-can-i-import-a-python-module-function-dynamically
#
#         import sys
#         import os
#
#         # append the directory to the system path for importing the file
#         directory_path = os.path.dirname(self.path)
#         sys.path.append(directory_path)
#
#         # Get the filename with out the extension for importing
#         package_name = str(os.path.basename(self.path)).split(sep='.')[0]
#         func = getattr(__import__(package_name), self._function)
#
#         # TODO finish me
#         # Call function with passed arguments and return result
#         return func(args)
