import glob
import os

from pypadre.core.model.code.code_mixin import Function
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import ICodeRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File, IChildFileRepository
from pypadre.pod.repository.local.file.generic.i_git_repository import IGitRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer, DillSerializer

NAME = "project_code"

META_FILE = File("metadata.json", JSonSerializer)
CODE_FILE = File("code.bin", DillSerializer)


class CodeFileRepository(IChildFileRepository, IGitRepository, ICodeRepository):

    @staticmethod
    def placeholder():
        return '{PROJECT_CODE_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(root_dir=os.path.join(backend.root_dir, NAME), backend=backend)

    def _get_by_dir(self, directory):
        path = glob.glob(os.path.join(self._replace_placeholders_with_wildcard(self.root_dir), directory))[0]
        
        # TODO implement generic for other code types

    def _put(self, obj, *args, directory: str, store_results=False, merge=False, **kwargs):
        code = obj
        self.write_file(directory, META_FILE, code.metadata)

        if code.__class__ is Function:
            self.write_file(directory, CODE_FILE, code.fn, mode="wb")

        # TODO implement generic for other code types
