from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import ICodeRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.local.file.generic.i_git_repository import IGitRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer, PickleSerializer

NAME = "code"

META_FILE = File("metadata.json", JSonSerializer)
CODE_FILE = File("code.bin", PickleSerializer)


class CodeFileRepository(IGitRepository, ICodeRepository):

    @staticmethod
    def placeholder():
        return '{CODE_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(name=NAME, backend=backend)

    def get_by_dir(self, directory):
        raise NotImplementedError()

    def _put(self, obj, *args, directory: str, store_results=False, merge=False, **kwargs):
        code = obj
        self.write_file(directory, META_FILE, code.metadata)
        self.write_file(directory, CODE_FILE, code.get_bin(), mode="wb")
