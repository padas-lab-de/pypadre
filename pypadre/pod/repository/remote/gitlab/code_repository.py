import errno
import os
import shutil
import tempfile

from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import ICodeRepository
from pypadre.pod.repository.local.file.code_repository import CodeFileRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.remote.gitlab.generic.gitlab import GitLabRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer, DillSerializer
from pypadre.pod.util.git_util import add_and_commit, get_repo


def copy(src, dest):
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)


NAME = "code"

META_FILE = File("metadata.json", JSonSerializer)
CODE_FILE = File("code.bin", DillSerializer)


# TODO rework!

class CodeGitlabRepository(GitLabRepository, ICodeRepository):

    @staticmethod
    def placeholder():
        return '{CODE_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(root_dir=os.path.join(backend.root_dir, NAME), gitlab_url=backend.url, token=backend.token
                         , backend=backend)
        self._file_backend = CodeFileRepository(backend=backend)
        self._group = self.get_group(name=NAME)

    def _get_by_dir(self, directory):
        return self._file_backend._get_by_dir(directory)

    def _get_by_repo(self, repo, path=''):
        # TODO rework
        with tempfile.TemporaryDirectory(suffix="code") as temp_dir:
            temp_local_repo = get_repo(path=temp_dir, url=self.url_oauth(self.get_repo_url(repo)))
            metadata = self._file_backend.get_file(temp_dir, META_FILE)
            return self._file_backend._create_object(metadata, directory=temp_dir, root_dir=temp_dir)

    def to_folder_name(self, code):
        # TODO only name for folder okay? (maybe a uuid, a digest of a config or similar?)
        return self._file_backend.to_folder_name(code)

    def _put(self, obj, *args, directory: str, local=True, **kwargs):
        # TODO rework!
        if local:
            self._file_backend._put(obj, *args, directory=directory, **kwargs)
            add_and_commit(directory, message="Adding the experiment's source code metadata to the code generic")
        if self.has_remote_backend(obj):
            self.push_changes()
        # if store_code:
        #     if isinstance(code, Function):
        #         self.write_file(directory, CODE_FILE, code.fn, mode="wb")
        #
        #     if isinstance(code, CodeFile):
        #         code_dir = os.path.join(directory, "code")
        #         if code.file is not None:
        #             if not os.path.exists(code_dir):
        #                 os.mkdir(code_dir)
        #             copy(os.path.join(code.path, code.file), os.path.join(directory, "code", code.file))
        #         else:
        #             copy(code.path, code_dir)
        #     add_and_commit(directory, message="Adding the serialized source code  to the code generic")
        # if self.remote is not None:
        #     self.push_changes()
