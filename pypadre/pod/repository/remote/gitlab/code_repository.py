import errno
import glob
import os
import re
import shutil

from pypadre.core.model.code.code_file import CodeFile
from pypadre.core.model.code.icode import ICode, Function
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import ICodeRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.local.file.generic.i_git_repository import IGitRepository
from pypadre.pod.repository.remote.gitlab.generic.gitlab import GitLabRepository
from pypadre.pod.repository.serializer.serialiser import JSonSerializer, DillSerializer
from pypadre.pod.util.git_util import git_hash, create_repo, add_and_commit


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


class CodeGitlabRepository(GitLabRepository, ICodeRepository):

    @staticmethod
    def placeholder():
        return '{CODE_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(root_dir=os.path.join(backend.root_dir, NAME), gitlab_url=backend.url, token=backend.token
                         , backend=backend)
        self._group = self.get_group(name=NAME)

    def get_by_dir(self, directory):
        path = glob.glob(os.path.join(self._replace_placeholders_with_wildcard(self.root_dir), directory))[0]

        metadata = self.get_file(path, META_FILE)

        # TODO what about inherited classes
        if metadata.get(ICode.CODE_CLASS) == str(Function):
            fn = self.get_file(path, CODE_FILE)
            code = Function(fn=fn, metadata=metadata)

        elif metadata.get(ICode.CODE_CLASS) == str(CodeFile):
            code = CodeFile(path=metadata.path, cmd=metadata.cmd, file=metadata.get("file", None))

        else:
            raise NotImplementedError()

        return code

    def get_by_repo(self,repo):

        metadata = self.get_file(repo, META_FILE)
        if metadata.get(ICode.CODE_CLASS) == str(Function):
            fn = self.get_file(repo, CODE_FILE)
            code = Function(fn=fn, metadata=metadata)

        elif metadata.get(ICode.CODE_CLASS) == str(CodeFile):
            code = CodeFile(path=metadata.path, cmd=metadata.cmd, file=metadata.get("file", None))

        else:
            raise NotImplementedError()

        return code

    def to_folder_name(self, code):
        # TODO only name for folder okay? (maybe a uuid, a digest of a config or similar?)
        return code.name

    def get_by_name(self, name):
        """
        Shortcut because we know name is the folder name. We don't have to search in metadata.json
        :param name: Name of the dataset
        :return:
        """
        return self.list({'folder': re.escape(name)})

    def _put(self, obj, *args, directory: str, store_code=False, **kwargs):
        code = obj
        self.write_file(directory, META_FILE, code.metadata)
        add_and_commit(directory,message="Adding the experiment's source code metadata to the code generic")
        if store_code:
            if isinstance(code, Function):
                self.write_file(directory, CODE_FILE, code.fn, mode="wb")

            if isinstance(code, CodeFile):
                code_dir = os.path.join(directory, "code")
                if code.file is not None:
                    if not os.path.exists(code_dir):
                        os.mkdir(code_dir)
                    copy(os.path.join(code.path, code.file), os.path.join(directory, "code", code.file))
                else:
                    copy(code.path, code_dir)
            add_and_commit(directory, message="Adding the serialized source code  to the code generic")
        if self.remote is not None:
            self.push_changes()

    def get_code_hash(self, obj=None, path=None, init_repo=False, **kwargs):
        code_hash = git_hash(path=path)
        if code_hash is None and init_repo is True:
            # if there is no generic present in the path, but the user wants to create a repo then
            # Create a repo
            # Add any untracked files and commit those files
            # Get the code_hash of the repo
            # TODO give git an id and hold some reference in workspace???
            dir_path = os.path.dirname(path)
            create_repo(dir_path)
            add_and_commit(dir_path)
            code_hash = git_hash(path=dir_path)

        if obj is not None:
            obj.set_hash(code_hash)
