import errno
import glob
import os
import re
import shutil

from pypadre.core.model.code.code_mixin import CodeMixin, PythonPackage, PythonFile, GenericCall, \
    GitIdentifier, CodeIdentifier, PipIdentifier, Function
from pypadre.pod.backend.i_padre_backend import IPadreBackend
from pypadre.pod.repository.i_repository import ICodeRepository
from pypadre.pod.repository.local.file.generic.i_file_repository import File
from pypadre.pod.repository.local.file.generic.i_git_repository import IGitRepository
from pypadre.pod.repository.local.file.project_code_repository import CODE_FILE
from pypadre.pod.repository.serializer.serialiser import JSonSerializer


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
# CODE_FILE = File("code.bin", DillSerializer)


class CodeFileRepository(IGitRepository, ICodeRepository):

    @staticmethod
    def placeholder():
        return '{CODE_ID}'

    def __init__(self, backend: IPadreBackend):
        super().__init__(root_dir=os.path.join(backend.root_dir, NAME), backend=backend)

    def _get_by_dir(self, directory):
        path = glob.glob(os.path.join(self._replace_placeholders_with_wildcard(self.root_dir), directory))[0]

        metadata = self.get_file(path, META_FILE)

        identifier_type = metadata.get(CodeMixin.REPOSITORY_TYPE)
        identifier_data = metadata.get(CodeMixin.IDENTIFIER)

        identifier = None
        if identifier_type == CodeIdentifier._RepositoryType.pip:
            version = identifier_data.get(PipIdentifier.VERSION)
            pip_package = identifier_data.get(PipIdentifier.PIP_PACKAGE)
            identifier = PipIdentifier(version=version, pip_package=pip_package)

        if identifier_type == CodeIdentifier._RepositoryType.git:
            path = identifier_data.get(GitIdentifier.PATH)
            git_hash = identifier_data.get(GitIdentifier.GIT_HASH)
            identifier = GitIdentifier(path=path, git_hash=git_hash)

        if identifier is None:
            raise ValueError("Identifier is not present in the meta information of code object in directory " + directory)

        if metadata.get(CodeMixin.CODE_TYPE) == str(CodeMixin._CodeType.function):
            fn_dir = glob.glob(os.path.join(self._replace_placeholders_with_wildcard(self.root_dir),
                                   os.path.abspath(os.path.join(directory, '..', 'function'))))[0]
            fn = self.get_file(fn_dir, CODE_FILE)
            code = Function(fn=fn, metadata=metadata, identifier=identifier)

        elif metadata.get(CodeMixin.CODE_TYPE) == str(CodeMixin._CodeType.package):
            code = PythonPackage(metadata=metadata, package=metadata.get(PythonPackage.PACKAGE), variable=metadata.get(PythonPackage.VARIABLE), identifier=identifier)

        elif metadata.get(CodeMixin.CODE_TYPE) == str(CodeMixin._CodeType.python_file):
            code = PythonFile(metadata=metadata, path=metadata.get(PythonFile.PATH), package=metadata.get(PythonFile.PACKAGE), variable=metadata.get(PythonFile.VARIABLE), identifier=identifier)

        elif metadata.get(CodeMixin.CODE_TYPE) == str(CodeMixin._CodeType.file):
            code = GenericCall(metadata=metadata, cmd=metadata.get(GenericCall.CMD), identifier=identifier)
        else:
            raise NotImplementedError(metadata.get(CodeMixin.CODE_TYPE) + " couldn't load from type.")

        return code

    def to_folder_name(self, code):
        # TODO only name for folder okay? (maybe a uuid, a digest of a config or similar?)
        return str(code.id)

    def list(self, search, offset=0, size=100):
        if hasattr(search, "name"):
            # Shortcut because we know name is the folder name. We don't have to search in metadata.json
            name = search.pop("name")
            search[self.FOLDER_SEARCH] = re.escape(name)
        return super().list(search, offset, size)

    def _put(self, obj, *args, directory: str, **kwargs):
        code = obj

        if isinstance(code, Function):
            # TODO fn repository
            self.write_file(os.path.abspath(os.path.join(directory, '..', 'function')), CODE_FILE, code.fn, mode="wb")

        self.write_file(directory, META_FILE, code.metadata)


        # if store_code:
        #     if isinstance(code, CodeFile):
        #         code_dir = os.path.join(directory, "code")
        #         if code.file is not None:
        #             if not os.path.exists(code_dir):
        #                 os.mkdir(code_dir)
        #             copy(os.path.join(code.path, code.file), os.path.join(directory, "code", code.file))
        #         else:
        #             copy(code.path, code_dir)

    # def get_code_hash(self, obj=None, path=None, init_repo=False, **kwargs):
    #
    #     code_hash = git_hash(path=path)
    #     if code_hash is None and init_repo is True:
    #         # if there is no repository present in the path, but the user wants to create a repo then
    #         # Create a repo
    #         # Add any untracked files and commit those files
    #         # Get the code_hash of the repo
    #         # TODO give git an id and hold some reference in workspace???
    #         dir_path = os.path.dirname(path)
    #         create_repo(dir_path)
    #         add_and_commit(dir_path)
    #         code_hash = git_hash(path=dir_path)
    #
    #     if obj is not None:
    #         obj.set_hash(code_hash)
