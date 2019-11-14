import importlib
from _py_abc import ABCMeta
from abc import abstractmethod
from typing import Callable

from ipython_genutils.py3compat import execfile

from pypadre.core.base import MetadataMixin
from pypadre.core.model.generic.i_executable_mixin import ExecuteableMixin
from pypadre.core.model.generic.i_storable_mixin import StoreableMixin
from pypadre.core.util.utils import _Const, persistent_hash
from pypadre.pod.util.git_util import get_repo


class CodeIdentifier:
    """
    This object identifies code by delivering a combination of retrieval informations.
    """

    @abstractmethod
    def __init__(self, type):
        self._type = type
        self._initialized = False

    @property
    def initialized(self):
        return self._initialized

    @initialized.setter
    def initialized(self, init: bool):
        self._initialized = init

    @abstractmethod
    def name(self):
        raise NotImplementedError()

    @abstractmethod
    def version(self):
        raise NotImplementedError()

    @abstractmethod
    def id_hash(self):
        raise NotImplementedError()

    @abstractmethod
    def meta(self):
        raise NotImplementedError()

    class _RepositoryType(_Const):
        pip = "pip"
        git = "git"

    @property
    def type(self):
        return self._type


class PipIdentifier(CodeIdentifier):
    """
    This objects is used to represent a pip managed code file.
    """

    VERSION = "version"
    PIP_PACKAGE = "pip_package"

    def __init__(self, pip_package, version):
        self._version = version
        self._pip_package = pip_package
        super().__init__(type=self._RepositoryType.pip)

        if False:
            # TODO look at installed packages and inform the user if some package is missing
            raise ValueError("Please install missing packages.")

    def name(self):
        return self._pip_package + ":" + self._version

    def version(self):
        return self._version

    def id_hash(self):
        return persistent_hash((self._pip_package, self._version))

    def meta(self):
        return {self.VERSION: self._version, self.PIP_PACKAGE: self._pip_package}

    @property
    def type(self):
        return self._type


class GitIdentifier(CodeIdentifier):
    """
    This object is used to manage a git managed code file. This should most likely be the mostly used variant of storage.
    """

    PATH = "path"
    GIT_HASH = "git_hash"

    def __init__(self, path=None, url=None, git_hash=None):
        self._url = url
        self._path = path
        self._git_hash = git_hash

        if self._git_hash is None:
            with get_repo(path=path, url=url) as _repo:
                if False:
                    # Todo check git repo state
                    raise ValueError("Git repository has uncommitted changes please commit.")
                if _repo is not None:
                    self._git_hash = _repo.head.object.hexsha
                else:
                    raise EnvironmentError("Couldn't extract hash of repo because repo couldn't be found.")

        super().__init__(type=self._RepositoryType.git)

    def name(self):
        return self._path + ":" + self._git_hash

    def id_hash(self):
        return persistent_hash((self._path, self._git_hash))

    def version(self):
        return self._git_hash

    def meta(self):
        return {self.PATH: self._path, self.GIT_HASH: self._git_hash}


# TODO in the future add additional code management repositories like for example maven or something else


class CodeMixin(StoreableMixin, MetadataMixin):
    """ Custom code to execute. """
    __metaclass__ = ABCMeta

    CODE_TYPE = "code_type"
    REPOSITORY_TYPE = "repository_type"
    CODE_CLASS = "code_class"
    NAME = "name"
    IDENTIFIER = "identifier"

    class _CodeType(_Const):
        function = "function"
        package = "package"
        python_file = "python_file"
        file = "file"

    def __init__(self, *, type, identifier: CodeIdentifier, **kwargs):
        """
        This is the base class of all custom code we can reference in the padre app.
        Managed code has to be within a git repository (url + commit_hash) or a pip package
        (name + version + repo=default).

        We can provide repositories for people or hold a reference to an existing repository or hold an reference to
        a pip package.

        Following types of to be imported packages should be supported:
        - Pickled functions in a repository (import with pickle / dill)
        - Source code to be imported by giving a file path / name / package and a function or variable name (import with importlib)
        - Source code of arbitrary files / binaries etc to be called via system (subprocess or similar)

        :param path: Path to the local git repository
        :param url: Path to the remote git repository
        :param git_hash: Git hash encoding the version of the code we want to reference.
        :param kwargs:
        """
        # TODO Add defaults
        defaults = {}

        self._identifier = identifier
        self._type = type

        # TODO Constants into ontology stuff
        # Merge defaults TODO some file metadata extracted from the path
        metadata = {**defaults,
                    **{self.NAME: identifier.name(), self.REPOSITORY_TYPE: identifier.type,
                       self.IDENTIFIER: identifier.meta(),
                       CodeMixin.CODE_TYPE: self._type, CodeMixin.CODE_CLASS: str(self.__class__.__name__)},
                    **kwargs.pop("metadata", {})}
        super().__init__(metadata=metadata, **kwargs)

    @abstractmethod
    def _call(self, ctx, **kwargs):
        raise NotImplementedError()

    def call(self, **kwargs):
        parameters = kwargs.pop("parameters", {})
        # kwargs are the padre context to be used
        return self._call(kwargs, **parameters)

    @property
    def identifier(self):
        return self._identifier

    @property
    def name(self):
        return self._identifier.name()

    @property
    def repo_type(self):
        return self.metadata.get(self.REPOSITORY_TYPE)

    @property
    def code_type(self):
        return self.metadata.get(self.CODE_TYPE)


# Code should be given by one of the following ways: A file (local, remote), a function to be persisted, a function
# on the environment

class PythonPackage(CodeMixin):
    VARIABLE = "variable"
    PACKAGE = "package"

    def __init__(self, *, package, variable, identifier: CodeIdentifier, **kwargs):
        """

        :param cmd: The command to be executed for given identifier. This could be something like: "java ./experiment.java"
        :param identifier: This is the reference to the related git or pip repository
        :param package: This the package from which to load the function
        :param variable: This is the variable name which has to be imported
        :param kwargs:
        """
        self._variable = variable
        self._package = package
        metadata = {**{self.PACKAGE: self._package, self.VARIABLE: self._variable, "id": persistent_hash((self._variable, self._package, identifier.id_hash()))},
                    **kwargs.pop("metadata", {})}
        super().__init__(identifier=identifier, type=self._CodeType.package, metadata=metadata, **kwargs)

    def _call(self, ctx, **kwargs):
        variable = getattr(importlib.import_module(self._package), self._variable)
        if isinstance(variable, Callable):
            return variable(ctx, **kwargs)
        elif isinstance(variable, ExecuteableMixin):
            return variable.execute(**ctx, parameters=kwargs)


class PythonFile(CodeMixin):
    VARIABLE = "variable"
    PACKAGE = "package"
    PATH = "path"

    def __init__(self, *, git_path, package, variable, identifier: CodeIdentifier, **kwargs):
        """
        Call of a python file.
        :param identifier: This is the reference to the related git or pip repository
        :param git_path: This the path to the file or package name
        :param package: Name of the package
        :param variable: Variable name
        :param kwargs:
        """

        self._variable = variable
        self._package = package
        self._path = git_path
        metadata = {**{self.PACKAGE: self._package, self.VARIABLE: self._variable, self.PATH: self._path,
                       "id": persistent_hash((self._variable, self._package, self._path, identifier.id_hash()))},
                    **kwargs.pop("metadata", {})}
        super().__init__(identifier=identifier, type=self._CodeType.python_file, metadata=metadata, **kwargs)

    def _call(self, ctx, **kwargs):
        if self._variable is None:
            # TODO get results
            return execfile(self._path)

        # Else append to import path and then import like a normal python package
        import sys
        import os
        sys.path.append(os.path.abspath(self._path))

        variable = getattr(importlib.import_module(self._package), self._variable)
        if isinstance(variable, Callable):
            return variable(ctx, **kwargs)
        elif isinstance(variable, ExecuteableMixin):
            return variable.execute(**ctx, parameters=kwargs)


class GenericCall(CodeMixin):
    CMD = "cmd"

    def __init__(self, cmd, *, identifier: CodeIdentifier, **kwargs):
        """
        Generic system call. This could be everything. We need to be able to read results of something like stdout.
        :param cmd: The command to be executed for given identifier. This could be something like:
        "java ./experiment.java"
        :param identifier: Identifier of the pip or git repository
        :param kwargs:
        """
        self._cmd = cmd
        metadata = {**{self.CMD: self._cmd, "id": persistent_hash((self._cmd, identifier.id_hash()))},
                    **kwargs.pop("metadata", {})}
        super().__init__(identifier=identifier, type=self._CodeType.file, metadata=metadata, **kwargs)

    def _call(self, ctx, **kwargs):
        raise NotImplementedError()
        # TODO get results
        # noinspection PyUnreachableCode
        return os.system(self._cmd)


class Function(CodeMixin):
    """
    Simple function holder
    """

    def __init__(self, *, fn: Callable, identifier: CodeIdentifier, transient=False, **kwargs):
        """
        This is a simple function holder to be pickled in a repository. In general don't use it if you can use a file
        instead.
        (Loss of readability and comparability on the git level of the management)
        :param fn: Fn to be pickled and executed
        :param identifier: Identifier of the git repository in which the function should be stored
        :param kwargs:
        """
        self._fn = fn
        self._transient = transient
        metadata = {**{"id": persistent_hash((fn.__name__, identifier.id_hash()))},
                    **kwargs.pop("metadata", {})}
        super().__init__(identifier=identifier, type=self._CodeType.function, metadata=metadata, **kwargs)

    @property
    def fn(self):
        # TODO we could dill this serialize
        # https://medium.com/@emlynoregan/serialising-all-the-functions-in-python-cd880a63b591
        # or write the maximum of possible information and warn the user about no possibility to reload
        return self._fn

    def _call(self, ctx, **kwargs):
        return self.fn(ctx, **kwargs)

    def send_put(self, **kwargs):
        if not self._transient:
            super().send_put(**kwargs)
        else:
            super().send_info(message="Put called on transient function " + str(self))
