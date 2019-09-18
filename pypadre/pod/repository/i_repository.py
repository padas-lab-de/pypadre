from abc import ABCMeta, abstractmethod

from pypadre.pod.repository.generic.i_repository_mixins import ILogRepository, IProgressable, IStoreable, IRepository, \
    ISearchable


class IMetricRepository(IRepository, ISearchable, IStoreable):
    """ This is a backend for metrics """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, parent: IStoreable, backend, **kwargs):
        super().__init__(parent=parent, backend=backend, **kwargs)


class ISourceRepository(IRepository, ISearchable, IStoreable):
    """ This is the interface of the execution backend. All data should be stored in a local file system. Currently
    we only store metadata. We store executions here. Executions are to be differentiated on code version (as well as
    dataset version???) and their call command (cluster, local???). These information are to be extracted from parent"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, parent: IStoreable, backend, **kwargs):
        super().__init__(parent=parent, backend=backend, **kwargs)


class IResultRepository(IRepository, ISearchable, IStoreable):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, parent: IStoreable, backend, **kwargs):
        super().__init__(parent=parent, backend=backend, **kwargs)


class ISplitRepository(IRepository, ISearchable, IStoreable, ILogRepository, IProgressable):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, parent: IStoreable, backend, **kwargs):
        super().__init__(parent=parent, backend=backend, **kwargs)


class IRunRepository(IRepository, ISearchable, IStoreable, ILogRepository, IProgressable):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, parent: IStoreable, backend, **kwargs):
        super().__init__(parent=parent, backend=backend, **kwargs)


class IExecutionRepository(IRepository, ISearchable, IStoreable, ILogRepository, IProgressable):
    """ This is the interface of the execution backend. All data should be stored in a local file system. Currently
    we only store metadata. We store executions here. Executions are to be differentiated on code version (as well as
    dataset version???) and their call command (cluster, local???). These information are to be extracted from parent"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, parent: IStoreable, backend, **kwargs):
        super().__init__(parent=parent, backend=backend, **kwargs)


class IExperimentRepository(IRepository, ISearchable, IStoreable, ILogRepository, IProgressable):
    """ This is the interface of the experiment backend. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, parent: IStoreable, backend, **kwargs):
        super().__init__(parent=parent, backend=backend, **kwargs)


class IProjectRepository(IRepository, ISearchable, IStoreable):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, backend, **kwargs):
        super().__init__(backend=backend, **kwargs)


class IDatasetRepository(IRepository, ISearchable, IStoreable):
    """ This is the interface of a data set backend. Data sets meta information should be stored in git. The data
    set itself can only be stored in something like git lfs"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, backend, **kwargs):
        super().__init__(backend=backend, **kwargs)
