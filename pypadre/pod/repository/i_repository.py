from abc import ABCMeta, abstractmethod

from pypadre.pod.repository.generic.i_repository_mixins import ILogRepository, IProgressableRepository, \
    IStoreableRepository, IRepository, \
    ISearchable


class IMetricRepository(IRepository, ISearchable, IStoreableRepository):
    """ This is a backend for metrics """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, parent: IStoreableRepository, backend, **kwargs):
        super().__init__(parent=parent, backend=backend, **kwargs)


class ISourceRepository(IRepository, ISearchable, IStoreableRepository):
    """ This is the interface of the execution backend. All data should be stored in a local file system. Currently
    we only store metadata. We store executions here. Executions are to be differentiated on code version (as well as
    dataset version???) and their call command (cluster, local???). These information are to be extracted from parent"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, parent: IStoreableRepository, backend, **kwargs):
        super().__init__(parent=parent, backend=backend, **kwargs)


class IRunRepository(IRepository, ISearchable, IStoreableRepository, ILogRepository, IProgressableRepository):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, parent: IStoreableRepository, backend, **kwargs):
        super().__init__(parent=parent, backend=backend, **kwargs)


class IExecutionRepository(IRepository, ISearchable, IStoreableRepository, ILogRepository, IProgressableRepository):
    """ This is the interface of the execution backend. All data should be stored in a local file system. Currently
    we only store metadata. We store executions here. Executions are to be differentiated on code version (as well as
    dataset version???) and their call command (cluster, local???). These information are to be extracted from parent"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, parent: IStoreableRepository, backend, **kwargs):
        super().__init__(parent=parent, backend=backend, **kwargs)


class IExperimentRepository(IRepository, ISearchable, IStoreableRepository, ILogRepository, IProgressableRepository):
    """ This is the interface of the experiment backend. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, parent: IStoreableRepository, backend, **kwargs):
        super().__init__(parent=parent, backend=backend, **kwargs)


class IProjectRepository(IRepository, ISearchable, IStoreableRepository):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, backend, **kwargs):
        super().__init__(backend=backend, **kwargs)


class IDatasetRepository(IRepository, ISearchable, IStoreableRepository):
    """ This is the interface of a data set backend. Data sets meta information should be stored in git. The data
    set itself can only be stored in something like git lfs"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, backend, **kwargs):
        super().__init__(backend=backend, **kwargs)


class IComputationRepository(IRepository, ISearchable, IStoreableRepository):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, backend, **kwargs):
        super().__init__(backend=backend, **kwargs)


class ISplitRepository(IRepository, ISearchable, IStoreableRepository, ILogRepository, IProgressableRepository):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, parent: IStoreableRepository, backend, **kwargs):
        super().__init__(parent=parent, backend=backend, **kwargs)


class IPipelineResultRepository(IRepository, ISearchable, IStoreableRepository):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, parent: IStoreableRepository, backend, **kwargs):
        super().__init__(parent=parent, backend=backend, **kwargs)


class ICodeRepository(IRepository, ISearchable, IStoreableRepository):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, backend, **kwargs):
        super().__init__(backend=backend, **kwargs)
