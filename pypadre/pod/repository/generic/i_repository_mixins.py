from abc import ABCMeta, abstractmethod


class IStoreableRepository:
    """ This is the interface for all backends being able to store objects onto some kind of persistence storage."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def get(self, uid):
        pass

    def exists(self, uid):
        # TODO don't load object for better performance
        return self.get(uid) is not None

    @abstractmethod
    def put(self, obj, *args, merge=False, allow_overwrite=False, **kwargs):
        pass

    @abstractmethod
    def delete_by_id(self, uid):
        pass

    @abstractmethod
    def delete(self, obj):
        pass


class ISearchable:
    """ Interface for backends being searchable. Search on json data, folder name etc. for REST, local etc."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def list(self, search: dict, offset=0, size=100) -> list:
        pass

    @staticmethod
    def filter(objs: list, search: dict):
        if search is None:
            return objs
        return [o for o in objs if ISearchable.in_search(o, search)]

    @staticmethod
    def in_search(obj, search: dict):
        # TODO Enable more sophisticated search
        return all([hasattr(obj, k) and getattr(obj, k) == v for k, v in search.items()])


class IProgressableRepository:
    """ This is the interface for all backends being able to progress the state of one of their
    currently running processes."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def put_progress(self, obj, **kwargs):
        pass


class ILogRepository:
    """ This is the interface for all backends which are able to log interactions into some kind of log store """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def log(self, msg):
        pass


class IRepository:
    """ This is the simple entry implementation of a backend. We define a hierarchical structure
    to the other backends here. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *, backend, **kwargs):
        self._backend = backend

    @property
    def backend(self):
        return self._backend
