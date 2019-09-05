from abc import ABCMeta, abstractmethod
from typing import List

from pypadre.pod.base import ChildEntity
from pypadre.pod.backend.interfaces.backend.generic.i_searchable import ISearchable
from pypadre.pod.backend.interfaces.backend.generic.i_storeable import IStoreable
from pypadre.pod.backend.interfaces.backend.i_backend import IBackend


class IBaseApp:
    """ Base class for apps containing backends. """
    __metaclass__ = ABCMeta

    def __init__(self, backends):
        b = backends if isinstance(backends, List) else [backends]
        self._backends = [] if backends is None else b

    @property
    def backends(self):
        return self._backends

    @abstractmethod
    def has_print(self) -> bool:
        pass

    @abstractmethod
    def print_(self, output, **kwargs):
        pass


class BaseChildApp(ChildEntity, IBaseApp):
    """ Base class for apps being a child of another app. """
    __metaclass__ = ABCMeta

    def __init__(self, parent: IBaseApp, backends: List[IBackend], **kwargs):
        ChildEntity.__init__(self, parent=parent, backends=backends, **kwargs)
        IBaseApp.__init__(self, backends=backends)

    def list(self, search, offset=0, size=100) -> list:
        entities = []
        for b in self.backends:
            backend: ISearchable = b
            [entities.append(e) for e in backend.list(search=search) if len(entities) < size and e not in entities]
        return entities

    def put(self, obj):
        for b in self.backends:
            backend: IStoreable = b
            backend.put(obj)

    def get(self, id):
        obj_list = []
        for b in self.backends:
            backend: IStoreable = b
            obj = backend.get(id)
            if obj is not None:
                obj_list.append(obj)
        return obj_list

    def delete(self, obj):
        for b in self.backends:
            backend: IStoreable = b
            backend.delete(obj)

    def delete_by_id(self, id):
        for b in self.backends:
            backend: IStoreable = b
            backend.delete_by_id(id)

    def has_print(self) -> bool:
        parent: IBaseApp = self.parent
        return parent.has_print()

    def print_(self, output, **kwargs):
        parent: IBaseApp = self.parent
        return parent.print_(output, **kwargs)