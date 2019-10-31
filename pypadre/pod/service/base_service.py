from _py_abc import ABCMeta
from typing import List, Type, Callable

from pypadre.core.events.events import Signaler
from pypadre.core.validation.validation import ValidateableFactory, ValidationErrorHandler
from pypadre.pod.repository.generic.i_repository_mixins import IStoreableRepository, ISearchable


class BaseService:
    """ Base class for apps containing backends. """
    __metaclass__ = ABCMeta

    def __init__(self, backends, *, model_clz: Type[Signaler], **kwargs):
        b = backends if isinstance(backends, List) else [backends]
        self._backends = [] if backends is None else b
        self._model_clz = model_clz

    @property
    def backends(self):
        return self._backends

    @property
    def model_clz(self):
        return self._model_clz

    def create(self, *args, handlers: List[ValidationErrorHandler]=None, **kwargs):
        if handlers is None:
            handlers = []
        return ValidateableFactory.make(self.model_clz, *args, handlers=handlers, **kwargs)

    def list(self, search, offset=0, size=100) -> list:
        """
        Lists all entities matching search.
        :param offset: Offset of the search
        :param size: Size of the search
        :param search: Search object
        :return: Entities
        """
        entities = []
        for b in self.backends:
            backend: ISearchable = b
            # TODO here the first backend takes priority can we change that?
            [entities.append(e) for e in backend.list(search=search, offset=offset, size=size) if len(entities) < size and e not in entities]
        return entities

    def put(self, obj, **kwargs):
        """
        Puts the entity
        :param obj: Entity to put
        :return: Entity
        """
        for b in self.backends:
            backend: IStoreableRepository = b
            backend.put(obj, **kwargs)

    def patch(self, obj):
        """
        Updates the entity
        :param obj: Entity to put
        :return: Entity
        """
        for b in self.backends:
            backend: IStoreableRepository = b
            backend.put(obj, allow_overwrite=True, merge=True)

    def get(self, uid):
        """
        Get the entity by id
        :param uid: Id of the entity to get
        :return: Entity
        """
        obj_list = []
        for b in self.backends:
            backend: IStoreableRepository = b
            obj = backend.get(uid)
            if obj is not None:
                obj_list.append(obj)
        return obj_list

    def delete(self, obj):
        """
        Delete the entity
        :param obj: Entity to delete
        :return: Entity
        """
        for b in self.backends:
            backend: IStoreableRepository = b
            backend.delete(obj)

    def delete_by_id(self, uid):
        """
        Delete the entity by id
        :param uid: Id of the entity to delete
        :return: Entity
        """
        for b in self.backends:
            backend: IStoreableRepository = b
            backend.delete_by_id(uid)

    def save_signal_fn(self, fn: Callable):
        setattr(self, "_signal_" + str(hash(fn)), fn)
