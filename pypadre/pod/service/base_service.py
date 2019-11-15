from _py_abc import ABCMeta
from abc import abstractmethod
from logging import info
from typing import List, Type, Callable

from pypadre.core.events.events import Signaler, connect_subclasses, CommonSignals
from pypadre.core.model.generic.i_storable_mixin import StoreableMixin
from pypadre.core.validation.validation import ValidateableFactory, ValidationErrorHandler
from pypadre.pod.repository.exceptions import ObjectAlreadyExists
from pypadre.pod.repository.generic.i_repository_mixins import IStoreableRepository, ISearchable


class ServiceMixin:
    """ Base class for services containing backends. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, backends, *args, **kwargs):
        b = backends if isinstance(backends, List) else [backends]
        self._backends = [] if backends is None else b

    @property
    def backends(self):
        return self._backends

    def save_signal_fn(self, fn: Callable):
        setattr(self, "_signal_" + str(hash(fn)), fn)


class CrudServiceMixin(ServiceMixin):

    @abstractmethod
    def __init__(self, backends, *args, **kwargs):
        super().__init__(backends, *args, **kwargs)

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
            try:
                backend.put(obj, **kwargs)
            except ObjectAlreadyExists as e:
                info("Couldn't store object " + str(obj) + "! " + str(e) + " Skipping storing the object.")

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

    def get_by_hash(self, hash):
        """
        Get the entity by id
        :param hash: Id of the entity to get
        :return: Entity
        """
        obj_list = []
        for b in self.backends:
            backend: IStoreableRepository = b
            obj = backend.get_by_hash(hash)
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


class ModelServiceMixin(CrudServiceMixin):
    """ Base serivce for services handeling model entites. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, backends, *, model_clz: Type[Signaler], **kwargs):
        super().__init__(backends, **kwargs)
        self._model_clz = model_clz

        @connect_subclasses(model_clz, name=CommonSignals.GET.name)
        def get(sender, **sended_kwargs):
            """
            Function to get an code object by name.
            :param sender:
            :param sended_kwargs:
            :return:
            """
            return_val = sended_kwargs.get(StoreableMixin.RETURN_VAL)
            uid = sended_kwargs.get("uid")
            if uid is not None:
                return_val[StoreableMixin.RETURN_VAL] = next(iter(self.get(uid)), None)
            else:
                hash = sended_kwargs.get("hash")
                if hash is not None:
                    return_val[StoreableMixin.RETURN_VAL] = next(iter(self.get_by_hash(hash)), None)
        self.save_signal_fn(get)

    @property
    def model_clz(self):
        return self._model_clz

    def create(self, *args, handlers: List[ValidationErrorHandler]=None, **kwargs):
        if handlers is None:
            handlers = []
        return ValidateableFactory.make(self.model_clz, *args, handlers=handlers, **kwargs)
