from abc import ABCMeta, abstractmethod

from pypadre.core.base import ChildMixin
from pypadre.core.util.inheritance import SuperStop
from pypadre.pod.service.base_service import BaseService


class IBaseApp(SuperStop):
    """ Base class for apps containing backends. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def has_print(self) -> bool:
        pass

    @abstractmethod
    def print_(self, output, **kwargs):
        pass


class BaseEntityApp(IBaseApp):
    __metaclass__ = ABCMeta

    def __init__(self, service: BaseService, **kwargs):
        self.service = service
        super().__init__(**kwargs)

    def list(self, search=None, offset=0, size=100) -> list:
        """
        Lists all entities matching search.
        :param offset: Offset of the search
        :param size: Size of the search
        :param search: Search object
        :return: Entities
        """
        return self.service.list(search, offset, size)

    def put(self, obj):
        """
        Puts the entity
        :param obj: Entity to put
        :return: Entity
        """
        return self.service.put(obj)

    def patch(self, obj):
        """
        Updates the entity
        :param obj: Entity to put
        :return: Entity
        """
        return self.service.patch(obj)

    def get(self, id):
        """
        Get the entity by id
        :param id: Id of the entity to get
        :return: Entity
        """
        return self.service.get(id)

    def delete(self, obj):
        """
        Delete the entity
        :param obj: Entity to delete
        :return: Entity
        """
        return self.service.delete(obj)

    def delete_by_id(self, id):
        """
        Delete the entity by id
        :param id: Id of the entity to delete
        :return: Entity
        """
        return self.service.delete_by_id(id)

    @abstractmethod
    def has_print(self) -> bool:
        pass

    @abstractmethod
    def print_(self, output, **kwargs):
        pass


class BaseChildApp(ChildMixin, BaseEntityApp):
    """ Base class for apps being a child of another app. """
    __metaclass__ = ABCMeta

    def __init__(self, parent: IBaseApp, service: BaseService, **kwargs):
        super().__init__(parent=parent, service=service, **kwargs)

    def has_print(self) -> bool:
        parent: IBaseApp = self.parent
        return parent.has_print()

    def print_(self, output, **kwargs):
        parent: IBaseApp = self.parent
        return parent.print_(output, **kwargs)

    def _add_clz_decorators(self, clz, obj):
        obj.decorators = clz(self.parent, obj)
        return obj
