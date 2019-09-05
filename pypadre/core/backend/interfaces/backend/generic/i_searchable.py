from abc import abstractmethod, ABCMeta


class ISearchable:
    """ Interface for backends being searchable. Search on json data, folder name etc. for REST, local etc."""
    __metaclass__ = ABCMeta

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
