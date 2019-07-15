from abc import abstractmethod, ABCMeta


class ISearchable:
    __metaclass__ = ABCMeta

    @abstractmethod
    def list(self, search: dict):
        pass

    @staticmethod
    def filter(objs: list, search: dict):
        return [o for o in objs if ISearchable.in_search(o, search)]

    @staticmethod
    def in_search(obj, search: dict):
        # TODO Enable more sophisticated search
        return all([hasattr(obj, k) and obj.get(k) == v for k, v in search.items()])
