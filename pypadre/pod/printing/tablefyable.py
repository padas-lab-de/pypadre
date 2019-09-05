from abc import abstractmethod, ABCMeta
from collections import OrderedDict

from pypadre.pod.util.dict_util import get_dict_attr

registry = OrderedDict()


class Tablefyable:
    __metaclass__ = ABCMeta
    """
    Used to allow for table printing of the object. Fill the registry for printing by implementing the _register_columns
     function.
    """

    @classmethod
    @abstractmethod
    def tablefy_register_columns(cls):
        pass

    @classmethod
    def _tablefy_register_columns(cls, properties):
        if cls.__name__ not in registry:
            registry[cls.__name__] = OrderedDict()
            registry[cls.__name__].update(properties)

    @classmethod
    def _tablefy_columns(cls):
        cls._tablefy_check_init()
        return registry[cls.__name__].keys()

    @classmethod
    def tablefy_columns(cls):
        return ", ".join(cls._tablefy_columns())

    @classmethod
    def _tablefy_check_init(cls):
        if cls.__name__ not in registry:
            cls.tablefy_register_columns()

    def tablefy_header(self, *args):
        """
        Gives the header list of the table.
        :param args: Names of the attributes to print
        :return:
        """
        self.__class__._tablefy_check_init()
        return [key for key, value in registry[self.__class__.__name__].items()
                if len(args) == 0 or len(args) >= 1 and key in args]

    def tablefy_to_row(self, *args):
        """
        Gives a row of the table.
        :param args: Names of the attributes to print
        :return:
        """
        self.__class__._tablefy_check_init()
        return [get_dict_attr(self, value)(self) if callable(get_dict_attr(self, value)) else
                get_dict_attr(self, value).fget(self) for key, value in registry[self.__class__.__name__].items()
                if len(args) == 0 or len(args) >= 1 and key in args]



