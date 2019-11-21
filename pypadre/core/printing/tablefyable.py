from abc import abstractmethod, ABCMeta
from collections import OrderedDict

from pypadre.core.util.inheritance import SuperStop
from pypadre.pod.util.dict_util import get_dict_attr

registry = OrderedDict()


class Tablefyable(SuperStop):
    __metaclass__ = ABCMeta
    """
    Used to allow for table printing of the object. Fill the registry for printing by implementing the _register_columns
     function.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    @abstractmethod
    def _tablefy_register_columns(cls):
        """
        Overwrite to register the property names you want to be able to access via tablefy

        Example:

        For function mappings:
        @classmethod
        def tablefy_register_columns(cls):
            cls._tablefy_register_columns({'id': 'id', 'name': 'name', 'type': 'type'})

        or

        For simple getters by property name:
        @classmethod
        def tablefy_register_columns(cls):
            cls._tablefy_register('id', 'name', 'type')

        :return:
        """
        pass

    @classmethod
    def tablefy_register(cls, *args: str):
        cls.tablefy_register_columns({key: key for key in args})

    @classmethod
    def tablefy_register_columns(cls, properties):
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
            # also register properties of the super class
            s = super()
            if isinstance(s, Tablefyable):
                s._tablefy_register_columns()

            # register the properties
            cls._tablefy_register_columns()

    @classmethod
    def tablefy_header(cls, *args):
        """
        Gives the header list of the table.
        :param args: Names of the attributes to print
        :return:
        """
        cls._tablefy_check_init()
        return [key for key, value in registry[cls.__name__].items()
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

    def __str__(self):
        self.__class__._tablefy_check_init()
        return self.__class__.__name__ + "[" + ", ".join(
            ["'" + key + ": " + str(get_dict_attr(self, value)(self)) + "'" if callable(get_dict_attr(self, value)) else
             "'" + key + ": " + str(get_dict_attr(self, value).fget(self)) + "'" for key, value in
             registry[self.__class__.__name__].items()]) + "]"
