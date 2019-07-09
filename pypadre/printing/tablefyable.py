from collections import OrderedDict


class Tablefyable:
    """
    Used to allow for table printing of the object. Fill the registry for printing in following way:
    self._registry.update({'id': get_dict_attr(self, 'id').fget, 'name': get_dict_attr(self, 'name').fget,
                               'type': get_dict_attr(self, 'type').fget, 'size': get_dict_attr(self, 'size').fget,
                               'format': get_dict_attr(self, 'binary_format')})
    """

    def __init__(self):
        self._registry = OrderedDict()

    def tablefy_header(self, *args):
        """
        Gives the header list of the table.
        :param args: Names of the attributes to print
        :return:
        """
        return [key for key, value in self._registry.items() if len(args) == 0 or len(args) >= 1 and key in args]

    def tablefy_to_row(self, *args):
        """
        Gives a row of the table.
        :param args: Names of the attributes to print
        :return:
        """
        return [value(self) for key, value in self._registry.items() if len(args) == 0 or len(args) >= 1 and key in args]



