

class Parameter(object):
    """
    A class representing a single parameter, that was extracted using the ExperimentVisitor.
    It contains the extracted value and the path in the original object to that parameter.
    """

    def __init__(self, value, path):
        """
        Constructs a new Parameter.
        :param value: the value of the parameter
        :param path: the path to the parameter from the original object
        """
        self.value = value
        self.path = path

    def type(self):
        """
        Returns the type of the value of the parameter.
        :return: the type of the value
        """
        return type(self.value)

    def __ne__(self, other):
        return self.value != other

    def __eq__(self, other):
        return self.value == other

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return "{ value: " + repr(self.value) + ", path: " + repr(self.path) + " }"

    def __str__(self):
        return str(self.value)

