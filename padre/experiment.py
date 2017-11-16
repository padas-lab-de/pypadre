"""
Classes for managing experiments. Contains
- format conversion of data sets (e.g. pandas<-->numpy)
- parameter settings for experiments
- hyperparameter optimisation
- logging
"""

import abc
from abc import abstractmethod

class ExperimentVisitor(abc.ABC):
    """
    A Visitor to inspect library-specific experiment setups and extract the setup information.
    """

    def __init__(self):
        """
        Default implementation of constructor.
        """
        super().__init__()

    def __call__(self, object, result = None):
        """
        Make a visitor callable. Calls extract.
        :param object: see extract
        :param result: (optional) the output-dictionary
        :return: see extract
        """
        if result is None:
            result = {}
        return self.extract(object, result)

    @abstractmethod
    def extract(self, object, result):
        """
        (abstract) Must be overridden by subclasses to implement custom extraction behaviour.
        :param object: the object of which information are to be extracted
        :param result: a dictionary containing all extracted information
        :return: a dictionary containing all extracted padre-keywords of the template
        """
        return

    @staticmethod
    def applyVisitor(object, result, template):
        """
        Applies the Visitor to the object depending on the type of template and stores the result in the result variable:
        - dict: the DictExperimentVisitor will be applied to the object using that dict as template
        - tuple: the TupleExperimentVisitor will be applied to the object using that tuple as template
        - str: the value of the object will be stored in the result dict using the value of the string as key
        - callable: the template will get called with the object and a reference to the result-dictionary as arguments
        - None: no information will be extracted
        """
        if type(template) is str:
            if template in result:
                print("Warning: Attribute '" + template + "' extracted multiple times! Last value will be used.")
            result[template] = object
        elif type(template) is dict:
            DictExperimentVisitor(template).extract(object, result)
        elif type(template) is tuple:
            TupleExperimentVisitor(template).extract(object, result)
        elif callable(template):
            template(object, result)
        elif template is None:
            pass
        else:
            raise TypeError("Template contains value of unexpected Type: " + str(type(template)))

        return result

class DictExperimentVisitor(ExperimentVisitor):
    """
    A Visitor-implementation to inspect library-specific experiment setups and extract the setup information by using a template-dictionary.
    For each key in the dicitionary the corresponding value will be applied as visitor to a member of object by the same name as key.
    """


    def __init__(self, template):
        """
        :param template: template dictionary describing the experiment hierarchy
        """
        super().__init__()
        self.template = template

    def extract(self, object, result):
        """
        Implementation of the ExperimentVisitor-interface. Returns the result of a call to extract_rec with the given arguments and the stored template.
        :param object: the object of which information are to be extracted
        :param result: a dictionary containing all extracted information
        :return: a dictionary containing all extracted padre-keywords of the template
        """
        return DictExperimentVisitor.extract_rec(object, result, self.template)

    @staticmethod
    def extract_rec(object, result, template):
        """
        Stores all information of object extracted using template in the dict result.
        :param object: an object that is part of the experiment description
        :param result: result dictionary used for recursion
        :param template: template used for recursion
        :return: a dictionary containing all extracted padre-keywords of the template
        """
        # Check if object is a dict or a class object
        if type(object) is dict:
            object_vars = object
        elif hasattr(object, "__dict__"):
            object_vars = vars(object)
        else:
            raise TypeError("Inspected object has unsupported Type: " + str(type(object)))

        for k in template:
            if k in object_vars:
                ExperimentVisitor.applyVisitor(object_vars[k], result, template[k])
            else:
                print("Warning: Attribute '" + k + "' could not be found in object " + repr(object) + ".")


        return result

class ListExperimentVisitor(ExperimentVisitor):
    """
    A Visitor-implementation to inspect library-specific experiment setups and extract the setup information of a list by using a template on every object of the list.
    The template is represented by a dict, where a value can be of any of the types supported by ExperimentVisitor.applyVisitor.
    """


    def __init__(self, prefix, template):
        """
        :param prefix: name, that is used to store the list of extracted values
        :param template: template dictionary describing the experiment hierarchy for the elements of the list
        """
        super().__init__()
        self.prefix = prefix
        self.template = template

    def extract(self, object, result):
        """
        Implementation of the ExperimentVisitor-interface. Inserts the list of dictionaries,
         each obtained by calling applyVisitor on the respective element in the input list, into result with the key given by self.prefix.
        :param object: the input-list of which information are to be extracted
        :param result: a dictionary containing all extracted information
        :return: the result dictionary
        """
        if self.prefix in result:
            print("Warning: Attribute '" + self.prefix + "' extracted multiple times! Last value will be used.")
        result[self.prefix] = [self.applyVisitor(e, {}, self.template) for e in object]
        return result


class TupleExperimentVisitor(ExperimentVisitor):
    """
    A Visitor-implementation to inspect library-specific experiment setups and extract the setup information of a tuple-like by applying a template-tuple pairwise on the objects of the tuple.
    The template is represented by a tuple, where a value can be of any of the types supported by ExperimentVisitor.applyVisitor.
    """

    def __init__(self, template):
        """
        :param template: template tuple describing the visitors for the elements in the input-tuple
        """
        super().__init__()
        self.template = template

    def extract(self, object, result):
        """
        Implementation of the ExperimentVisitor-interface. Applies pairwise the template-visitor to the elements of the input-object.
        :param object: the input-tuple of which information are to be extracted
        :param result: a dictionary containing all extracted information
        :return: a dictionary containing all extracted padre-information
        """
        if len(object) != len(self.template):
            print("Warning: Tuple size doesn't match!")
        for i in range(min(len(object), len(self.template))):
            self.applyVisitor(object[i], result, self.template[i])
        return result

class SelectExperimentVisitor(ExperimentVisitor):
    """
    A Visitor-implementation to inspect library-specific experiment setups and select the correct visitor using a decision-function.
    """

    def __init__(self, decision):
        """
        :param decision: a callable, that takes one object and returns a visitor-object
        """
        super().__init__()
        self.decision = decision

    def extract(self, object, result):
        """
        Implementation of the ExperimentVisitor-interface. Applies the visitor returned by the decision-function to object.
        :param object: the input-tuple of which information are to be extracted
        :param result: a dictionary containing all extracted information
        :return: a dictionary containing all extracted padre-information
        """
        return self.applyVisitor(object, result, self.decision(object))