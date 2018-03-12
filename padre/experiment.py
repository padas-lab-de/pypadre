"""
Classes for managing experiments. Contains
- format conversion of data sets (e.g. pandas<-->numpy)
- parameter settings for experiments
- hyperparameter optimisation
- logging
"""
from padre.base import MetadataEntity
from padre.visitors.scikit.scikitpipeline import SciKitVisitor
import sys



class Experiment(MetadataEntity):
    """
    Experiment class covering functionality for executing and evaluating machine learning experiments.

    An experiment requires:
    - a pipeline. Frist, we support only sklearn pipelines
      (http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
    - a dataset. First, we only support a single dataset
    - a splitting strategy (see http://scikit-learn.org/stable/model_selection.html)

    The parameters and setup of the Experiment are automatically extracted using the visitor functionality
    """
    def __init__(self, dataset,
                 pipeline,
                 splitting_strategy,
                 **metadata):
        super().__init__(**metadata)
        self._dataset = dataset
        self._pipeline = pipeline
        self._splitting_strategy = splitting_strategy
        self._update_configuration()

    def _update_configuration(self):
        self._metadata["configuration"] = SciKitVisitor(self._pipeline)

    def configuration(self):
        return self._metadata["configuration"]

    def hyperparameters(self):
        """
        returns the hyperparameters per pipeline element as dict from the extracted configruation
        :return:
        """
        params = dict()
        steps = self.configuration()[0]["steps"]
        for step in steps:
            params = dict(step)
            if "doc" in params:
                del params["doc"]
        return params


    @staticmethod
    def create(config):
        """
         create an experiment from a provided configuration dictionary
        :param config: dictionary containing the configuration
        :return:
        """
        pass

    def run(self, reporter=sys.stdout):
        """
        runs the experiment
        :param reporter:
        :return:
        """
        pass

    def access(self):
        """
        placeholder for access functionality. to be defined later
        :return:
        """
        pass
