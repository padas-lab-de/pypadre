from typing import List, Set

from pypadre.core.base import MetadataEntity, ChildEntity
from pypadre.core.model.generic.i_model_mixins import IStoreable, IProgressable
from pypadre.core.printing.tablefyable import Tablefyable
from pypadre.core.model.generic.i_executable_mixin import IExecuteable

WRITE_RESULTS = "write_results"
WRITE_METRICS = "write_metrics"


class Run(IExecuteable, IStoreable, MetadataEntity, ChildEntity, Tablefyable):
    """
    A run is an execution of the pipeline on a specific dataset. Each time an experiment is executed a new run is
    created.
    """

    EXECUTION_ID = "execution_id"

    @classmethod
    def _tablefy_register_columns(cls):
        pass

    def __init__(self, execution,  **kwargs):
        # Add defaults
        defaults = {}

        # Merge defaults
        metadata = {**defaults, **{self.EXECUTION_ID: execution.id}, **kwargs.pop("metadata", {})}
        super().__init__(schema_resource_name="run.json", parent=execution, result=self, metadata=metadata, **kwargs)

    def _execute_helper(self, *args, **kwargs):

        # Send signal
        self.send_put()

        # Start execution of the pipeline
        # pipeline_parameters = kwargs.get('parameters', None)
        pipeline_parameters, write_parameters = \
            self.separate_hyperparameters_and_component_parameters(kwargs.pop('parameters', {}))
        return self.pipeline.execute(dataset=self.dataset, run=self, pipeline_parameters=pipeline_parameters,
                                     write_parameters=write_parameters,
                                     *args, **kwargs)

    @property
    def execution(self):
        return self.parent

    @property
    def dataset(self):
        return self.execution.dataset

    @property
    def experiment(self):
        return self.execution.experiment

    @property
    def pipeline(self):
        return self.execution.pipeline

    def separate_hyperparameters_and_component_parameters(self, parameters:dict):

        parameter_dict = dict()
        write_result_metric_dict = dict()

        # Iterate through every parameter
        for component_name in parameters:
            params = parameters.get(component_name)
            # Save the hyperparamters in a separate dictionary
            if params.get('parameters', None) is not None:
                parameter_dict[component_name] = params.get('parameters')

            # If the user wants to dump the results of the component, set the Flag
            write_results = params.get(WRITE_RESULTS, False)

            # The user can specify multiple metrics if needed, so it is a list
            write_metrics = params.get(WRITE_METRICS, None)

            write_result_metric_dict[component_name] = {WRITE_RESULTS: write_results,
                                                        WRITE_METRICS: write_metrics}
        if not parameter_dict:
            parameter_dict = None

        return parameter_dict, write_result_metric_dict

