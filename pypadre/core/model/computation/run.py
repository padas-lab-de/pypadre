import uuid

import pyhash

from pypadre.core.base import MetadataMixin, ChildMixin
from pypadre.core.model.generic.i_executable_mixin import ValidateableExecutableMixin
from pypadre.core.model.generic.i_storable_mixin import StoreableMixin
from pypadre.core.util.utils import persistent_hash
from pypadre.core.validation.json_validation import make_model

WRITE_RESULTS = "write_results"
WRITE_METRICS = "write_metrics"
METRICS = "metrics"

run_model = make_model(schema_resource_name='run.json')


class Run(StoreableMixin, ValidateableExecutableMixin, MetadataMixin, ChildMixin):
    """
    A run is an execution of the pipeline on a specific dataset. Each time an experiment is executed a new run is
    created.
    """
    EXECUTION_ID = "execution_id"

    @classmethod
    def _tablefy_register_columns(cls):
        super()._tablefy_register_columns()

    def __init__(self, execution, **kwargs):
        # Add defaults
        defaults = {}

        # Merge defaults
        metadata = {**defaults,
                    **{"id": uuid.uuid4().__str__() + "-" + str(persistent_hash(execution.id, algorithm=pyhash.city_64())),
                       self.EXECUTION_ID: execution.id}, **kwargs.pop("metadata", {})}
        super().__init__(model_clz=run_model, parent=execution, result=self, metadata=metadata, **kwargs)

    def _execute_helper(self, *args, **kwargs):

        # Send signal
        self.send_put()

        # Start execution of the pipeline
        # pipeline_parameters = kwargs.get('parameters', None)
        pipeline_parameters, write_parameters, metrics_map = \
            self.separate_hyperparameters_and_component_parameters(kwargs.pop('parameters', {}))
        return self.pipeline.execute(dataset=self.dataset, run=self, pipeline_parameters=pipeline_parameters,
                                     write_parameters=write_parameters, metrics_map=metrics_map,
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

    @property
    def execution_id(self):
        return self.parent.id

    def separate_hyperparameters_and_component_parameters(self, parameters: dict):

        parameter_dict = dict()
        write_result_metric_dict = dict()
        metrics_map = dict()

        # Iterate through every parameter
        if parameters is not None:
            for component_name in parameters:
                params = parameters.get(component_name)
                # Save the hyperparamters in a separate dictionary
                if params.get('parameters', None) is not None:
                    parameter_dict[component_name] = params.get('parameters')

                # If the user wants to dump the results of the component, set the Flag
                write_results = params.get(WRITE_RESULTS, False)

                # The user can specify multiple metrics if needed, so it is a list
                write_metrics = params.get(WRITE_METRICS, None)

                # Or the user can specify it within the metrics field
                metrics = params.get(METRICS, None)

                # if the user specifies the different metrics that should be written, then set write_metrics to True
                # and add the metrics to the variable
                if write_metrics is not None and  not isinstance(write_metrics, list) and \
                        write_metrics not in [True, False]:
                    metrics = write_metrics
                    write_metrics = True
                    metrics_map[component_name] = [metrics]

                elif isinstance(write_metrics, list):
                    metrics_map[component_name] = write_metrics

                elif metrics is not None:
                    metrics_map[component_name] = metrics if isinstance(metrics, list) else [metrics]

                write_result_metric_dict[component_name] = {WRITE_RESULTS: write_results,
                                                            WRITE_METRICS: write_metrics}

            if not parameter_dict:
                parameter_dict = None

            return parameter_dict, write_result_metric_dict, metrics_map
        else:
            return {}, {}, {}
