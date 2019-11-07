import numpy as np

from pypadre import _version, _name
from pypadre.binding.model.sklearn_gridsearch import SKLearnGridSearch
from pypadre.binding.visitors.scikit import SciKitVisitor
from pypadre.core.base import phases
from pypadre.core.model.computation.training import Training
from pypadre.core.model.pipeline.components.component_mixins import ProvidedComponentMixin, EstimatorComponentMixin, \
    ParameterizedPipelineComponentMixin
from pypadre.core.util.utils import unpack


def is_sklearn_pipeline(pipeline):
    """
    checks whether pipeline is a sklearn pipeline
    :param pipeline:
    :return:
    """
    # we do checks via strings, not isinstance in order to avoid a dependency on sklearn
    return type(pipeline).__name__ == 'Pipeline' and type(pipeline).__module__ == 'sklearn.pipeline'


def estimate(ctx, **kwargs):
    (component,) = unpack(ctx, "component")
    return component.estimate(ctx, **kwargs)


class SKLearnEstimator(ProvidedComponentMixin, EstimatorComponentMixin, ParameterizedPipelineComponentMixin):
    """
    This class encapsulates an sklearn workflow which allows to run sklearn pipelines or a list of sklearn components,
    report the results according to the outcome via the experiment logger.

    A workflow is a single run of fitting, transformation and inference.
    It does not contain any information on the particular split or state of an experiment.
    Workflows are used for abstracting from the underlying machine learning framework.
    """

    def __init__(self, *, pipeline=None, parameter_provider=None, **kwargs):
        # TODO don't change state of pipeline!!!
        # check for final component to determine final results
        # if step wise is true, log intermediate results. Otherwise, log only final results.
        # distingusish between training and fitting in classification.
        # TODO use default parameter provider if none is given

        if parameter_provider is None:
            parameter_provider = SKLearnGridSearch()

        if not pipeline or not is_sklearn_pipeline(pipeline):
            raise ValueError("SKLearnEstimator needs a delegate defined as sklearn.pipeline")
        self._pipeline = pipeline

        super().__init__(package=__name__, fn_name="estimate",  requirement=_name.__name__, version=_version.__version__, name="SKLearnEstimator", parameter_provider=parameter_provider, **kwargs)

    def estimate(self, ctx, **kwargs):
        (split, component, run, initial_hyperparameters) = unpack(ctx, "data", "component", "run",
                                                                  "initial_hyperparameters")

        self.set_parameter_values(parameters=kwargs)

        self.send_start(phase='sklearn.' + phases.fitting)
        y = None
        if split.train_targets is not None:
            y = split.train_targets.reshape((len(split.train_targets),))
        else:

            # Create dummy target of zeros if target is not present.
            y = np.zeros(shape=(len(split.train_features, )))
        self._pipeline.fit(split.train_features, y)
        self.send_stop(phase='sklearn.' + phases.fitting)
        if self.is_scorer():
            self.send_start(phase=f"sklearn.scoring.trainset")
            score = self._pipeline.score(split.train_features, y)
            self.send_stop(phase=f"sklearn.scoring.trainset")
            # TODO use other signals?
            self.send_log(keys=['training score'], values=[score], message="Logging the training score")

            if split.has_valset():
                y = split.val_targets.reshape((len(split.val_targets),))
                self.send_start(phase='sklearn.scoring.valset')
                score = self._pipeline.score(split.val_features, y)
                self.send_stop(phase='sklearn.scoring.valset')
                self.send_log(keys=['validation score'], values=[score], message="Logging the validation score")
        return Training(split=split, component=component, run=run, model=self._pipeline, parameters=kwargs,
                        initial_hyperparameters=initial_hyperparameters)

    def hash(self):
        # TODO hash should not change with training
        return self.pipeline.__hash__()

    def configuration(self):
        return SciKitVisitor(self._pipeline)

    def is_inferencer(self):
        return getattr(self._pipeline, "predict", None)

    def is_scorer(self):
        return getattr(self._pipeline, "score", None)

    def is_transformer(self):
        return getattr(self._pipeline, "transform", None)

    @property
    def pipeline(self):
        return self._pipeline

    def set_parameter_values(self, parameters):

        for parameter in parameters:
            # split_params[0] will be the name of the estimator
            # split_params[1] will be the name of the parameter
            split_params = parameter.split(sep='.')
            estimator_name = split_params[0]
            parameter_name = split_params[1]

            estimator = self.pipeline.named_steps.get(estimator_name)
            if estimator is None:
                # Check if the estimator name is present int he alternate name mappings
                estimator_name = self.find_estimator_name_in_mapping(estimator_name)
                estimator = self.pipeline.named_steps.get(estimator_name)

            # If the estimator is still not available throw an exception
            assert (estimator is not None)

            # The hyperparameters should be set to the variable which is available in the path of the
            # hyperparameter name in the mappings file
            parameter_path = parameter_name
            if not hasattr(estimator, parameter_name):
                # Check if the estimator has such a parameter and get its path
                # Get the actual estimator name which would correspond to the mappings.json
                parameter_path = self.get_parameter_path(estimator_name=estimator_name, parameter_name=parameter_name)

            assert (parameter_name is not None)

            estimator.set_params(**{parameter_path: parameters[parameter]})

    def find_estimator_name_in_mapping(self, name):
        # This function is used to return the actual estimator name as specified in the mappings file
        # Estimators might have alternate names and users might specify the estimators using the alternate names
        # So to find the valid parameters, we need to find the estimator name as specified in the mappings file
        from pypadre.core.visitors.mappings import alternate_name_mappings
        return alternate_name_mappings.get(name.lower())

    def get_parameter_path(self, estimator_name, parameter_name):
        from pypadre.core.visitors.mappings import name_mappings

        # If the alternate estimator name is given, convert the alternate estimator name
        # to actual estimator name
        if name_mappings.get(estimator_name, None) is None:
            estimator = self.find_estimator_name_in_mapping(estimator_name)

        else:
            estimator = estimator_name

        parameters = name_mappings.get(estimator).get('hyper_parameters').get('model_parameters')

        # If the parameter name matches return the path of the parameter
        for parameter in parameters:
            if parameter.get('name') == parameter_name:
                return parameter.get('scikit-learn').get('path')

        return None

    def get_initial_hyperparameters(self):
        from pypadre.core.visitors.mappings import name_mappings

        hyperparameter_dict = dict()

        for estimator_name in self.pipeline.named_steps:
            # Find all the hyperparameters of the estimator from the mappings file
            # Get the values of the hyperparameter from the object
            actual_estimator_name = None
            component_hyperparameter_dict = dict()

            if name_mappings.get(estimator_name, None) is None:
                actual_estimator_name = self.find_estimator_name_in_mapping(estimator_name)
            else:
                actual_estimator_name = estimator_name

            if actual_estimator_name is None:
                raise ValueError('Estimator {name} is not present in the mappings file.'.format(name=estimator_name))
            parameters = name_mappings.get(actual_estimator_name).get('hyper_parameters').get('model_parameters')
            estimator_obj_dict = self.pipeline.named_steps.get(estimator_name).__dict__
            for parameter_dict in parameters:
                # Find the variable name of each parameter
                parameter_path = self.get_parameter_path(actual_estimator_name, parameter_dict.get('name'))
                # Assign the value of the parameter to the name of the parameter
                component_hyperparameter_dict[parameter_dict.get('name')] = estimator_obj_dict.get(parameter_path)

            hyperparameter_dict[actual_estimator_name] = component_hyperparameter_dict

        return {'sklearn': hyperparameter_dict}

