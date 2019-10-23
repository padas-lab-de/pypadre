from collections import Callable
from typing import Union, Type

import numpy as np
from padre.PaDREOntology import PaDREOntology
from sklearn.pipeline import Pipeline

from pypadre.binding.visitors.scikit import SciKitVisitor
from pypadre.core.base import phases
from pypadre.core.model.code.icode import ICode
from pypadre.core.model.computation.evaluation import Evaluation
from pypadre.core.model.computation.training import Training
from pypadre.core.model.pipeline import pipeline
from pypadre.core.model.pipeline.components import EstimatorComponent, EvaluatorComponent, \
    ParameterizedPipelineComponent, ProvidedComponent
from pypadre.core.model.pipeline.parameters import GridSearch
from pypadre.core.model.pipeline.pipeline import DefaultPythonExperimentPipeline
from pypadre.core.util.utils import unpack
from pypadre.core.visitors.mappings import name_mappings, alternate_name_mappings


def _is_sklearn_pipeline(pipeline):
    """
    checks whether pipeline is a sklearn pipeline
    :param pipeline:
    :return:
    """
    # we do checks via strings, not isinstance in order to avoid a dependency on sklearn
    return type(pipeline).__name__ == 'Pipeline' and type(pipeline).__module__ == 'sklearn.pipeline'


class SKLearnEstimator(ProvidedComponent, EstimatorComponent, ParameterizedPipelineComponent):
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

        if not pipeline or not _is_sklearn_pipeline(pipeline):
            raise ValueError("SKLearnEstimator needs a delegate defined as sklearn.pipeline")
        self._pipeline = pipeline

        super().__init__(name="SKLearnEstimator", parameter_provider=parameter_provider, **kwargs)

    def _call(self, ctx, **kwargs):
        (split, component, run) = unpack(ctx, "data", "component", "run")

        self.set_parameter_values(parameters=kwargs.get('parameters', {}))

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
        return Training(split=split, component=component, run=run, model=self._pipeline, **kwargs)

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


class SKLearnEvaluator(ProvidedComponent, EvaluatorComponent, ParameterizedPipelineComponent):
    """
    This class takes the output of an sklearn workflow which represents the fitted model along with the corresponding split,
    report and save all possible results that allows for common/custom metric computations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _call(self, ctx, **kwargs):
        data, predecessor, component, run = unpack(ctx, "data", ("predecessor", None), "component", "run")
        model = data["model"]
        split = data["split"]

        # TODO CLEANUP. METRICS SHOULDN'T BE CALCULATED HERE BUT CALCULATED BY INDEPENDENT METRICS MEASURES
        # TODO still allow for custom metrics which are added by using sklearn here?

        train_idx = split.train_idx
        test_idx = split.test_idx

        self.send_error(message="Test set is missing.", condition=not split.has_testset())

        self.send_start(phase='sklearn.' + phases.inferencing)
        train_idx = train_idx.tolist()
        test_idx = test_idx.tolist()

        y_predicted_probabilities = None
        y = split.test_targets.reshape((len(split.test_targets),))

        y_predicted = np.asarray(model.predict(split.test_features))
        self.send_stop(phase='sklearn.' + phases.inferencing)

        results = {'predicted': y_predicted.tolist(),
                   'truth': y.tolist()}

        modified_results = dict()

        self.send_log(mode='probability', pred=y_predicted, truth=y,
                      message="Checking if the workflow supports probability computation or not.")

        # Check if the final estimator has an attribute called probability and if it has check if it is True
        compute_probabilities = True
        if hasattr(model.steps[-1][1], 'probability') and not model.steps[-1][1].probability:
            compute_probabilities = False

        # log the probabilities of the result too if the method is present

        final_estimator_name = model.steps[-1][0]
        if name_mappings.get(final_estimator_name) is None:
            # If estimator name is not present in name mappings check whether it is present in alternate names
            estimator = alternate_name_mappings.get(str(final_estimator_name).lower())
            final_estimator_type = name_mappings.get(estimator).get('type')
        else:
            final_estimator_type = name_mappings.get(model.steps[-1][0]).get('type')

        self.send_error(condition=final_estimator_type is None,
                        message='Final estimator could not be found in names or alternate names')

        if final_estimator_type == 'Classification' or \
                (final_estimator_type == 'Neural Network' and np.all(np.mod(y_predicted, 1)) == 0):
            results['type'] = PaDREOntology.SubClassesExperiment.Classification.value

            if compute_probabilities:
                y_predicted_probabilities = model.predict_proba(split.test_features)
                self.send_log(mode='probability', pred=y_predicted, truth=y, probabilities=y_predicted_probabilities,
                              message="Computing and saving the prediction probabilities")
                results['probabilities'] = y_predicted_probabilities.tolist()
        else:
            results['type'] = PaDREOntology.SubClassesExperiment.Regression.value

        if self.is_scorer(model):
            self.send_start(phase=f"sklearn.scoring.testset")
            score = model.score(split.test_features, y, )
            self.send_stop(phase=f"sklearn.scoring.testset")
            self.send_log(keys=["test score"], values=[score], message="Logging the testing score")

        results['dataset'] = split.dataset.name
        results['train_idx'] = train_idx
        results['test_idx'] = test_idx
        results['training_sample_count'] = len(train_idx)
        results['testing_sample_count'] = len(test_idx)
        results['split_num'] = split.number

        # TODO results as object?

        return Evaluation(training=predecessor, result=results, component=component, run=run, **kwargs)

    def hash(self):
        # TODO
        return self.__hash__()

    @staticmethod
    def is_inferencer(model=None):
        return getattr(model, 'predict', None)

    @staticmethod
    def is_scorer(model=None):
        return getattr(model, 'score', None)

    @staticmethod
    def is_transformer(model=None):
        return getattr(model, 'transform', None)


class SKLearnPipeline(DefaultPythonExperimentPipeline):
    def __init__(self, *, splitting: Union[Type[ICode], Callable] = None, pipeline_fn: Callable, **kwargs):
        """

        :param splitting:
        :param pipeline_fn: A function that returns a Sklearn pipeline as the return value
        :param kwargs:
        """
        pipeline = pipeline_fn()
        visitor = SciKitVisitor(pipeline)
        # TODO use visitor to extract parameter schema from pipeline

        # Check if the return type is a Sklearn Pipeline
        assert(isinstance(pipeline, Pipeline))

        # Verify running two instances of the function creates two Pipeline objects
        assert(pipeline is not pipeline_fn())
        sk_learn_estimator = SKLearnEstimator(pipeline=pipeline)
        sk_learn_evaluator = SKLearnEvaluator()
        super().__init__(splitting=splitting, estimator=sk_learn_estimator, evaluator=sk_learn_evaluator, **kwargs)


class SKLearnPipelineV2(pipeline.Pipeline):
    def __init__(self, *, splitting: Union[ICode, Callable] = None, pipeline: Pipeline, **kwargs):
        super().__init__(**kwargs)
        # TODO for each pipeline element in sklearn create a pipeline component


class SKLearnGridSearch(GridSearch):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_combinations(self, parameters: dict):
        import itertools

        # Generate every possible combination of the provided hyper parameters.
        master_list = []
        params_list = []
        for estimator in parameters:
            param_dict = parameters.get(estimator)
            # assert_condition(condition=isinstance(param_dict, dict),
            #                  source=self,
            #                  source=self,
            #                  message='Parameter dictionary is not of type dictionary for estimator:' + estimator)
            for params in param_dict:
                # Append only the parameters to create a master list
                master_list.append(param_dict.get(params))

                # Append the estimator name followed by the parameter to create a ordered list.
                # Ordering of estimator.parameter corresponds to the value in the resultant grid tuple
                params_list.append(''.join([estimator, '.', params]))

        grid = itertools.product(*master_list)
        return grid, params_list
