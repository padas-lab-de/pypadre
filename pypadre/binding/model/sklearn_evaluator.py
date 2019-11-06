import numpy as np
from padre.PaDREOntology import PaDREOntology

from pypadre import _version, _name
from pypadre.core.base import phases
from pypadre.core.model.computation.evaluation import Evaluation
from pypadre.core.model.pipeline.components import IProvidedComponent, EvaluatorComponent, \
    ParameterizedPipelineComponent
from pypadre.core.util.utils import unpack
from pypadre.core.visitors.mappings import name_mappings, alternate_name_mappings

# Constant strings that are used in creating the results dictionary
DATASET_NAME = 'dataset'
SPLIT_NUM = "split_num"
TRAINING_SAMPLES = "training_samples"
TESTING_SAMPLES = "testing_samples"
TRAINING_IDX = "training_indices"
TESTING_IDX = "testing_indices"
TYPE = "type"


def evaluate(ctx, **kwargs):
    (component,) = unpack(ctx, "component")
    return component.evaluate(ctx, **kwargs)


class SKLearnEvaluator(IProvidedComponent, EvaluatorComponent, ParameterizedPipelineComponent):
    """
    This class takes the output of an sklearn workflow which represents the fitted model along with the corresponding split,
    report and save all possible results that allows for common/custom metric computations.
    """

    def __init__(self, **kwargs):
        super().__init__(package=__name__, fn_name="evaluate",  requirement=_name.__name__,
                         version=_version.__version__, name='SKLearnEvaluator', **kwargs)

    def evaluate(self, ctx, **kwargs):
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

        y_predicted_probabilities = None

        self.send_error(condition=final_estimator_type is None,
                        message='Final estimator could not be found in names or alternate names')

        if final_estimator_type == 'Classification' or \
                (final_estimator_type == 'Neural Network' and np.all(np.mod(y_predicted, 1)) == 0):
            type_ = PaDREOntology.SubClassesExperiment.Classification.value

            if compute_probabilities:
                y_predicted_probabilities = model.predict_proba(split.test_features)
                self.send_log(mode='probability', pred=y_predicted, truth=y, probabilities=y_predicted_probabilities,
                              message="Computing and saving the prediction probabilities")
                y_predicted_probabilities = y_predicted_probabilities.tolist()
        else:
            type_ = PaDREOntology.SubClassesExperiment.Regression.value

        if self.is_scorer(model):
            self.send_start(phase=f"sklearn.scoring.testset")
            score = model.score(split.test_features, y, )
            self.send_stop(phase=f"sklearn.scoring.testset")
            self.send_log(keys=["test score"], values=[score], message="Logging the testing score")

        results = self.create_results_dictionary(split_num=split.number, train_idx=train_idx, test_idx=test_idx,
                                                 dataset=split.dataset.name,
                                                 truth=y.tolist(), predicted=y_predicted.tolist(), type_= type_,
                                                 probabilities=y_predicted_probabilities)


        # TODO results as object?

        return Evaluation(training=predecessor, result_format=type_, result=results, component=component, run=run,
                          parameters=kwargs)

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

    @staticmethod
    def create_results_dictionary(*, split_num:int, train_idx: list, test_idx: list, dataset: str, type_: str,
                                  truth: list, predicted: list, probabilities: list):
        from pypadre.core.model.pipeline.components import EvaluatorComponent

        results = dict()
        results[DATASET_NAME] = dataset
        results[TRAINING_SAMPLES] = len(train_idx)
        results[TESTING_SAMPLES] = len(test_idx)
        results[SPLIT_NUM] = split_num
        results[TRAINING_IDX] = train_idx
        results[TESTING_IDX] = test_idx
        results[TYPE] = type_

        # Whether the probabilities of predictions should be written
        write_probabilites = True
        if probabilities is None or len(probabilities) != len(truth):
            write_probabilites = False

        # Predictions are a dictionary
        predictions = dict()
        for idx, test_row_index in enumerate(test_idx):
            # Each prediction is a dictionary with the row id as the key. This key would point to the exact
            # row that was tested
            # The dictionary contains the truth value, the predicted value and if there are probabilities,
            # the probabilities of the classes
            curr_row_dict = dict()
            curr_row_dict[EvaluatorComponent.TRUTH] = truth[idx]
            curr_row_dict[EvaluatorComponent.PREDICTED] = predicted[idx]
            curr_row_dict[EvaluatorComponent.PROBABILITIES] = probabilities[idx] if write_probabilites is True else []
            predictions[test_row_index] = curr_row_dict

        # Add the predictions to the results dictionary
        results[EvaluatorComponent.PREDICTIONS] = predictions

        return results
5

