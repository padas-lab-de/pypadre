import unittest

# noinspection PyMethodMayBeStatic
import numpy as np

from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.core.events.events import connect_base_signal, LOG_EVENT
from pypadre.core.model.experiment import Experiment
from pypadre.core.model.pipeline.components import SplitComponent
from pypadre.pod.importing.dataset.dataset_import import SKLearnLoader

test_numpy_array = np.array([[1.0, "A", 2],
                             [2.0, "B", 2],
                             [3.0, "A", 3],
                             [3.0, "C", 4]])


def create_test_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    estimators = [('SVC', SVC(probability=True))]
    return Pipeline(estimators)


class TestSKLearnPipeline(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSKLearnPipeline, self).__init__(*args, **kwargs)

    def test_default_sklearn_pipeline(self):
        def log(sender, *, message, log_level="", **kwargs):
            if log_level is "":
                print(str(sender) + ": " + message)
            else:
                print(log_level.upper() + ": " + str(sender) + ": " + message)

        def log_event(sender, *, signal, **kwargs):
            log(sender, message="Triggered " + str(signal.name) + " with " + str(kwargs))

        connect_base_signal("log", log)
        connect_base_signal(LOG_EVENT, log_event)

        # TODO clean up experiment creator
        pipeline = SKLearnPipeline(pipeline=create_test_pipeline())

        loader = SKLearnLoader()
        iris = loader.load("sklearn", utility="load_iris")
        experiment = Experiment(dataset=iris, pipeline=pipeline)

        experiment.execute()
        # TODO asserts and stuff


if __name__ == '__main__':
    unittest.main()
