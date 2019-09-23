import unittest

# noinspection PyMethodMayBeStatic
import numpy as np
from padre.PaDREOntology import PaDREOntology

from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.core.model.dataset.dataset import Dataset
from pypadre.core.model.experiment import Experiment
from pypadre.pod.experimentcreator import ExperimentCreator
from pypadre.pod.importing.dataset.dataset_import import SKLearnLoader

test_numpy_array = np.array([[1.0, "A", 2],
                             [2.0, "B", 2],
                             [3.0, "A", 3],
                             [3.0, "C", 4]])


class TestSKLearnPipeline(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSKLearnPipeline, self).__init__(*args, **kwargs)

    def test_default_sklearn_pipeline(self):

        # TODO clean up experiment creator
        experiment_helper = ExperimentCreator()
        pipeline = SKLearnPipeline(pipeline=experiment_helper.create_test_pipeline(['SVC']))

        loader = SKLearnLoader()
        iris = loader.load("sklearn", utility="load_iris")
        experiment = Experiment(dataset=iris, pipeline=pipeline)

        experiment.execute()
        print("test")
        # TODO asserts and stuff


if __name__ == '__main__':
    unittest.main()
