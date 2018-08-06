"""
This file shows an example on how to use the pypadre app.
"""
import unittest

from padre.ds_import import load_sklearn_toys
from padre.experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class TestBasic(unittest.TestCase):

    def test_SVC_SKLeanr(self):
        ds = [i for i in load_sklearn_toys()]
        ex = Experiment(name="Test Experiment SVM",
                        description="Testing Support Vector Machines via SKLearn Pipeline\n"
                                    "- no persisting via a backend\n"
                                    "- manual data set loading\n"
                                    "- default parameters",
                        dataset=ds[2],
                        keep_runs=True,
                        workflow=Pipeline([('clf', SVC(probability=True))]))
        ex.run()
        assert len(ex.runs) == 1
        # TODO: we currently do not store splits and we do have no means to evaluate performance measures
        # from runs where there is NO BACKEND! If there is no backend, we should keep runs and splits in memory
