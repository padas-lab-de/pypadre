"""
This file shows an example on how to use PyPaDRE via decorators defining multipe experiments.

Note: it is a proof of concept now rather than a test.
"""
# Note that we want to include all decorator at once using package import
import unittest

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from padre import *
from padre.app import pypadre
from padre.ds_import import load_sklearn_toys


@Workflow(exp_name="test_AppExp",
          description="Test experiment with decorators")
def create_test_pipeline():
    estimators = [('clf', SVC(probability=True))]
    return Pipeline(estimators)


@Workflow(exp_name="test_AppExp2",
          description="Test2 experiment with decorators")
def create_test_pipeline():
    estimators = [('clf', SVC(probability=True, C=0.2))]
    return Pipeline(estimators)


@Dataset(exp_name="test_AppExp")
def get_dataset():
    ds = [i for i in load_sklearn_toys()][2]
    return ds


@Dataset(exp_name="test_AppExp2")
def get_dataset():
    ds = [i for i in load_sklearn_toys()][3]
    return ds


class TestApp(unittest.TestCase):

    def _del_test_exps(self):
        pypadre.experiments.delete_experiments("test.*")

    def test_delete(self):
        self._del_test_exps() # every test should begin with that
        ex_list = pypadre.experiments.list_experiments()
        for ex in ex_list:
            self.assertTrue(not ex.startswith("test"))

    def test_sklearn_SVC(self):
        self._del_test_exps()
        ex_list = pypadre.experiments.list_experiments()
        # Do basic experiment
        ds = [i for i in load_sklearn_toys()][2]
        ex = pypadre.experiments.run(name="test_APPSKLEARN",
                                     description="Testing Support Vector Machines via SKLearn Pipeline",
                                     dataset=ds,
                                     workflow=Pipeline([('clf', SVC(probability=True))]))
        self.assertTrue(ex is not None)
        # check its storage
        ex_list_n = pypadre.experiments.list_experiments()
        self.assertTrue(len(ex_list)+1 == len(ex_list_n))
        self.assertTrue("test_APPSKLEARN" in ex_list_n)
        # Delete experiment and chekc if correct
        pypadre.experiments.delete_experiments("test_APPSKLEARN")
        ex_list_n = pypadre.experiments.list_experiments()
        self.assertTrue("test_APPSKLEARN" not in ex_list_n)


    def test_decorators(self):
        self._del_test_exps()
        ex = pypadre.experiments.run(decorated=True)
        # Now test existence at experiments, list experiments etc.
