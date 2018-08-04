"""
This file shows an example on how to use the pypadre app.
"""
import unittest

from padre.ds_import import load_sklearn_toys
from padre.app import pypadre
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class TestSklearn(unittest.TestCase):

    def test_SVC(self):
        pypadre.set_printer(print)
        try:
            pypadre.datasets.list_datasets()
            ds = pypadre.datasets.get_dataset("http://localhost:8080/api/datasets/5")
        except:
            ds = [i for i in load_sklearn_toys()][2]

        ex = pypadre.experiments.run(name="Test Experiment SVM",
                                     description="Testing Support Vector Machines via SKLearn Pipeline",
                                     dataset=ds,
                                     workflow=Pipeline([('clf', SVC(probability=True))]))
        print("========Available experiments=========")
        for idx, ex in enumerate(pypadre.experiments.list_experiments()):
            print("%d: %s" % (idx, str(ex)))
            for idx2, run in enumerate(pypadre.experiments.list_runs(ex)):
                print("\tRun: %s" % str(run))

