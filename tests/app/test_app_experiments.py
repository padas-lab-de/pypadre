"""
This file shows an example on how to use PyPaDRE via decorators defining multipe experiments.

Note: it is a proof of concept now rather than a test.
"""
# Note that we want to include all decorator at once using package import
import unittest

from pypadre.app import p_app
from tests.app.experiments_decorated import *
from pypadre.pod.importing.dataset.ds_import import load_sklearn_toys


class TestApp(unittest.TestCase):

    def _del_test_exps(self):
        p_app.experiments.delete_experiments("test.*")

    def test_delete(self):
        """
        test delete as prerequsite for upcoming tests
        :return:
        """
        self._del_test_exps() # every test should begin with that
        ex_list = p_app.experiments.list_experiments()
        for ex in ex_list:
            self.assertTrue(not ex.startswith("test"))

    def test_sklearn_SVC(self):
        """
        test basic experiment handling
        :return:
        """
        self._del_test_exps()
        ex_list = p_app.experiments.list_experiments()
        # Do basic experiment
        ds = [i for i in load_sklearn_toys()][2]
        ex = p_app.experiments.run(name="test_APPSKLEARN",
                                   description="Testing Support Vector Machines via SKLearn Pipeline",
                                   dataset=ds,
                                   workflow=Pipeline([('clf', SVC(probability=True))]))
        self.assertTrue(ex is not None)
        # check its storage
        ex_list_n = p_app.experiments.list_experiments()
        self.assertTrue(len(ex_list)+1 == len(ex_list_n))
        self.assertTrue("test_APPSKLEARN" in ex_list_n)
        # Delete experiment and chekc if correct
        p_app.experiments.delete_experiments("test_APPSKLEARN")
        ex_list_n = p_app.experiments.list_experiments()
        self.assertTrue("test_APPSKLEARN" not in ex_list_n)

    def test_decorators(self):
        """
        test the use of decorators (via importing experiments_decorated)
        :return:
        """
        self._del_test_exps()
        ex_list = p_app.experiments.list_experiments()
        ex = p_app.experiments.run(decorated=True)
        self.assertTrue(ex is not None)
        # check its storage
        ex_list_n = p_app.experiments.list_experiments()
        self.assertTrue(len(ex_list) + 2 == len(ex_list_n))
        self.assertTrue("test_AppExp" in ex_list_n)
        self.assertTrue("test_AppExp2" in ex_list_n)
        # Now test existence at experiments, list experiments etc.
        p_app.experiments.delete_experiments("test_AppExp.*")
        ex_list_n = p_app.experiments.list_experiments()
        self.assertTrue("test_AppExp" not in ex_list_n)
        self.assertTrue("test_AppExp2" not in ex_list_n)
