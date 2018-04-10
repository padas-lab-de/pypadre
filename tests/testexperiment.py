
import unittest

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline

from padre.visitors import DictVisitor
from padre.visitors.scikit import SciKitVisitor


class TestExperimentVisitor(unittest.TestCase):
    def test_extract_string(self):
        class TestClass(object):
            def __init__(self):
                self.test = "attribute"

        v = DictVisitor({"test": "found"})
        d = v.extract(TestClass(), {})

        self.assertIn("found", d)
        self.assertEqual(d["found"], "attribute")

    def test_extract_dict(self):
        class TestClass(object):
            def __init__(self):
                self.test = {"rec": "attribute"}

        v = DictVisitor({"test": {"rec" : "found"}})
        d = v.extract(TestClass(), {})

        self.assertIn("found", d)
        self.assertEqual(d["found"], "attribute")


class TestSciKitExperimentVisitor(unittest.TestCase):

    def test_extract_pipeline(self):
        estimators = [('step1', Ridge())]
        pipe = Pipeline(estimators)

        d = SciKitVisitor(pipe)

        self.assertIn("steps", d[0])
        for s in d[0]["steps"]:
            for k,v in s.items():
                print(k,v)

    def test_extract_linear_regression(self):
        lreg = LinearRegression()

        d = SciKitVisitor(lreg)
        for s in d[0]["steps"]:
            for k,v in s.items():
                print(k,v)

    def test_extract_ridge_regression(self):
        r = Ridge()

        d = SciKitVisitor(r)

        print(d)



if __name__ == '__main__':
    unittest.main()
