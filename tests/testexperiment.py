import unittest

from padre.experiment import DictExperimentVisitor, ListExperimentVisitor, SelectExperimentVisitor

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from padre.visitors.scikit.scikitpipeline import SciKitPipelineVisitor

class TestExperimentVisitor(unittest.TestCase):
    def test_extract_string(self):
        class TestClass(object):
            def __init__(self):
                self.test = "attribute"

        v = DictExperimentVisitor({"test": "found"})
        d = v.extract(TestClass(), {})

        self.assertIn("found", d)
        self.assertEqual(d["found"], "attribute")

    def test_extract_dict(self):
        class TestClass(object):
            def __init__(self):
                self.test = {"rec": "attribute"}

        v = DictExperimentVisitor({"test": {"rec" : "found"}})
        d = v.extract(TestClass(), {})

        self.assertIn("found", d)
        self.assertEqual(d["found"], "attribute")


class TestSciKitExperimentVisitor(unittest.TestCase):

    def test_extract_pipeline(self):
        estimators = [('reduce_dim', PCA()), ('clf', SVC())]
        pipe = Pipeline(estimators)

        d = SciKitPipelineVisitor(pipe)

        print(d)
        print(pipe)




if __name__ == '__main__':
    unittest.main()
