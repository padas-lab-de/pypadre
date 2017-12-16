
import unittest

from sklearn.neighbors import KNeighborsClassifier

from padre.experimentvisitors import DictVisitor, ListVisitor, SelectVisitor

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from padre.visitors.scikit.scikitpipeline import SciKitVisitor


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
        estimators = [('reduce_dim', PCA()), ('clf', SVC())]
        pipe = Pipeline(estimators)

        d = SciKitVisitor(pipe)

        self.assertIn("steps", d[0])

        print(d)

    def test_extract_svc(self):
        estimator = PCA()

        d = SciKitVisitor(estimator)

        print(d)

    def test_extract_kneighborsclassifier(self):
        knn = KNeighborsClassifier()

        d = SciKitVisitor(knn)

        print(d)



if __name__ == '__main__':
    unittest.main()
