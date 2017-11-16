from padre.experiment import DictExperimentVisitor, ListExperimentVisitor, SelectExperimentVisitor

from sklearn.svm import SVC
from sklearn.decomposition import PCA

PCA_Visitor = {}
SVC_Visitor = {"degree": "deg", "C": "C"}


def selectSciKitEstimatorVisitor(o):
    if isinstance(o, PCA):
        return PCA_Visitor
    elif isinstance(o, SVC):
        return SVC_Visitor
    else:
        raise TypeError("Unsupported Estimator encountered: " + str(type(o)))


SciKitPipelineVisitor = DictExperimentVisitor(
            {
                "steps":
                    ListExperimentVisitor("steps",
                                          (
                                              None,
                                              SelectExperimentVisitor(selectSciKitEstimatorVisitor)
                                          )
                                          )
            }
        )