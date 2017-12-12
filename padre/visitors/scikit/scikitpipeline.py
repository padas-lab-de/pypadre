from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from padre.experimentschema import PipelineSchema
from padre.experimentvisitors import DictVisitor, ListVisitor, SelectVisitor, CombineVisitor, ConstantVisitor, \
    SubpathVisitor

from sklearn.svm import SVC
from sklearn.decomposition import PCA

EstimatorVisitor = {"__doc__": "doc"}

PCA_Visitor = CombineVisitor([
    EstimatorVisitor,
    ConstantVisitor(
        {"algorithm": "PCA"}
    )
])

SVC_Visitor = CombineVisitor([
    EstimatorVisitor,
    {
        "degree": "degree",
        "C": "C",
        "kernel": "kernel"
    },
    ConstantVisitor(
        {"algorithm": "SVC"}
    )
])


SciKitEstimatorSelector = {
    PCA : PCA_Visitor,
    SVC : SVC_Visitor
}


SciKitPipelineVisitor = DictVisitor(
            {
                "steps":
                    ListVisitor("steps",
                                (
                                              None,
                                              SelectVisitor(SciKitEstimatorSelector)
                                          )
                                ),
                "__doc__":
                    "doc"
            }
        )

KNeighborsClassifier_Visitor = CombineVisitor([
    EstimatorVisitor,
    ConstantVisitor(
        {"algorithm": "KNeighbors"}
    )
])

SciKitClassifierSelector = {
    KNeighborsClassifier : KNeighborsClassifier_Visitor
}

SciKitSelector = {
    Pipeline : SciKitPipelineVisitor,
}
SciKitSelector.update({c : SubpathVisitor("steps[]", SciKitEstimatorSelector[c]) for c in SciKitEstimatorSelector})
SciKitSelector.update({c : SubpathVisitor("steps[]", SciKitClassifierSelector[c]) for c in SciKitClassifierSelector})



SciKitVisitor = SelectVisitor(SciKitSelector, PipelineSchema)