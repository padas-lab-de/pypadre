
from sklearn.pipeline import Pipeline

from pypadre.core.visitors.experimentschema import PipelineSchema
from pypadre.core.visitors.generalvisitor import generalizeVisitor
from pypadre.core.visitors.visitor import AlgorithmVisitor, ListVisitor, SubpathVisitor, SelectVisitor

SciKitPipelineVisitor = {
                "steps":
                    ListVisitor("steps",
                                (
                                              None,
                                              generalizeVisitor(AlgorithmVisitor())
                                          )
                                ),
                "__doc__":
                    "doc"
            }


SciKitSelector = {
    Pipeline : SciKitPipelineVisitor,
    None : SubpathVisitor("steps[]", generalizeVisitor(AlgorithmVisitor()))
}


SciKitVisitor = SelectVisitor(SciKitSelector, PipelineSchema)
