
from sklearn.pipeline import Pipeline

from .experimentschema import PipelineSchema
from .generalvisitor import generalizeVisitor
from .visitor import AlgorithmVisitor, ListVisitor, SubpathVisitor, SelectVisitor

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
