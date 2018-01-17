
from sklearn.pipeline import Pipeline

from padre.experimentschema import PipelineSchema
from padre.experimentvisitors import DictVisitor, ListVisitor, SelectVisitor, CombineVisitor, ConstantVisitor, \
    SubpathVisitor, AlgorithmVisitor

GeneralVisitor = {"__doc__": "doc"}


SciKitPipelineVisitor = {
                "steps":
                    ListVisitor("steps",
                                (
                                              None,
                                              CombineVisitor([GeneralVisitor, AlgorithmVisitor()])
                                          )
                                ),
                "__doc__":
                    "doc"
            }


SciKitSelector = {
    Pipeline : SciKitPipelineVisitor,
}


SciKitVisitor = SelectVisitor(SciKitSelector, SubpathVisitor("steps[]", CombineVisitor([GeneralVisitor, AlgorithmVisitor()])), PipelineSchema)