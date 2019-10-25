# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

from types import GeneratorType
from typing import Optional, Iterable

from pypadre.core.base import MetadataEntity, ChildEntity
from pypadre.core.model.computation.run import Run
from pypadre.core.model.generic.i_model_mixins import IProgressable, IStoreable
from pypadre.core.printing.tablefyable import Tablefyable


class Computation(IStoreable, IProgressable, MetadataEntity, ChildEntity, Tablefyable):
    COMPONENT_ID = "component_id"
    COMPONENT_CLASS = "component_class"
    RUN_ID = "run_id"
    PREDECESSOR_ID = "predecessor_computation_id"

    @classmethod
    def _tablefy_register_columns(cls):
        pass

    def __init__(self, *, component, run: Run, predecessor: Optional[Computation] = None, result_format=None, result,
                 parameters=None, branch=False, **kwargs):
        if parameters is None:
            parameters = {}

        # Add defaults
        defaults = {}

        # Merge defaults
        metadata = {**defaults, **kwargs.pop("metadata", {}), **{self.COMPONENT_ID: component.id,
                                                                 self.COMPONENT_CLASS: str(component.__class__),
                                                                 self.RUN_ID: str(run.id),
                                                                 self.PREDECESSOR_ID: predecessor.id if predecessor is not None else None}}

        super().__init__(parent=run, metadata=metadata, **kwargs)
        self._component = component
        self._result = result
        # Todo add result schema (this has to be given by the component) At best a component can return directly a computation object
        # Todo allow for multiple predecessors
        self._predecessor = predecessor
        self._parameters = parameters
        self._branch = branch
        self._format = result_format

        if self.branch and not isinstance(self.result, GeneratorType) and not isinstance(self.result, Iterable):
            raise ValueError("Can only branch if the computation produces a list or generator of data")

    @property
    def type(self):
        # TODO this should be done via ontology
        return str(self.__class__)

    @property
    def format(self):
        # TODO Use Ontology here (Maybe even get this by looking at owlready2)
        return self._format if self._format is not None else str(self.__class__)

    @property
    def run(self):
        return self.parent

    @property
    def component(self):
        return self._component

    @property
    def predecessor(self):
        return self._predecessor

    @property
    def parameters(self):
        return self._parameters

    @property
    def run(self):
        return self.parent

    @property
    def branch(self):
        return self._branch

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, result):
        self._result = result

    def iter_result(self):
        if self.branch:
            return self.result
        else:
            return [self.result]
