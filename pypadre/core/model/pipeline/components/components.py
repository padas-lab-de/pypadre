from typing import Callable, List, Optional

from pypadre.core.model.code.code_mixin import Function, GitIdentifier
from pypadre.core.model.generic.custom_code import ProvidedCodeHolderMixin
from pypadre.core.model.pipeline.components.component_mixins import SplitComponentMixin, PipelineComponentMixin, \
    ParameterizedPipelineComponentMixin
from pypadre.core.model.split.split import Split
from pypadre.core.model.split.splitter import default_split
from pypadre.core.util.utils import unpack


class PipelineComponent(PipelineComponentMixin):
    """
    Generic pipeline component taking a consumes and provides definition.
    """

    def __init__(self, *, name: str, consumes: str = None, provides: List[str], metadata: Optional[dict] = None,
                 **kwargs):
        super().__init__(name=name, metadata=metadata, **kwargs)
        self._consumes = consumes
        self._provides = provides

    @property
    def consumes(self) -> str:
        return self._consumes

    @property
    def provides(self) -> List[str]:
        return self._provides


class SplitComponent(SplitComponentMixin):
    """
    Split component holding a custom split definition.
    """

    def __init__(self, name="splitter", code=None, **kwargs):
        # TODO wrap in something better than function
        if code is Callable:
            code = Function(fn=code, repository_identifier=GitIdentifier())
        super().__init__(name=name, code=code, **kwargs)


class DefaultSplitComponent(ProvidedCodeHolderMixin, SplitComponent, ParameterizedPipelineComponentMixin):

    def __init__(self, **kwargs):
        super().__init__(name="default_split", fn=self.call, **kwargs)

    def call(self, ctx, **kwargs):
        return default_split.call(parameters=kwargs, **ctx)


def custom_splitting_wrapper(fn: Callable):
    """
     Custom split wrapper to give a template for users.
    """

    def custom_split(ctx, **kwargs):
        num = -1
        train_idx, test_idx, val_idx = fn(ctx, **kwargs)
        (data, run, component, predecessor) = unpack(ctx, "data", "run", "component", ("predecessor", None))
        yield Split(run=run, num=++num, train_idx=train_idx, test_idx=test_idx,
                    val_idx=val_idx, component=component, predecessor=predecessor)

    return custom_split
