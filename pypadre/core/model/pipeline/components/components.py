from typing import Callable, List, Optional

from pypadre.core.model.code.codemixin import Function
from pypadre.core.model.pipeline.components.component_mixins import SplitComponentMixin, PipelineComponentMixin
from pypadre.core.model.split.split import Split
from pypadre.core.model.split.splitter import Splitter
from pypadre.core.util.utils import unpack


class CustomSplit(Function):

    def __init__(self, *, fn: Callable):
        def custom_split(ctx, **kwargs):

            def splitting_iterator():
                num = -1
                train_idx, test_idx, val_idx = fn(ctx, **kwargs)
                (data, run, component, predecessor) = unpack(ctx, "data", "run", "component", ("predecessor", None))
                yield Split(run=run, num=++num, train_idx=train_idx, test_idx=test_idx,
                            val_idx=val_idx, component=component, predecessor=predecessor)
            return splitting_iterator()
        super().__init__(fn=custom_split)


class SplitComponent(SplitComponentMixin):

    def __init__(self, name="default_splitter", code=None, **kwargs):
        if code is None:
            code = Splitter()
        if code is Callable:
            code = Function(fn=code)
        super().__init__(name=name, code=code, **kwargs)


class PipelineComponent(PipelineComponentMixin):

    def __init__(self, *, name: str, consumes: str=None, provides: List[str], metadata: Optional[dict] = None, **kwargs):
        super().__init__(name=name, metadata=metadata, **kwargs)
        self._consumes = consumes
        self._provides = provides

    @property
    def consumes(self) -> str:
        return self._consumes

    @property
    def provides(self) -> List[str]:
        return self._provides
