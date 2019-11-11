from typing import List

from pypadre.core.events.events import connect
from pypadre.core.model.computation.pipeline_output import PipelineOutput
from pypadre.core.model.split.split import Split
from pypadre.pod.repository.i_repository import IPipelineOutputRepository
from pypadre.pod.service.base_service import ModelServiceMixin


class PipelineOutputService(ModelServiceMixin):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[IPipelineOutputRepository], **kwargs):
        super().__init__(model_clz=Split, backends=backends, **kwargs)

        @connect(PipelineOutput)
        def put(obj, **kwargs):
            self.put(obj, **kwargs)
        self.save_signal_fn(put)

        @connect(PipelineOutput)
        def delete(obj, **kwargs):
            self.delete(obj)
        self.save_signal_fn(delete)
