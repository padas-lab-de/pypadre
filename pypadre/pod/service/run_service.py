from typing import List

from pypadre.core.events.events import connect
from pypadre.core.model.computation.run import Run
from pypadre.pod.repository.i_repository import IRunRepository
from pypadre.pod.service.base_service import ModelServiceMixin


class RunService(ModelServiceMixin):
    """
    Class providing commands for managing datasets.
    """

    def __init__(self, backends: List[IRunRepository], **kwargs):
        super().__init__(model_clz=Run, backends=backends, **kwargs)

        @connect(Run)
        def put(obj, **kwargs):
            self.put(obj)
        self.save_signal_fn(put)

        @connect(Run)
        def delete(obj, **kwargs):
            self.delete(obj)
        self.save_signal_fn(delete)
