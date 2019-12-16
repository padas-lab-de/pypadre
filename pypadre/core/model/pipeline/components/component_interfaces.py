from typing import List

from pypadre.core.util.inheritance import SuperStop


class IConsumer(SuperStop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def consumes(self) -> str:
        raise NotImplementedError()


class IProvider(SuperStop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def provides(self) -> List[str]:
        raise NotImplementedError()
