from pypadre.core.model.computation.computation import Computation


class HyperParameterSearch(Computation):

    def __init__(self, *, component, execution, result, **kwargs):
        super().__init__(component=component, execution=execution, result=result, **kwargs)
        pass
