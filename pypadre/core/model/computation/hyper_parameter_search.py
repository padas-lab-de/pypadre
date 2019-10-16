from pypadre.core.model.computation.computation import Computation


class HyperParameterGrid(Computation):

    def __init__(self, *, component, execution, result, **kwargs):
        super().__init__(component=component, execution=execution, result=result, **kwargs)
        self._parameter_names = kwargs.pop('parameter_names', None)

    def iter_result(self):
        # FIXME CHRISTOFER
        for element in self._result:
            execution_params = dict()
            for param, idx in zip(self._parameter_names, range(0, len(self._parameter_names))):
                execution_params[param] = element[idx]

            yield execution_params
