from pypadre.core.model.computation.computation import Computation


class HyperParameterGrid(Computation):

    def __init__(self, *, component, execution, result, **kwargs):
        super().__init__(component=component, execution=execution, result=result, **kwargs)
        self.parameters = kwargs.pop('parameters', None)
        self.parameter_names = kwargs.pop('parameter_names', None)
        self.branch = kwargs.get('branch', False)
        # Parameters should not be None for Hyperparameter search
        assert(self.parameters is not None)
        assert(self.parameter_names is not None)

    @property
    def result(self):
        for element in self.parameters:
            execution_params = dict()
            for param, idx in zip(self.parameter_names, range(0, len(self.parameter_names))):
                execution_params[param] = element[idx]

            yield execution_params

    @property
    def branch(self):
        return self.branch
