from pypadre.core.model.computation.hyper_parameter_search import HyperParameterSearch


class PipelineParameters:

    def __init__(self, parameter_map):
        self._map = parameter_map

    @property
    def map(self):
        return self._map

    def get_for(self, component):
        return self.get(component.id)

    def get(self, identifier):
        if hasattr(self.map, identifier):
            return self.map.get(identifier)
        else:
            # TODO look by names etc. instead
            return {}

    def is_grid_search(self, component):
        pass

    def combinations(self, *, execution, component, predecessor):

        from pypadre.core.model.computation.computation import Computation
        parameters = self.get_for(component)

        # TODO parameters could also be a generator function if this is the case just call it and
        #  check if combinations are valid regarding the schema
        # TODO look through the parameters and add combination if one of it is a iterable
        #  instead of an expected parameter type
        # TODO expected parameter types are to be given in the component schema FIXME Christofer

        # If the parameters are returned within a function
        hyperparameters = parameters() if callable(parameters) else parameters
        assert(isinstance(hyperparameters, dict))

        # The params_list contains the names of the hyperparameters in the grid
        grid, params_list = self.create_combinations(hyperparameters)

        # We need to either create multiple components
        # based on the number of elements in the grid or iterate of the grid
        for element in grid:

            # Set the params to the component either via a dictionary all together or individually
            execution_params = dict()
            for param, idx in zip(params_list, range(0, len(params_list))):
                execution_params[param] = element[idx]

            # TODO set the parameters to the component
            # yield Computation(component=component, execution=execution, parameters=execution_params, branch=False)
            # TODO Decide whether the grid creation logic should be within the HyperParameter Search Component or not
            yield HyperParameterSearch(component=component, execution=execution,
                                       parameters=execution_params, predecessor=predecessor, branch=False)

    def create_combinations(self, parameters):
        """
        Creates all the possible combinations of hyper parameters passed
        :param parameters: Dictionary containing hyperparameter names and their possible values
        :return: A list containing all the combinations and a list containing the hyperparameter names
        """

        import itertools

        param_dict = dict()
        params_list = []
        master_list = []

        for parameter in parameters:

            # Append only the parameters to create a master list
            parameter_values = param_dict.get(parameter)

            # If the parameter value is a dict wrap it in a dictionary,
            # so that the values of the dictionary are not unpacked
            parameter_values = [parameter_values] if isinstance(parameter_values, dict) else parameter_values

            master_list.append(parameter_values)

            # Append the estimator name followed by the parameter to create a ordered list.
            # Ordering of estimator.parameter corresponds to the value in the resultant grid tuple
            params_list.append(parameter)

        # Create the grid
        grid = itertools.product(*master_list)

        return grid, params_list


        # # Generate every possible combination of the provided hyper parameters.
        # master_list = []
        # params_list = []
        # for estimator in parameters:
        #     param_dict = parameters.get(estimator)
        #     # assert_condition(condition=isinstance(param_dict, dict),
        #     #                  source=self,
        #     #                  message='Parameter dictionary is not of type dictionary for estimator:' + estimator)
        #     for params in param_dict:
        #         # Append only the parameters to create a master list
        #         master_list.append(param_dict.get(params))
        #
        #         # Append the estimator name followed by the parameter to create a ordered list.
        #         # Ordering of estimator.parameter corresponds to the value in the resultant grid tuple
        #         params_list.append(''.join([estimator, '.', params]))
        #
        # grid = itertools.product(*parameters)
        #
        # # Get the total number of iterations
        # grid_size = 1
        # for idx in range(0, len(parameters)):
        #     grid_size *= len(parameters[idx])
        #
        # # Starting index
        # curr_executing_index = 1
        #
        # # For each tuple in the combination create a run
        # for element in grid:
        #     for param, idx in zip(params_list, range(0, len(params_list))):
        #         split_params = param.split(sep='.')
        #         estimator = workflow._pipeline.named_steps.get(split_params[0])
        #
        #         if estimator is None:
        #             assert_condition(condition=estimator is not None, source=self,
        #                           message=f"Estimator {split_params[0]} is not present in the pipeline")
        #             break
        #
        #         estimator.set_params(**{split_params[1]: element[idx]})
        #
        #     r = Run(self, workflow, **dict(self._metadata))
