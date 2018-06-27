******************
Experiment Creator
******************

Summary
=======


This Class helps is creating experiments and performing the necessary validations on the experiment parameters.
The class abstracts the acquisition methods of the datasets too. The main aim of the class is to wrap the experiments
and enable the execution of multiple experiment sequentially. It also enables the execution of a single experiment
on multiple datasets.


Function List
=============

Initialization Functions
------------------------

* initialize_workflow_components: This function reads all the estimators available for the system. This is obtained from the mappings.json file.
* intialize_estimator_parameter_implementation: This function reads all the parameters available to the function and also identifies the implementation of the function for dynamic loading. This function also returns the possible datatypes for each parameter.
* initialize_dataset_names: This function reads all the names of all the available datasets

Get Functions
-------------
* get_dataset_names: This function returns the names of all the available datasets in the system.
* get_estimators: This function return the names of all the available estimators
* get_estimator_params(estimator_name): This function returns all the parameters belonging to an estimator.
* get_estimator_object(estimator_name): Returns an object of the estimator as specified in the mappings.json implementation
* get_local_dataset(name): Returns the instance of the dataset
* get_param_values(experiment_name): Returns all the non-default parameters set for that experiment in the same format that is used to set the parameters

Set Functions
-------------
* set_param_values(experiment_name, param): Used to set the parameters for a particular experiment. The param contains the name of the estimator and the parameters to be set for that estimator. The parameters for the experiment are set after validation. The param could be a string or a dictionary. The format is given by estimator_name.parameter_name:[parameter_value1, parameter_value2, ...., parameter_valueN]. Multiple parameters can be chained by the '|' operator. Any parameter or value that could not be inserted will be discarded.
* set_parameters(estimator_object, estimator_name, param_dict): This function sets the parameters for the estimator object after validating that the estimator has such an available parameter.

Validation Functions
--------------------
* validate_parameters(param_dict): Validates all the parameters given for estimators and returns the validated parameters
* validate_pipeline(workflow): This function checks whether each component in the pipeline has a
        fit and fit_transform attribute or transform attribute. The last estimator
        can be none and if it is not none, has to implement a fit attribute. Returns Boolean based on whether the workflow is valid or not.

Experiment Execution Functions
------------------------------
* execute_experiments: Executes all the created experiments
* do_experiments(experiment_datasets): Executes the same experiment with multiple datasets

Properties
----------
* experiments: Returns all the experiments that have been created
* experiment_names: Returns the names of all the experiments that have been created
* components: Returns all the names of estimators available to the user

Conversion Functions
--------------------
convert_param_string_to_dictionary(param): This function converts a string given as input into the appropriate parameter name and parameter values. The string is of the format estimator_name.parameter_name:[value1, value2, ..., valueN]
typecast_variable(param, allowed_types): This function typecasts a string variable into one of the options in allowed_types.

Create Functions
---------------
create_experiment(name, description, dataset, workflow, backend, params): creates an experiment with the experiment name(if experiment name is not given, a default one will be created by the system), its description, the name of the dataset or the dataset object itself, the workflow, backend and the parameters of the estimators that need to be changed, if any.
create_test_pipeline(estimator_list, params): Creates a workflow from the given list of estimators. The params argument is optional and can be used to set the parameters of the estimators.

Experiment Creation from the Command Line Interface
-------------------------

#. Display all the available estimators using "**components**" command
#. Display all the available datasets using "**dataset**" command
#. Create an experiment using the "**create_experiment**" command with parameters a unique experiment_name, a description of the experiment, the name of the dataset to be used(names obtained in the above step) and the list of estimators to beused for the workflow(estimators are separated by a comma)
#. Set experiment parameters using the **set_param_values** command
#. Execute the experiment using the commmand "**run**"
#. Multiple experiments can be created in this fashion. To view the list of created experiments use the command "**experiment**"

