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
* intialize_estimator_parameter_implementation: This function reads all the parameters available to the function and also identifies the implementation of the function for dynamic loading
* initialize_dataset_names: This function reads all the names of all the available datasets

Get Functions
-------------
* get_dataset_names: This function returns the names of all the available datasets in the system.
* get_estimators: This function return the names of all the available estimators
* get_estimator_params(estimator_name): This function returns all the parameters belonging to an estimator.
* get_estimator_object(estimator_name): Returns an object of the estimator as specified in the mappings.json implementation
* get_local_dataset(name): Returns the instance of the dataset

Set Functions
-------------
* set_param_values(experiment_name, param_dict): Used to set the parameters for a particular experiment.
The param_dict contains the name of the estimator and the parameters to be set for that estimator. The parameters
for the experiment are set after validation.
* set_parameters(estimator_object, estimator_name, param_dict): This function sets the parameters for the estimator
object after validating that the estimator has such an available parameter.

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

