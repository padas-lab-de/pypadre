*********************************
Experiment Creator
*********************************

.. automodule:: ExperimentCreator

.. autoclass:: ExperimentCreator.ExperimentCreator
   :members:

Experiment Creation from the Command Line Interface
------------------------------------------------------------

#. Display all the available estimators using "**components**" command
#. Display all the available datasets using "**dataset**" command
#. Create an experiment using the "**create_experiment**" command with parameters a unique experiment_name, a description of the experiment, the name of the dataset to be used(names obtained in the above step) and the list of estimators to beused for the workflow(estimators are separated by a comma)
#. Set experiment parameters using the **set_param_values** command
#. Execute the experiment using the commmand "**run**"
#. Multiple experiments can be created in this fashion. To view the list of created experiments use the command "**experiment**"

Experiment execution using JSON configuration files
--------------------------------------------------------------

Multiple experiments can be created using the JSON configuration files and run from the CLI
Run the CLI command load_config_file --file_path.json
The experiments present within the file are loaded and executed in a sequential manner
All functionalities supported by the CLI is possible through the experimental configuration files.

