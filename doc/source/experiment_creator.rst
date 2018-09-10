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

