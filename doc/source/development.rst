PyPaDRE Development
===================

Module Architecture
*******************

PyPadre App
+++++++++++

PyPaDRe provides a CLI to interact with PyPaDRe and this is done via apps. There are different apps within PyPaDRe such
as the Project App, Experiment App and so on. Apps provide a method to interact with components of PyPaDRe. The apps
support different functions such as listing, searching, and deleting. There are different apps such as
- project app
- experiment app
- execution app
- run app
- computation app
- dataset app
- metric app
- computation app
- config app

The PaDRe app is the main app and interfaces with all the other apps. These apps provide functionalities regarding their
modules to the user.

PyPadre CLI
+++++++++++

The PyPadre CLI is the command line interface to the PyPadre App. The app can be invoked from the command line
by typing "pypadre". Projects, experiments, executions, runs, and computations can be accessed from the CLI.

- list: Supported by all the modules. It simply lists all the modules at which it is called. For example: if the list
is used in conjunction with experiment("experiment list"), all the experiments are listed.

- get: Supported by all the modules. It loads the object into the memory.

- active: Supported by all modules except computation. This commands sets an active module. For example, setting an
active project so that all the following commands can take the active project as the one set by the user.

- create: Supported by project alone. This command is used to create new projects.

- initialize: Supported by the experiment module alone. This is done to initialize the experiment with the required
parameters.

- execute: Supported by the experiment module alone. This command is used to execute an experiment once it has been
initialized.

- load: loads a dataset from the given source

- sync: this command is used to synchronize the datasets among the different backends

- compare: Supported by the experiment and execution modules only. For experiments, this is used to compare the metadata
and the pipelines of two experiments. For executions, this is used to compare the source code which is referencing each execution.

- compare_metrics: supported by the metric app to compare the metrics of two different runs

- get_available_estimators: lists all the available estimators for the specified experiment

- list_experiments: lists all possible experiments that can be compared

- reevaluate_metrics: reevaluates all the metrics from the results of the experiment

- set: lets the user set a key value pair in the current configuration


Unit Testing and Examples
-------------------------

- Unit tests for each module are placed within those modules.
- Tests must use the prefix `tests_`
- Experiments created during tests will be removed when the tests are completed. It is the same for the configuration files too.
- Unit tests should be written as often as possible (i know, its a pain) and as a proof of concept.


