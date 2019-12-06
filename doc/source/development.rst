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

The PaDRe app is the main app and interfaces with all the other apps. These apps provide functionalities regarding their
modules to the user.

PyPadre CLI
+++++++++++

The PyPadre CLI is the command line interface to the PyPadre App. The app can be invoked from the command line
by typing "pypadre". Projects, experiments, executions, runs, and computations can be accessed from the CLI.

The CLI supports the following functionalities for all modules

- list: lists all available modules of similar type. For example: project list, it lists all the projects
- get: loads the specified module

The following functionalities are supported only for Projects, Experiments, Executions, and Run modules
- select: sets an object to be the current active object

In addition, each module supports some custom functions as following
Projects support the following functionalities via the CLI
- create: Creates a new project

For experiments, these are the unique functionalities
- initialize: Creates an experiment with default parameters and opens a text editor for the user to configure the rest
- execute: Executes an experiment

Execution module have the capability to compare two different executions
- compare: To compare two different executions




Unit Testing and Examples
-------------------------

- Unit tests for each module are placed within those modules.
- Tests must use the prefix `tests_`
- Experiments created during tests will be removed when the tests are completed. It is the same for the configuration files too.
- Unit tests should be written as often as possible (i know, its a pain) and as a proof of concept.


