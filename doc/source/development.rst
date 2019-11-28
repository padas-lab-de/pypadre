PyPaDRE Development
===================

Module Architecture
*******************

PyPadre App
+++++++++++

PyPaDRe provides a CLI to interact with PyPaDRe and this is done via apps. There are different apps within PyPaDRe such
as the Project App, Experiment App and so on. Apps provide a method to interact with components of PyPaDRe. The apps
support different functions such as listing, searching, and deleting.

PyPadre CLI
+++++++++++

The PyPadre CLI is the command line interface to the PyPadre App.

# Add the different modules


Unit Testing and Examples
-------------------------

- Unit tests for each module are placed within those modules.
- Tests must use the prefix `tests_`
- Experiments created during tests will be removed when the tests are completed. It is the same for the configuration files too.
- Unit tests should be written as often as possible (i know, its a pain) and as a proof of concept.


