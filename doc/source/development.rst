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

TODO: Integrate modul documentation here.


Unit Testing and Examples
-------------------------

- Unit tests for each module are placed within those modules.
- Examples must use the prefix `example_`
- Examples created during tests must use the prefix `example` in their name
- Tests must use the prefix `tests_`
- Experiments created during tests must use the prefix `test` in their name
    - note that all experiments with prefix `test` will be removed during unit testing!!!
        --> TODO: we should make a separate testing config for running the tests
- Unit tests should be written as often as possible (i know, its a pain) and as a proof of concept.


