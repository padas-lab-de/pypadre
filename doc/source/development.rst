PyPaDRE Development
===================

Module Architecture
*******************

PyPadre App
+++++++++++

The PyPadre App (`padre.app.padre_app`) is the Python API for interacting with pypadre. The following classes are important

TODO: Integrate modul documentation here.

PyPadre CLI
+++++++++++

The PyPadre CLI is the command line interface to the PyPadre App.

TODO: Integrate modul documentation here.


Unit Testing and Examples
-------------------------

- Unit test and examples should be placed under tests
- Examples must use the prefix `example_`
- Examples created during tests must use the prefix `example` in their name
- Tests must use the prefix `tests_`
- Experiments created during tests must use the prefix `test` in their name
    - note that all experiments with prefix `test` will be removed during unit testing!!!
        --> TODO: we should make a separate testing config for running the tests
- Unit tests should be written as often as possible (i know, its a pain) and as a proof of concept.


