Setting up Experiments
======================


Single Pipeline Experiments
---------------------------

Single pipeline experiments can be created in different ways:

1. Through class instantiation (see `test/example.py`)

.. literalinclude:: ../../tests/example.py

2. Through decorators

.. literalinclude:: ../../tests/experiments/example_single_decorator.py


Hyperparameter Optimization
---------------------------

2. Through decorators

.. literalinclude:: ../../tests/experiments/example_decorator_hyperparameters.py

Multi-pipline, multi-data Experiments
-------------------------------------

1. Through using the `ExperimentCreator` class

.. literalinclude:: ../../tests/example3.py


2. Through decorators

Decorators in a single file:

.. literalinclude:: ../../tests/experiments/example_multi_decorator.py

Decorators in a path

.. literalinclude:: ../../tests/experiments/decorator_import/ex1.py

.. literalinclude:: ../../tests/experiments/decorator_import/ex2.py

.. literalinclude:: ../../tests/experiments/decorator_import/example_multi_decorator_by_import.py