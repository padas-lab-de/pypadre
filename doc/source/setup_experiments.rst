Setting up Experiments
======================


Single Pipeline Experiments
---------------------------

Single pipeline experiments can be created in different ways:

1. Through class instantiation (see `examples/09_metrics/09_metrics.py`)

.. literalinclude:: ../../examples/09_metrics/09_metrics.py

2. Through decorators

.. literalinclude:: ../../tests/experiments/example_single_decorator.py


Hyperparameter Optimization
---------------------------
1. Through parameters passed to the experiment execute function. The parameters are passed as a dictionary with the
key as the component name and an inner dictionary. The inner dictionary contains the parameter name as the key and
an array of values that are to be used for hyperparameter optimization

2. Through decorators

.. literalinclude:: ../../tests/experiments/example_decorator_hyperparameters.py

Multi-pipline, multi-data Experiments
-------------------------------------

Currently, not suppoted