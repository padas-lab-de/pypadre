Setting up Experiments
======================

Single Pipeline Experiments
---------------------------

Single pipeline experiments can be created in different ways:

1. Through class instantiation (see `examples/09_metrics/09_metrics.py`)

.. code-block:: python

    from pypadre.core.model.project import Project
    from pypadre.core.model.experiment import Experiment
    from pypadre.binding.metrics import sklearn_metrics
    print(sklearn_metrics)

    self.app.datasets.load_defaults()
    project = Project(name='Sample Project',
                      description=Example Project',
                      creator=Function(fn=self.test_full_stack))

    _id = '_iris_dataset'
    dataset = self.app.datasets.list({'name': _id})

    experiment = Experiment(name='Sample Experiment', description='Example Experiment',
                            dataset=dataset.pop(), project=project,
                            pipeline=SKLearnPipeline(pipeline_fn=create_test_pipeline_multiple_estimators),
                            reference=self.test_full_stack)
    experiment.execute()

2. Through decorators

.. include:: ../../examples/09_metrics/09_metrics.py


Hyperparameter Optimization
---------------------------
1. Through parameters passed to the experiment execute function. The parameters are passed as a dictionary with the
key as the component name and an inner dictionary. The inner dictionary contains the parameter name as the key and
an array of values that are to be used for hyperparameter optimization

2. Through decorators

.. include:: ../../examples/07_seeding/07_seeding.py

Multi-pipline, multi-data Experiments
-------------------------------------

Currently, not suppoted