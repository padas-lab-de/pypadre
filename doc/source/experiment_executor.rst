*********************
Experiment Executor
*********************
.. automodule:: experimentexecutor


.. autoclass:: experimentexecutor.ExperimentExecutor
   :members:

Steps for execution of experiments
------------------------------------------------------------

#. Create the experiments using Experiment Creator
#. Obtain the experiments list using the function experiment_creator.createExperimentList()
#. Experiments the experiments using the function experiments_executor.execute(local_run=True/False, threads=num_threads)