===============
PyPaDRE Concept
===============

Goals
-----

PaDRE (Open Science Machine Learning Reproduciblity Environment) provides a web-based platform to
collect, compare and analys machine learning experiments in a systematic and open way. It builds on
Open Science principles and strives for improving research and education processes in disciplines
related to Machine Learning and Data Science.

PyPaDRE is the Python-centric environment to create and execute Machine Learning and Data Science experiments.
It aims to be a **minimal-invasive environment** for evaluating Machine Learning and Data Science experiments in a
**systematic and structured** way.
While PyPaDRE can be used as standalone, local solution for running and comparing experiments, all results can - and in the spirit of Open Science should - be made available via the PaDRE platform.

Architecture
------------

The following figure shows the overall architecture.

.. image:: images/current-architecture.png

PyPaDRE (in blue) provides means to

- Manage local and remote data sets
- Create, run and compare experiments
- Automatically identify hyperparameters and components of workflows from SKLearn and PyTorch
- Provide an easy to use Python and CLI interface
- Syncronize all resources with the PaDRE Server

while being minimal invasive.

Researchers should use their favorite tools and environments in order to conduct there research while PyPaDRE takes
care of managing resources and experiments in the background.

Experiments and Pipelines
*************************

In order to do so, PyPaDRE defines four core concepts: pipelines, experiments, runs and splits.

- **Pipelines/Workflows** are the actual machine learning workflows which can be trained and tested. Currently we support SKLearn and PyTorch based pipelines, where in its simplest case a pipeline only needs to implement an infer and a fit fucntion
- **Experiments** define the experimental setup consisting of a pipeline (i.e. the executable code), a dataset, and hyperparameters controlling the experiment as well as parameters controlling the experimental setup (e.g. splitting strategy)
- **Runs** are single instances of experiments with a specific set of hyperparameter-value
- **Splits** are the actual executions of a run, i.e. the execution of the workflow, over a dataset split.

In general, users do not have to care about Experiments, Runs and Splits.
They need to implement their pipeline or machine learning component that provides a `fit` and an `infer` function.
Additionally, a experiment configuration needs to be provided including a dataset.
When executing the experiment, PyPaDRE stores results and intermediate steps locally and adds it to the database of experiments.
Afterwards, it provides easy means to evaluate the experiments and compare them, as outlined below.

For more details please refer to Setting up Experiments :ref:`setup_experiments`.

Experiment Evaluation
---------------------




Storage
-------


Metasearch and Automated Machine Learning
-----------------------------------------

