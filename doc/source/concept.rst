===============
PyPaDRE Concept
===============

Goals
-----

PaDRE (oPen mAchine learning and Data science Reproduciblity Environment [for education and research]) provides a web-based platform to
collect, compare and analys machine learning experiments in a systematic and open way. It builds on
Open Science principles and strives for improving research and education processes in disciplines
related to Machine Learning and Data Science.

PyPaDRE is the Python-centric environment to create and execute Machine Learning and Data Science experiments.
It aims to be a **minimal-invasive environment** for evaluating Machine Learning and Data Science experiments in a
**systematic and structured** way.
While PyPaDRE can be used as standalone, local solution for running and comparing experiments, all results can - and in the spirit of Open Science should - be made available via the PaDRE platform.

Use Cases
---------

Rapid Development Support
*************************

Padre's first use case is for rapid development by reducing scaffolding scientific experiments.
Users should be given

- easy access to datasets
- convenience functions for setting up experiments
- running the experiments
- comparing and visualising results
- packaging and deploying the experiment

Every experiment run should be logged in order to keep the experiments traceable / comparable.

Hyperparameter Optimization
***************************

After setting up an initial prototype, the next step is to do some initial hyperparameter tests.
Padre should support this via a few lines of code and make it easy to analyse the optimized hyperparameters / runs.

Large Scale Experimentation
***************************

Large scale experimentation aims at a later stage in the research cycle.
After developing an initial prototype there is often the need to setup larger experiments and compare the different
runs with each other, show error bars, test for statistical significance etc.
Large scale experimentation aims to define an experiment and hyperparameters to be tested and to deploy the
experiment runs over a larger cluster. Results could be checked online and compared online using web-based visualisation.
Results can be converted in different formats (e.g. latex, markedown).


Visual Analysis and Collaborative Exploration
*********************************************

Experimental results should be visualised / analysed in a interactive frontend and potentially discussed with others.

Interactive Publishing and Versioned Result Publishing
******************************************************

Experimental results should be accessible online and embedded in online publications (e.g. Blogs).
When new versions of experiments are available, a new version of the online publication should appear.
The versions need to be trackable.

Research Management and Collaboration
*************************************

Managing a smaller (PhD) or larger (funded project) research project should be fully supported by padre. This includes

- source code management
- experiment management and evaluation
- formulation of research questions
- collaboration on experiments, interpretation and writing
- issue tracking and discussion

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

- **Pipelines/Workflows** are the actual machine learning workflows which can be trained and tested. The pipeline is composed of different components which are basically data processes that work on the input data. They can be preprocessing algorithms, splitting strategies, classification or regression algorithms. PaDRe currently supports custom code, SKLearn pipelines and the PyTorch framework within SKLearn pipelines.
- **Experiments** define the experimental setup consisting of a pipeline (i.e. the executable code), a dataset, and hyperparameters controlling the experiment as well as parameters controlling the experimental setup (e.g. splitting strategy)
- **Executions are running of the workflows which are associated with a particular version of the source code. If the source code is changed, it results in a new execution.
- **Runs** are single executions of the pipelines. Each time the experiment is executed a new run is generated in the corresponding execution directory.
- **Computations** are the actual executions of a run, i.e. the execution of the workflow, over a dataset split.

In general, users do not have to care about Experiments, Executions, Runs and Components.
They need to implement their pipeline or machine learning component and wrap it with the wrapper PaDRe provides.
Additionally, a experiment configuration needs to be provided including a dataset.
When executing the experiment, PyPaDRE stores results and intermediate steps locally and adds it to the database of experiments.
Afterwards, it provides easy means to evaluate the experiments and compare them, as outlined below.

For more details please refer to Setting up Experiments :ref:`setup_experiments`.

Components and Hyperparameters
******************************

Hyperparameters are distinguished between

- model parameters: parameters, that influence the model
- optimizer parameters: parameters, that influence the optimizer
- other parameters: parameters, not fitting into the above classes

Hyperparameters can be specified by the individual components directly in code (recommended for smaller experiments) or
via a mappings file, which is a `json` file that links metadata to the implementation in a library.
The mapping file also provides an extensible mechanism to add new frameworks easily.
Via an inspector pattern padre can extract from relevant parameters and components from an instantiated pipeline.

Components follow some implementation details and provide `fit`, `infer` and configuration commands.

TODO: describe more details.



Experiment Evaluation
---------------------

Experiments should store the following results

- **Raw Results** currently consisting of regression targets, classification scores (thresholded), classification
probabilities, transformations (e.g. embeddings).Results are stored per instance (per split).

- **Aggregated Results** are calculated from raw results. This includes precision, recall, f1 etc.

Evaluation should include standard measures and statistics, but also instance based analysis.

Evaluation results will be released on static pages (and thus archived via zenodo).


Research Assets Management
-------------------------

Beyond experiment support, the platform should also help to manage research assets, like papers, software, projects
research questions etc. Currently, these artifacts can be managed via adding them to the source code folder and let it be Git managed.



Metasearch and Automated Machine Learning
-----------------------------------------



PyPadre App and CLI
-------------------

One core criterion of PyPaDRE is its ease of use and hence we support a class interface,
a high-level app interface and a command line interface.

Python Class Interface
**********************

First, when knowing the details of all packages PyPaDRE can be used in code.
This is either done by creating an :class:`padre.experiment.Experiment` or
through using decorators (currently under development). However, in this case
the user is responsible for using the correct backends to persist results to.

.. code-block:: python

    from padre.ds_import import load_sklearn_toys
    from padre.core import Experiment
    ds = [i for i in load_sklearn_toys()]
    ex = Experiment(name="Test Experiment SVM",
                    description="Testing Support Vector Machines via SKLearn Pipeline\n"
                                    "- no persisting via a backend\n"
                                    "- manual data set loading\n"
                                    "- default parameters",
                    dataset=ds[2],
                    workflow=Pipeline([('clf', SVC(probability=True))]))
    ex.run()

Please note, that this is not the standard case and proper evaluation classes are currently under development.

Python App Interface
********************

As a second interface, PyPaDRE support a high-level app. This high-level app integrates experiments, file backends, configuration
files and http server interface in a high level, easy to use interface.

.. code-block:: python

    from padre.ds_import import load_sklearn_toys
    from padre.app import pypadre
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    ex = pypadre.experiments.run(name="Test Experiment SVM",
                                     description="Testing Support Vector Machines via SKLearn Pipeline",
                                     dataset=ds,
                                     workflow=Pipeline([('clf', SVC(probability=True))]))
    print("========Available experiments=========")
    for idx, ex in enumerate(pypadre.experiments.list_experiments()):
        print("%d: %s" % (idx, str(ex)))
        for idx2, run in enumerate(pypadre.experiments.list_runs(ex)):
            print("\tRun: %s" % str(run))


TODO: add more details here.

Python CLI Interface
********************

The third interface is a command line interface for using Python via a command line. Please note that not all
functions are available.