# PyPaDRe - Python Passau Data Science Reproducability Environment
PaDRe is an open source tool for managing machine learning projects and experiments, tracking the life cycle of each experiment, adding semantic 
meaning to the experiments and keeping track of different results and metrics.
Client System for the [PaDRE Servant Server](https://gitlab.dimis.fim.uni-passau.de/RP-17-PaDReP/PaDRE-Servant/wikis/home). It should provide the following functions

- Manage data sets and splits of data sets
- Client Side code to conduct machine learning projects and experiments including
  - creating and managing projects for grouping of experiments
  - creating and managing experiments within a project
  - fetching and splitting data sets
  - logging training tasks
  - logging test tasks
  - providing results
  - Hyperparameter Optimization using different strategies (Grid Search, Evolutionary Algorithms[not yet implementd])
  - inspecting results of individual experiments 
  - Describing experiments using (semantic) metadata
  - Linking experiments to git code (e.g. automatically push a git repository when running experiment) including client source code
  - Caching data sets client side
  - Managing external data for experiments (e.g. external models, embeddings, additional data)

From the clients perspective, PaDRE could be also understood as package manager for data sets and experiments.

## Example Usages 

### Examples are under `tests`

- `example.py` shows how to use pypadre with an example sklearn tool
- `testexperiment.py` shows how to extract parameters from pypadre

### Examples using the command line client

The command line client can be found under `padre/app/padre_cli`. 
Note that a start from the command line requires the python path set 
to the local directory, i.e. `PYTHONPATH="./"`.

Show usage and help:
```
PYTHONPATH="./" python3 padre/app/padre_cli.py
```

List datasets (requires running padre server)
```
PYTHONPATH="./" python3 padre/app/padre_cli.py datasets
```

Show single datatset properties
```
PYTHONPATH="./" python3 padre/app/padre_cli.py dataset 3
```

## Wiki 

The project is documented and steered via the [wiki](https://gitlab.dimis.fim.uni-passau.de/RP-17-PaDReP/PyPaDRe/wikis/home)

# How PaDRe works
PaDRe is made for reproducibility and tracking of experiments over their lifetime. The backbone of PaDRe is git. Every experiment is added to the git and the user based source code is also git versioned.
It can be done either automatically or by the user provided and maintained git repository. Running an experiment requires the following
1. Project to which the experiment belongs
2. Name of the experiment
3. Description of the experiment
4. Pipeline specification of the experiment which would be a function
5. Dataset for the experiment
6. Splitting strategy for the experiment
7. A reference to the source code of the experiment

Using these parameters, an experiment object can be created. The user can then execute the experiment. While executing the experiment, the user can customize certain functionlities such as
1. Providing hyperparameters to the components in the pipeline
2. Dump intermediate results and/or metrics 
3. Specify what metrics are to be used
4. Specify whether the user needs the results written to disk or not

# Sharing of results
The padre experiment can be shared simply by sharing the git repository or by sharing the folder which contains the experiment.

# Installation instruction
Installation of Padre is simple and can be installed simply by running pip install pypadre

