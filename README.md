# PyPaDRe - Python Passau Data Science Reproducability Environment

Client System for the [PaDRE Servant Server](https://gitlab.dimis.fim.uni-passau.de/RP-17-PaDReP/PaDRE-Servant/wikis/home). It should provide the following functions

- Manage data sets and splits of data sets
- Client Side code to conduct experiments including
  - fetching and splitting data sets
  - logging training tasks
  - logging test tasks
  - providing results
  - Hyperparameter Optimization using different strategies (Grid Search, Evolutionary Algorithms
  - inspecting results of individuell experiments and compare it to other related experiments
  - Describing experiments using (semantic) metadata
  - Linking experiments to git code (e.g. automatically push a git repository when running experiment)
  - Caching data sets client side
  - Managing external data for experiments (e.g. external models, embeddings, additional data)

From the clients perspective, PaDREV could be also understood as package manager for data sets and experiments.

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

