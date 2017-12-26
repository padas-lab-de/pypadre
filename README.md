# PyPaDRe - Python Passau Data Science Reproducability Environment

** Machine Learning Experiments on Steroids **

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


## Wiki 

The project is documented and steered via the [wiki](https://gitlab.dimis.fim.uni-passau.de/RP-17-PaDReP/PyPaDRe/wikis/home)

