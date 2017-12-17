# PyPaDRe - Python Passau Data Science Reproducability Environment

Client/Server System for the following functions

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
  
- Server side code for managing data sets and managing + analysing experiments:
  - compare experiments
  - provide test data sets
  - fetch and push testdatasets to zenodo
  - freeze results and push to zenodo for giving it a DOI
  - provide rankings between algorithms / data sets
  - provide hyperparameter suggestions
  - conduct grid search

From the clients perspective, PaDREV could be also understood as package manager for data sets and experiments.


## Wiki 

The project is documented and steered via the [wiki](https://gitlab.dimis.fim.uni-passau.de/RP-17-PaDReP/PyPaDRe/wikis/home)



   
 

