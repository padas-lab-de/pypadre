# PaDaS Reproducability Environment

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
  


## Architecture

The system consists of a web server and several clients depending on the programming language (maybe split up over different repositories).

The web server handles test data and logging of experiments. The client provides a convienience interface for the programming language / machine learning framework at hand. 

The overhead should not be recognized by the user, hence caching is needed. We assume honest participants, i.e. it is not a challgenge setting where information on the test data needs to be hidden from the client. Maybe in the future this will be the case.


## Usage Examples

Note that the usage examples are design examples. Changes during development (according to best practices in software architecture) are possible.

```
  # list data sets with a search string
  l = padrev.datasets.list_datasets('some search string')
  # fetch one data set. might be only a stub to the web server depending on size of data set
  d = padrev.datasets.fetch_dataset('some id of the dataset', cached=true)
  # conduct a split according to some splitting strategy plus parameters (e.g. bootstrapping 60/20)
  splits = d.split(SplittingStragey.xy, param1=a, param2=b)
  # register new experiment
  ex = padrev.experiments.new_experiment(splits, description_of_machine_learning_experiment)
  # start trainig. we will have a logger to provide training information
  logger = ex.new_logger()
  # log the experiment setup
  logger.log(LoggerEvent.SETUP, start=true)
  my_ml_algo = new_machine_learning_experiment()
  logger.log(LoggerEvent.SETUP, end=true)
  # log the training
  logger.log(LoggerEvent.TRAINING, strat=true)
  while not my_ml_algo.finished:
	my_ml_algo.learnstep()
	logger.log(LoggerEvent.TRAINSTEP, parameters_from_training_step_as_keyval)
  logger.log(LoggerEvent.TRAINING, end=true)
  # log the evaluation
  logger.log(LoggerEvent.TESTING, strat=true)
  logger.log(LoggerEvent.TEST_RESULTS, my_ml_algo.test(d))
  logger.log(LoggerEvent.TESTING, end=true)
  # print results
  ex.get_results()
  # compare to other experiments
  ex.get_rank(ranking_criterion)
  # get similar experiments
  ex.get_similar_experiments()
  # guess the next hyperparameters
  ex.guess_new_hyperparameters()
  
```

The `description_of_machine_learning_experiment` will be a simple key value store in the beginning. However, aftewards the code of a scikit learn (or other framework) should be parsed automatically and parameters plus strcture should be stored AUTOMATICALLy. Moreover, we distinguish between different kinds of experiment, yielding different reproducability evidence criterions:
   - User provided experiment information - this is the key value list
   - Parsed experiment information - this is parsing the parameter values and setup within a known framework
   
The same holds true for parameters provided during traiing /testing
   
In addition, we distinguish between different repeatability criterions 
   - No archived source code
   - Provided git snapshot
   - provided virtual machine
