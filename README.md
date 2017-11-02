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
 

## Architecture

The system consists of a web server and several clients depending on the programming language (maybe split up over different repositories).

The web server handles test data and logging of experiments. The client provides a convenience interface for the programming language / machine learning framework at hand. 

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
   

## Server 
   
   - The server manages the following entities: **content**, **datasets**, **experiments**, **users**, **algorithms**/**methods**, **ressources**
   - **content** manages raw content like a set of text documents (assigned to some categories). Preprocessing **content** results in **datasets**
   - **datasets** are used in **experiments**
   - **experiments** are conducted by **users**
   - **experiments** consists of a list of **algorithms**/**methods**
   - **ressources** can be used by **experiments** or **algorithms**
   - **ressources** should be hosted on external services as good as possible.
   - **datasets** maybe taken from other services like [zenodo](http://zenodo.org) in case they are too large
   - **users** should be able to upload **datasets**
   - **users** should authenticate either with [orcid](https://orcid.org), [github](https://github.com) or [gitlab](https://gitlab.com]
   - Every entity is described by additional metadata. In  a first version, simple key/value pairs.
   - **algorithms** and **datasets** should contain links to scientific literatur or the original source of the dataset
   - Future extensions should allow to run **challenges**
   - **challenges** are focused **experiments** on given **datasets** using specific **resources** in order to obtain a **rank** amongst users.
   - **datasets** should be under version control using git. So experiments are conducted on a version of a dataset. Only one version of a dataset is active at a given time.
   
   
   
   
### Functionality 
   
   - User Management (optional, register with Github / ORCID)
   - Providing different views and access points
   - Comparing experiments (algorithms over different datasets, datasets over different algorithms)
   - Allows to generate rankings of users for algorithms or datasets 
   - Every page / url (e.g. algorithms, dataset etc.) should have a discussion section
   - Includes visual exploration tools using [grafana.com](http://grafana.com/)
   
### URL schema

   
   ```
      /datasets/{text|image|video|activtiy|networks}/{id}-{name}/ 
      /experiments/{user}/{supervised|unsupervised|reinforcement|semi-supervised}/
      /experiments/{supervised|unsupervised|reinforcement|semi-supervised}/{user}/
      /methods/algorithms/{algo-name}
      /methods/preprocessing/{name}
   ```
  
API definition should follow the [JSON API V1.0 Schema](http://jsonapi.org/). 
Python libraries with flask are [flask-REST-JSONAPI](http://flask-rest-jsonapi.readthedocs.io/en/latest/) which should be used. 
JSON API allows to name relations between objects in a standardized way, which is important for us.
 
According to good RESTFull interfaces, the following conventions should hold
   
   - content negotation, i.e. html pages for browsers and JSON for API calls
   - lists are ending with an 's' (e.g. algorithms) 
   - User Management with 
   
Note that binary data can be platform/language specific (i.e. serialized numpy arrays) or platform independent. 
The server should be able to conduct conversions and in later versions to provide streams.
   
   
### Examples

- Dataset Repository of the [UCI](http://archive.ics.uci.edu/ml/index.php)
- Datasets and challenges at [Kaggle](http://kaggle.com)
- Stanford Network Datasets [SNAP](https://snap.stanford.edu/data/)
- [OpenML](http://openml.org)
   

## Interesting Software

- [Luigi - Batch Execution Framework](https://github.com/spotify/luigi)
- [SKLearn - Pipelines](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [SKLearn - Datasets](http://scikit-learn.org/stable/datasets/index.html)


## Roadmap

- Start with dataset management following [sklearns interface](http://scikit-learn.org/stable/datasets/index.html), but via a web server. Client language python.
- Try to replicate similar functionality as for the [MLdata Repository - datasets and experimental results](https://mldata.org/), but combined with the sklearn ability.
- Allow upload of testdatasets via the web page and via the command line (python). Data sets need metadata, as for example in the UCI Machine Learning Repository
- Make sure that metadata can be easily extended (use key/value pairs and JSON). 
- Server side, multivariate data sets should be representable via a Pandas Frame. Media data should be either raw or in preprocessed form (vector representation), network data as edge list.
- Data set characteristics should be calculated (not provided). 
