=================
Backend
=================

.. automodule:: backend.dual_backend


.. autoclass:: backend.dual_backend.DualBackend
   :members:

Http Backend
------------
Http backend is used to exchange information between client and server. Its further divided into HttpBackendDatasets
and HttpBackendExperiments.
Base class for Http backend is used to setup server information, authenticate user, check if client is online,
retrieve token and do generic http calls.

Important functions for base class :class:`pypadre.core.backend.http.PadreHTTPClient` are following

#. .. autofunction:: pypadre.core.backend.http.PadreHTTPClient.authenticate

#. .. autofunction:: pypadre.core.backend.http.PadreHTTPClient.get_access_token

Http backend is further divided into Http Experiments and Datasets backends

1. Http Experiments Backend
***************************
Http experiments backend is used to communicate experiments, runs and splits, metrics and results information between
client and server. Following are some important functions implemented in this backend.

#. .. autofunction:: pypadre.core.backend.http_experiments.HttpBackendExperiments.put_experiment
#. .. autofunction:: pypadre.core.backend.http_experiments.HttpBackendExperiments.get_experiment
#. .. autofunction:: pypadre.core.backend.http_experiments.HttpBackendExperiments.list_experiments
#. .. autofunction:: pypadre.core.backend.http_experiments.HttpBackendExperiments.put_run
#. .. autofunction:: pypadre.core.backend.http_experiments.HttpBackendExperiments.get_run
#. .. autofunction:: pypadre.core.backend.http_experiments.HttpBackendExperiments.put_split
#. .. autofunction:: pypadre.core.backend.http_experiments.HttpBackendExperiments.get_split

2. Http Datasets Backend
***************************

Dual Backend
-----------
For realising transparent file and http backend, dual backend is implemented that
combines both file and http backend. Transparent thereby means, that the file backend
is used as cache or that we can sync file cache easily with the http server.

All functions for file and http backend will be implemented here and then from dual backend function calls
are delegated to http or file backend.
Functionality for experiments described below:

#. put_experiment: It receives an experiment. If it is an instance of the experiment class, the experiment class is first stored to disk using the file backend. If it is a string with the name, the experiment is loaded from the file backend and stored to the serer. The experiment in the file backend should receive an url in its metadata file to indicate that it has been uploaded to the server.
#. delete_experiment: Deletes the experiment given by ex (either as class or as string) from the local file cache or the remote server. It also receives mode whose possible values can be all, remote or local
#. get_experiment: Downloads the experiment given by ex, where ex is a string with the id or url of the experiment. The experiment is downloaded from the server and stored in the local file store if the server version is newer than the local version or no local version exists. The function returns an experiment class, which is loaded from file.
