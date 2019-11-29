=================
Backend
=================

.. automodule:: pypadre

Http Backend
------------
Http backend is used to exchange information between client and server. Base class for Http backend is used
 to setup server information, authenticate user, check if client is online, retrieve token and do generic
 http calls. It was implemented as a Proof Of Concept and is not included currently in the package.

1. Backend
***************************
Backend is used to communicate experiments, runs and splits, metrics and results information.
Following are some important functions implemented in this backend. Backends handle flow of information from the
pypadre framework to the outside world. Backends are used for Creating, Updating, Reading, and Deleting the
different PaDRe objects. Backends can be a File Backend, HTTP Backend, Git Backend etc

#. .. autofunction:: pypadre.pod.backend.file.PadreFileBackend.log

2. Backend Functionalities
***************************

Functionality for experiments described below:

#. put_experiment: It receives an experiment. If it is an instance of the experiment class,
the experiment class is first stored to disk using the file backend. If it is a string with the name,
the experiment is loaded from the file backend and stored to the serer. The experiment in the file backend
should receive an url in its metadata file to indicate that it has been uploaded to the server.
#. delete_experiment: Deletes the experiment given by ex (either as class or as string) from the local
file cache or the remote server. It also receives mode whose possible values can be all, remote or local
#. get_experiment: Downloads the experiment given by ex, where ex is a string with the id or url of the
experiment. The experiment is downloaded from the server and stored in the local file store if the
server version is newer than the local version or no local version exists. The function returns an experiment
class, which is loaded from file.

3. Communication to the Backend
*********************************
Communication with the backends are done via the service layer. The service of each module such as the Dataset,
Experiment, Project, Execution call their corresponding services via events. The services then access the backends
and execute the functions for listing objects, loading objects, etc.
