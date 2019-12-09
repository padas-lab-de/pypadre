=================
Backend
=================

Backend
***************************
Backend is used to communicate experiments, runs and splits, metrics and results information.
Following are some important functions implemented in this backend. Backends handle flow of information from the
pypadre framework to the outside world. Backends are used for Creating, Updating, Reading, and Deleting the
different PaDRe objects. Backends can be a File Backend, HTTP Backend, Git Backend etc

Http Backend
------------
Http backend is used to exchange information between client and server. Base class for Http backend is used
 to setup server information, authenticate user, check if client is online, retrieve token and do generic
 http calls. It was implemented as a Proof Of Concept and is not included currently in the package.

Backend Functionalities
***************************

Backends link to each part of the experiment such as projects, experiments, executions, runs and computations.
All the backends provide get functionalities to retrieve the individual modules.

For example the functionality for experiments described below:

1. put_experiment: It receives an experiment. If it is an instance of the experiment class,
the experiment class is first stored to disk using the file backend. If it is a string with the name,
the experiment is loaded from the file backend and stored to the serer. The experiment in the file backend
should receive an url in its metadata file to indicate that it has been uploaded to the server.

2. delete_experiment: Deletes the experiment given by ex (either as class or as string) from the local
file cache or the remote server. It also receives mode whose possible values can be all, remote or local

3. get_experiment: Downloads the experiment given by ex, where ex is a string with the id or url of the
experiment. The experiment is downloaded from the server and stored in the local file store if the
server version is newer than the local version or no local version exists. The function returns an experiment
class, which is loaded from file.

4. create

Communication to the Backend
*********************************
Communication with the backends are done via the service layer. The service of each module such as the Dataset,
Experiment, Project, Execution call their corresponding services via events. The services then access the backends
and execute the functions for listing objects, loading objects, etc.
