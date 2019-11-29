Experiment Storage and Online Synchronization
=============================================


Local Storage Structure
-----------------------
Git is the backbone of PaDRe. Every experiment, project and dataset is a git repository. Datasets are added with Git LFS.
Each experiment is a submodule of the project. For every experiment, a source code version is associated with the experiment.
Every time the source code changes, a new execution is created. Every execution of the experiment is a run. A run is
defined as the execution of the pipeline with a dataset. Runs are contained in executions so that every particular run
is associated with a specific version of the source code. Within each run, there are computations which are nodes in the pipeline.
The metadata and if needed output of the nodes are stored within these computations.

Server Synchronisation
----------------------

The server supports uploading and downloading of experiments but was developed as a Proof Of Concept. It is not
addressed in this documentation.

Versioning
**********
Versioning of PaDRe is done via Git. the user can use the Git to track the lifecycle of the experiment.

Live Logging
************
Logging of experiments is done via blinker event handling.
