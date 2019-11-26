=================
Design Principles
=================

Identity Management
-------------------

Every padre object that is stored/retrieved (e.g. dataset, experiment) will get a unique id.
The id will be assigned by the backend. If an id is `None`, it has not been persisted by the backend.

Projects
----------
A project is simply a collection of experiments. Each project has a unique name associated with it. The project name is
the name of the directory in the file structure.


Experiment Naming and Uniqueness of Experiments
-----------------------------------------------

Every experiment is associated with a project. The experiment name has to be unique within a project.
The experiment directory is created within each project directory, and the name of the directory is the experiment name