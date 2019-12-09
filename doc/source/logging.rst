Logging in PaDRe
=================

Padre has the ability to log to multiple different locations such as file, http, database etc at the same time provided
there is a backend define for it. Currently, the logging is only done to the file backend, which is then taken up by the
Git backend when the experiment is committed. The file backend writes incoming messages to the local disk at a central
location. The user can easily extend the logging functionality to incorporate their own code for user specific handling
of events and logging. A basic framework is provided and the user can add new backends and handle each event based on
their custom backends. This is useful when the user has experiments running on a cluster and needs to write the
experiments to a slack service or use python code to periodically notify the user about the experiment updates.

The incoming logging messages can be of three levels of logging
- INFO: These are the normal messages which are logged, like the start of an experiment, the beginning of different
stages and its completion along with the respective time stamps
- WARN: These are warning information which could lead to the framework crashing if the user does not heed to the
messages. For example if the dataset has no target but is loaded, a warning message appears saying that this particular
dataset should not be used for supervised learning.
- ERROR: These log messages are of critical importance and will disrupt the proper functioning of the framework and
cause it to crash. A condition is passed along with the log error event and if the condition is satisfied all the
backends are written with the error information and the Padre framework stops its execution.

Currently, the logging is done via firing of events via the blinker package in Python. The fired event is then handled
at the service which then passes it to the backend. The backend then logs the event as per defined. For the FileBackend
the log format of the message is current time + LOG\_LEVEL(INFO, WARN or ERROR) followed by the message.