Getting Started - First Steps with PyPadre
==========================================

Installation
------------

Server
******

In case you aim to develop locally, you should install the Padre Servant. You need
- Postgres
  - add padre database
  - add user padre/padre
- Java / Maven
- mvn install
- Lombok needs to be installed (intellij only)


Configuration
-------------

```
from padre.app.padre_app import pypadre  # import the pypadre API. This loads the standard configuration
```
