"""
The metadata module contains classes for managing the metadata.

Usually, a repository should define the metadata to be expected. Hence, we need metadata mapper that convert between
repositories.

For example, sklearn defines the "DESCR" attribute for the data set description, while PyPaDRE uses "description"

Maybe we do not need a separate module for it, but integrate it with the repository

"""

# TODO: implement Mapper and Metadata Management
# TODO: implement schemas per classes (e.g. dataset, experiment, loggers etc.)

"""
Metadata keys available to describe datasets
"""
metadata_keys = {
    "description": "Natural language description of the dataset",
    "version": "version of the dataset",
    "source": "url of the original source of the dataset or reference to the original dataset",
    "md5hash": "md5hash of the data set",
    "doi": "digital object identifier of the dataset in its current version",
    "derived_from": "url of the original dataset this dataset is derived from."
}

# TODO: make separate metadata object that can be checked against a schema?

class Mapper(object):
    """
    A mapper is repsonsible for mapping metadata onto each other.
    """
    pass



class Metadata(dict):
    """
    Metadata object which provides an optional mapper
    """
    pass
