import os

from git import Repo


def git_hash(self, path: str):
    # The current path given might be of a file which is within the working tree.
    # We need to search parent directories until we find the root git directory

    # If the passed path is a file, then get its containing directory
    if os.path.isfile(path=path):
        dir_path = os.path.dirname(path)

    elif os.path.isdir(path=path):
        dir_path = path

    else:
        # This shouldn't occur
        raise ValueError("Path of the repository is invalid " + path)

    repo = open_existing_repo(dir_path, search_parents=True)

    if repo is not None:
        return repo.head.object.hexsha

    # If no repository was found return none
    return None


def pull(repo, name=None):
    origin = repo.remote(name=name)
    origin.pull()


def repo_exists(dir_path):
    if os.path.exists(os.path.join(dir_path, '.git')):
        return True
    else:
        return False


def commit(repo, message):
    """
    Commit a repository
    :param repo: Repo object
    :param message: Message when committing
    :return:
    """
    repo.git.commit(message=message)


def create_sub_module(repo, sub_repo_name, path_to_sub_repo, url, branch='master'):
    """
    Creating a submodule
    :param repo: Repo object where the submodule has to be created
    :param sub_repo_name: Name for the sub module
    :param path_to_sub_repo: Path to the submodule
    :param url: URL of the remote repo
    :param branch:
    :return:
    """
    repo.create_submodule(sub_repo_name, path_to_sub_repo, url, branch)


def create_tag(repo, tag_name, ref_branch, message):
    """
    Creates a new tag for the branch
    :param repo: Repo object where the tag has to be created
    :param tag_name: Name for the tag
    :param ref_branch: Branch where the tag is to be created
    :param message: Message for the tag
    :return:
    """
    tag = repo.create_tag(tag_name, ref=ref_branch, message=message)
    tag.commit()


def create_head(repo, name):
    """
    Creates a new branch
    :param name: Name of the new branch
    :return: Object to the new branch
    """
    new_branch = repo.create_head(name)
    assert (repo.active_branch != new_branch)
    return new_branch


def create_remote(repo, remote_name, url=''):
    """
    :param repo: The repo object that has to be passed
    :param remote_name: Name of the remote repository
    :param url: URL to the remote repository
    :return:
    """
    return repo.create_remote(name=remote_name, url=url)


def create_repo(path, bare=True):
    """
    Creates a local repository
    :param bare: Creates a bare git repository
    :return: Repo object
    """
    return Repo.init(path, bare)


def open_existing_repo(path: str, search_parents=True):
    import os
    if os.path.exists(path=path):
        if repo_exists(dir_path=path):
            return Repo(path=path, search_parent_directories=search_parents)

        elif search_parents is True:
            return Repo(path=path, search_parent_directories=search_parents)

    return None
