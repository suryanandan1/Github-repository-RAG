import os
from git import Repo

from src.utils import clean_directory

def clone_repository(repo_url: str, clone_path: str):
    """
    Clone GitHub repository.
    """
    clean_directory(clone_path)

    Repo.clone_from(
        repo_url,
        clone_path
    )

    return clone_path

