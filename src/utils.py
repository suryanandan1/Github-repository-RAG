import os
import shutil

def clean_directory(path: str):
    """
    Remove existing repository before cloning again.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
