import os
from pathlib import Path

def get_repo_folder():
    """Find the repository folder by traversing up the directory tree."""
    current = Path.cwd()
    if current.name == "PrivateTeacherAgent":
        return current
    for parent in current.parents:
        if parent.name == "PrivateTeacherAgent":
            return parent
    raise FileNotFoundError("Repository folder 'PrivateTeacherAgent' not found.")

def get_token_count_file_path():
    return get_repo_folder() / os.path.join('tokens_count','total_tokens.csv')