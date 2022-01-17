import os
import subprocess

def get_git_commit():
    cmd = "git log --pretty=oneline -n 1"
    git_commit = str(os.system(cmd))
    return git_commit

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8").rstrip()

def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
