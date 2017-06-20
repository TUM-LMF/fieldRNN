import os
import shutil

def replace_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)