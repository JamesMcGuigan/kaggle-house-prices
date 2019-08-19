import os

def reset_root_dir():
    os.chdir( os.path.join( os.path.dirname(__file__), '../..' ))
    return os.getcwd()