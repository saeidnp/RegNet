import time
import pickle
from pathlib import Path
import os

def myassert(condition, msg):
    if not condition:
        raise Exception(msg)

def myassertList(my_var, my_list):
    if my_var not in my_list:
        msg = 'Expected one of ' + ','.join(map(str,my_list)) + \
                ' but got {0}'.format(my_var)
        raise Exception(msg)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def doesFileExist(path):
    my_file = Path(path)
    if my_file.is_file():
        return True
    return False

def doesDirExist(path):
    my_file = Path(path)
    if my_file.is_dir():
        return True
    return False

def makeDir(path):
    os.makedirs(path)

def checkMakeDir(path):
    if not doesDirExist(path):
        makeDir(path)

def wipeDir(path):
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
            raise e

class StopWatch:
    def __init__(self):
        self.start_time = None

    def start(self):
        """Starts the timer"""
        self.start_time = time.time()
        self.last_lap = self.start_time
        return self.start_time

    def stop(self):
        """Stops the timer.  Returns the time elapsed"""
        self.stop_time = time.time()
        ret_val = self.stop_time - self.start_time
        self.start_time = None
        return ret_val

    def lap(self):
        if self.start_time == None:
            self.start()
            return 0

        new_lap = time.time()
        lap_time = new_lap - self.last_lap
        self.last_lap = new_lap
        return lap_time

    def getElapsed(self):
        return time.time() - self.start_time