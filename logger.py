import os 
class Logger(object):
    def __init__(self, path):
        self.file = open(path, 'a+')

    def write(self, log):
        self.file.write(log)
        print(log)

    def close(self):
        self.file.close()

    def flush(self):
        self.file.flush()
