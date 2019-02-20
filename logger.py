class Logger(object): 
    def __init__(self, path,write_type): 
        self.file = open(path,'a+') 
    def write(self,log): 
        self.file.write(log)
        print(log) 
    def close(self): 
        self.file.close() 
    def flush(self): 
        self.file.flush() 