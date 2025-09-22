

class Logger:
    def __int__(self):
        super(Logger, self).__int__()

    def open(self, name, mode):
        self.txt = open(name, mode=mode)

    def write(self, str_):
        self.txt.write(str_)
        self.txt.write('\n')
        print(str_)

    def close(self):
        self.txt.close()

def createLogger(args):
    log = Logger()
    log.open(args.train_logs, mode='w')
    log.write('\n')
    return log


