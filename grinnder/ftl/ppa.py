
import ctypes

class PPA:
    def __init__(self, data):
        self.ch = data[0]
        self.lun = data[1]
        self.pl = data[2]
        self.blk = data[3]
        self.pg = data[4]
        self.sec = data[5]
        self.rsv = 1

    def get_data(self):
        return [self.ch, self.lun, self.pl, self.blk, self.pg, self.sec, self.rsv]