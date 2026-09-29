
import ctypes

class ppa:
    def __init__(self, data):
        self.blk = data[0]
        self.pg = data[1]
        self.sec = data[2]
        self.pl = data[3]
        self.lun = data[4]
        self.ch = data[5]
        self.rsv = 1

    def get_data(self):
        return [self.blk, self.pg, self.sec, self.pl, self.lun, self.ch, self.rsv]