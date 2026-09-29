
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

    def get_index(self):
        t = 0
        if self.ch != 0:
            t = self.ch

        if self.lun != 0:
            t *= self.lun

        if self.pl != 0:
            t *= self.pl

        if self.blk != 0:
            t *= self.blk

        if self.pg != 0:
            t *= self.pg

        if self.sec != 0:
            t *= self.sec

        return t