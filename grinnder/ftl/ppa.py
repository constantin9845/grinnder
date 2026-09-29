
import ctypes

class PPA:
    def __init__(self, data, ssd_data):

        # ppa data
        self.ch = data[0]
        self.lun = data[1]
        self.pl = data[2]
        self.blk = data[3]
        self.pg = data[4]
        self.sec = data[5]
        self.rsv = 1

        # ssd data
        self.nch = ssd_data[0]
        self.nlun = ssd_data[1]
        self.npl = ssd_data[2]
        self.nblk = ssd_data[3]
        self.npg = ssd_data[4]
        self.nsec = ssd_data[5]

        self.pages_per_plane = self.npg * self.nblk
        self.pages_per_lun = self.pages_per_plane * self.npl
        self.pages_per_ch = self.pages_per_lun * self.nlun
        self.tt_pages = self.pages_per_ch * self.nch

    def get_data(self):
        return [self.ch, self.lun, self.pl, self.blk, self.pg, self.sec, self.rsv]

    def get_index(self):

        pgidx = (
            (self.nch * self.pages_per_ch) 
            + (self.nlun * self.pages_per_lun) 
            + (self.npl * self.pages_per_plane) 
            + (self.nblk * self.npg) 
            + self.pg)
        
        if pgidx > self.tt_pages:
            print(f"Invalid pgidx = {pgidx}")

        return pgidx