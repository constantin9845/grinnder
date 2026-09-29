
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

        self.sec_per_pg    = self.nsec
        self.sec_per_blk   = self.sec_per_pg  * self.npg
        self.sec_per_pl    = self.sec_per_blk * self.nblk
        self.sec_per_lun   = self.sec_per_pl  * self.npl
        self.sec_per_ch    = self.sec_per_lun * self.nlun
        self.total_sectors = self.sec_per_ch  * self.nch

    def get_data(self):
        return [self.ch, self.lun, self.pl, self.blk, self.pg, self.sec, self.rsv]

    def get_index(self):

        lba = (
            (self.ch * self.sec_per_ch) 
            + (self.lun * self.sec_per_lun) 
            + (self.pl * self.sec_per_pl) 
            + (self.blk * self.sec_per_blk) 
            + (self.pg * self.sec_per_pg) 
            + self.sec)
        
        if lba > self.total_sectors:
            print(f"Invalid lba = {lba}")

        return lba