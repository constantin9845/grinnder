import ctypes

# FEMU ppa format
BLK_BITS = 16
PG_BITS = 16
SEC_BITS = 8
PL_BITS = 8
LUN_BITS = 8
CH_BITS = 7
RSV_BITS = 1

class PPA_Bitfield(ctypes.LittleEndianStructure):
    _fields_ = [
        ("blk", ctypes.c_uint64, BLK_BITS),
        ("pg", ctypes.c_uint64, PG_BITS),
        ("sec", ctypes.c_uint64, SEC_BITS),
        ("pl", ctypes.c_uint64, PL_BITS),
        ("lun", ctypes.c_uint64, LUN_BITS),
        ("ch", ctypes.c_uint64, CH_BITS),
        ("rsv", ctypes.c_uint64, RSV_BITS),
    ]

class StructPPA(ctypes.Union):
    _anonymous_ = ("g",)
    _fields_ = [
        ("g", PPA_Bitfield),
        ("ppa", ctypes.c_uint64),
    ]

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

        self._c_ppa = StructPPA()
        self._c_ppa.blk = self.blk
        self._c_ppa.pg = self.pg
        self._c_ppa.sec = self.sec
        self._c_ppa.pl = self.pl
        self._c_ppa.lun = self.lun
        self._c_ppa.ch = self.ch
        self._c_ppa.rsv = self.rsv

        # ssd data
        self.nch = ssd_data[0]
        self.nlun = ssd_data[1]
        self.npl = ssd_data[2]
        self.nblk = ssd_data[3]
        self.npg = ssd_data[4]
        self.nsec = ssd_data[5]

        self.sec_per_pg = self.nsec

        self.sec_per_blk = (
            self.sec_per_pg * self.npg
        )

        self.sec_per_pl = (
            self.sec_per_blk * self.nblk
        )

        self.sec_per_lun = (
            self.sec_per_pl * self.npl
        )

        self.sec_per_ch = (
            self.sec_per_lun * self.nlun
        )

        self.total_sectors = (
            self.sec_per_ch * self.nch
        )

        self.pages_per_plane = (
            self.nblk * self.npg
        )

        self.pages_per_lun = (
            self.npl * self.pages_per_plane
        )

        self.pages_per_channel = (
            self.nlun * self.pages_per_lun
        )

        self.total_pages = (
            self.nch * self.pages_per_channel
        )

    def get_data(self):
        return [self.ch, self.lun, self.pl, self.blk, self.pg, self.sec, self.rsv]

    def to_uint64(self):
        return self._c_ppa.ppa

    @classmethod
    def from_uint64(cls, raw_ppa, ssd_data):
        c_ppa = StructPPA(ppa=raw_ppa)
        return cls(
            [
                c_ppa.ch,
                c_ppa.lun,
                c_ppa.pl,
                c_ppa.blk,
                c_ppa.pg,
                c_ppa.sec,
                c_ppa.rsv,
            ],
            ssd_data
        )

    def get_index(self):

        index = (
            (self.ch * self.sec_per_ch)
            + (self.lun * self.sec_per_lun)
            + (self.pl * self.sec_per_pl)
            + (self.blk * self.sec_per_blk)
            + (self.pg * self.sec_per_pg)
            + self.sec
        )

        if index >= self.total_sectors:
            raise ValueError(
                f"Invalid physical sector index: {index}"
            )

        return index

    