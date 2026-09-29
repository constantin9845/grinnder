"""
struct femu_ppa {
    union {
        struct {
            uint32_t blk : BLK_BITS;
            uint32_t pg  : PG_BITS;
            uint32_t sec : SEC_BITS;
            uint32_t pl  : PL_BITS;
            uint32_t lun : LUN_BITS;
            uint32_t ch  : CH_BITS;
            uint32_t rsv : 1;
        } g;

        uint32_t ppa;
    };
};
PPA:
    BLK
    PG
    SEC
    PLANE
    LUN
    CHANNEL
"""

from ppa import PPA


class FTL:

    def __init__(self, device):
        self.layout = self.get_flash_layout(device)
        self.mapping_table = self.generate_map()

        print(f"Device : {device}")
        for i in self.mapping_table:
            print(f"{i.get_data()}")

    def open_device(self, device):
        # open if closed
        pass

    def close_device(self, device):
        # close if open
        pass

    def get_flash_layout(self, device):
        # send admin query to ssd
        t = {
            "ch" : 4,
            "lun" : 4,
            "plane" : 1,
            "block" : 64,
            "page" : 64,
            "sector" : 1
        }
        return t

    def generate_map(self):
        map = []
        nchannels = self.layout['ch']
        for channel in range(nchannels):
            self.init_ch(channel, map)

        return map

    def init_ch(self, channel, map):
        luns_per_channel = self.layout['lun']
        for lun in range(luns_per_channel):
            self.init_planes([channel, lun], map)


    def init_planes(self, data, map):
        planes_per_lun = self.layout['plane']
        for plane in range(planes_per_lun):
            self.init_block(data + [plane], map)

    def init_block(self, data, map):
        blocks_per_plane = self.layout['block']
        for block in range(blocks_per_plane):
            self.init_page(data + [block], map)

    def init_page(self, data, map):
        pages_per_block = self.layout['page']
        for page in range(pages_per_block):
            self.init_sector(data + [page], map)

    def init_sector(self, data, map):
        sectors_per_page = self.layout['sector']
        for sector in range(sectors_per_page):
            full = data + [sector]
            map.append(PPA(full))


    def lba_to_ppa(self, lba):
        pass

    def ppa_to_lba(self, ppa):
        pass



if __name__ == "__main__":
    ftl = FTL("/dev/nvme0n1")