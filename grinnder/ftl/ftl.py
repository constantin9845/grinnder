import torch
import os
import ctypes

def _load_ops():
    """Lazy-load C++ extension ops."""
    try:
        import grinnder._C as _C
        return _C
    except ImportError:
        return None

# FEMU ppa format
BLK_BITS = 16
PG_BITS = 16
SEC_BITS = 8
PL_BITS = 4
LUN_BITS = 7
CH_BITS = 12
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

        self.sec_per_pg    = self.nsec
        self.sec_per_blk   = self.sec_per_pg  * self.npg
        self.sec_per_pl    = self.sec_per_blk * self.nblk
        self.sec_per_lun   = self.sec_per_pl  * self.npl
        self.sec_per_ch    = self.sec_per_lun * self.nlun
        self.total_sectors = self.sec_per_ch  * self.nch

    def get_data(self):
        return [self.ch, self.lun, self.pl, self.blk, self.pg, self.sec, self.rsv]

    def to_uint64(self):
        return self._c_ppa.ppa

    @classmethod
    def from_uint64(cls, raw_ppa):
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
            ]
        )

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

class FTL:

    def __init__(self, device):
        self.layout = self.get_flash_layout(device)
        self.ssd_data = [
            self.layout['ch'], 
            self.layout['lun'], 
            self.layout['plane'], 
            self.layout['block'], 
            self.layout['page'], 
            self.layout['sector'],
            self.layout['sector_size'],
        ]
        self.mapping_table = self.generate_map()

        self.reverse_table = {
            tuple(ppa.get_data()[:6]): lba 
            for lba, ppa in enumerate(self.mapping_table)
        }

        self.next_free_lba = 0
        self.tensor_table = {}

        self._ops = _load_ops()

        self.device_fd = self.open_device("/dev/nvme0n1")

    def open_device(self, device):
        # open if closed
        fd = os.open(device, os.O_RDWR | os.O_DIRECT)
        return fd

    def close_device(self, device):
        device.close()

    def get_flash_layout(self, device):
        # send admin query to ssd
        t = {
            "ch" : 8,
            "lun" : 8,
            "plane" : 1,
            "block" : 512,
            "page" : 512,
            "sector" : 1,
            "sector_size" : 4096
        }
        return t

    def generate_map(self):
        map = []
        nchannels = self.layout['ch']
        print("Starting init SSD\n")
        for channel in range(nchannels):
            self.init_ch(channel, map)
            print(f"Init channel {channel} done")

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
            map.append(PPA(full, self.ssd_data))


    def lba_to_ppa(self, lba):
        return self.mapping_table[lba]

    def ppa_to_lba(self, ppa):
        ppa_key = tuple(ppa.get_data()[:6])
        return self.reverse_table.get(ppa_key, None)

    def map_tensor(self, file_id, tensor):

        if file_id in self.tensor_table:
            raise KeyError(f"Tensor '{file_id}' is already registered!")

        tensor = tensor.contiguous()
        element_size = tensor.element_size()  
        total_elements = tensor.numel()
        total_bytes = total_elements * element_size
        sectors_needed = (total_bytes + self.ssd_data[-1] - 1) // self.ssd_data[-1]

        start_lba = self.next_free_lba
        end_lba = start_lba + sectors_needed - 1

        if end_lba >= len(self.mapping_table):
            raise MemoryError("Out of SSD capacity!")

        mapped_lbas = list(range(start_lba, end_lba + 1))
        mapped_ppas = [
            self.lba_to_ppa(lba).to_uint64() for lba in mapped_lbas
        ]

        self.tensor_table[file_id] = {
            "file_id": file_id,
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "num_elements": total_elements,
            "element_size": element_size,
            "total_bytes": total_bytes,
            "sector_size": self.ssd_data[-1],
            "start_lba": start_lba,
            "end_lba": end_lba,
            "num_sectors": sectors_needed,
            "lba_list": mapped_lbas,  # Array of allocated LBAs
            "ppa_uint64_list": mapped_ppas,  # Array of FEMU packed uint64 PPAs
        }

        self.next_free_lba = end_lba + 1

        print(
            f"[FTL] Registered '{file_id}': LBAs {start_lba}..{end_lba} "
            f"({sectors_needed} sectors, {total_bytes / (1024**2):.2f} MB)"
        )
        return self.tensor_table[file_id]


    def get_tensor_ppa_map(self, file_id):
        if(file_id) not in self.tensor_table:
            raise KeyError(f"Tensor '{file_id}' not found in FTL table.")

        meta = self.tensor_table[file_id]
        return list(zip(meta["lba_list"], meta["ppa_uint64_list"]))

    def delete_tensor(self, file_id):
        if file_id in self.tensor_table:
            del self.tensor_table[file_id]
            print(f"[FTL] Unmapped tensor '{file_id}'")

    def device_write(self, meta, tensor, file_id):
        assert not tensor.is_cuda, "device write uses cpu tensor"
        assert tensor.is_contiguous(), "device_read_and_verify requires contiguous tensor"

        ppa_tensor = torch.tensor(meta['ppa_uint64_list'], dtype=torch.int64, device="cpu")

        self._ops.device_write(self.device_fd, tensor, ppa_tensor, self.ssd_data[-1])

    def verify_write(self, meta, tensor, file_id):
        assert tensor.is_cuda, "device_read_and_verify requires a CUDA tensor"
        assert tensor.is_contiguous(), "device_read_and_verify requires contiguous tensor"

        ppa_tensor = torch.tensor(meta['ppa_uint64_list'], dtype=torch.int64, device="cpu")

        is_correct = self._ops.device_read_and_verify(
            self.device_fd,
            tensor,
            ppa_tensor,
            self.ssd_data[-1]  # sector size
        )
        return is_correct
