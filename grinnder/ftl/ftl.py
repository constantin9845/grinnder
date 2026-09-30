from ppa import PPA
import torch


class FTL:

    def __init__(self, device):
        self.layout = self.get_flash_layout(device)
        self.ssd_data = [
            self.layout['ch'], 
            self.layout['lun'], 
            self.layout['plane'], 
            self.layout['block'], 
            self.layout['page'], 
            self.layout['sector']
        ]
        self.mapping_table = self.generate_map()

        self.reverse_table = {
            tuple(ppa.get_data()[:6]): lba 
            for lba, ppa in enumerate(self.mapping_table)
        }

        self.next_free_lba = 0
        self.tensor_table = {}

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
            map.append(PPA(full, self.ssd_data))


    def lba_to_ppa(self, lba):
        return self.mapping_table[lba]

    def ppa_to_lba(self, ppa):
        ppa_key = tuple(ppa.get_data()[:6])
        return self.reverse_table.get(ppa_key, None)

    def map_tensor(self, file_id, tensor, sector_size):

        if file_id in self.tensor_table:
            raise KeyError(f"Tensor '{file_id}' is already registered!")

        tensor = tensor.contiguous()
        element_size = tensor.element_size()  
        total_elements = tensor.numel()
        total_bytes = total_elements * element_size
        sectors_needed = (total_bytes + sector_size - 1) // sector_size

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
            "sector_size": sector_size,
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




ftl = FTL(None)