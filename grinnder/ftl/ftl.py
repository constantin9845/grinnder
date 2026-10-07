import os
import ctypes
import numpy as np
import torch

def _load_ops():
    """Lazy-load C++ extension ops."""
    try:
        import grinnder._C as _C
        return _C
    except ImportError:
        return None

# FEMU PPA Bitfield Constants (64-bit total)
BLK_BITS = 16
PG_BITS = 16
SEC_BITS = 8
PL_BITS = 4
LUN_BITS = 7
CH_BITS = 12
RSV_BITS = 1

class FTL:
    def __init__(self, device="/dev/nvme0n1"):
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

        # SSD Geometry Sector Offsets (Replaces generating a 134M element mapping list)
        self.sec_per_pg    = self.layout['sector']
        self.sec_per_blk   = self.sec_per_pg  * self.layout['page']
        self.sec_per_pl    = self.sec_per_blk * self.layout['block']
        self.sec_per_lun   = self.sec_per_pl  * self.layout['plane']
        self.sec_per_ch    = self.sec_per_lun * self.layout['lun']
        self.total_sectors = self.sec_per_ch  * self.layout['ch']

        self.next_free_lba = 0
        self.tensor_table = {}

        self._ops = _load_ops()
        self.device_fd = self.open_device(device)
        self.cuFD = self._ops.register_fd(self.device_fd) if self._ops is not None else None

    def open_device(self, device):
        return os.open(device, os.O_RDWR)

    def close_device(self):
        if hasattr(self, 'device_fd') and self.device_fd is not None:
            os.close(self.device_fd)

    def get_flash_layout(self, device):
        return {
            "ch" : 8,
            "lun" : 8,
            "plane" : 1,
            "block" : 512,
            "page" : 512,
            "sector" : 8,
            "sector_size" : 512
        }

    def lba_to_ppa_uint64_vectorized(self, lba_array: np.ndarray) -> np.ndarray:
        """
        Converts LBAs directly into 64-bit uint64 FEMU PPAs at C-speed.
        Replaces self.lba_to_ppa(lba).to_uint64() Python object calls.
        """
        lbas = lba_array.astype(np.uint64)

        ch = lbas // self.sec_per_ch
        rem = lbas % self.sec_per_ch

        lun = rem // self.sec_per_lun
        rem %= self.sec_per_lun

        pl = rem // self.sec_per_pl
        rem %= self.sec_per_pl

        blk = rem // self.sec_per_blk
        rem %= self.sec_per_blk

        pg = rem // self.sec_per_pg
        sec = rem % self.sec_per_pg
        rsv = np.uint64(1)

        # Bitpack matching StructPPA bitfield
        ppa_uint64 = (
            (blk & np.uint64((1 << BLK_BITS) - 1)) |
            ((pg & np.uint64((1 << PG_BITS) - 1)) << np.uint64(16)) |
            ((sec & np.uint64((1 << SEC_BITS) - 1)) << np.uint64(32)) |
            ((pl & np.uint64((1 << PL_BITS) - 1)) << np.uint64(40)) |
            ((lun & np.uint64((1 << LUN_BITS) - 1)) << np.uint64(44)) |
            ((ch & np.uint64((1 << CH_BITS) - 1)) << np.uint64(51)) |
            ((rsv & np.uint64((1 << RSV_BITS) - 1)) << np.uint64(63))
        )
        return ppa_uint64

    def map_tensor(self, file_id: str, tensor: torch.Tensor):
        if file_id in self.tensor_table:
            raise KeyError(f"Tensor '{file_id}' is already registered!")

        tensor = tensor.contiguous()
        element_size = tensor.element_size()  
        total_elements = tensor.numel()
        total_bytes = total_elements * element_size

        sector_step = self.ssd_data[5]  # 8 sectors (4096 bytes)
        bytes_per_page = sector_step * self.ssd_data[6]  # 8 * 512 = 4096
        pages_needed = (total_bytes + bytes_per_page - 1) // bytes_per_page
        sectors_needed = pages_needed * sector_step

        if self.next_free_lba % sector_step != 0:
            self.next_free_lba += (sector_step - (self.next_free_lba % sector_step))

        start_lba = self.next_free_lba
        end_lba = start_lba + sectors_needed - 1

        if end_lba >= self.total_sectors:
            raise MemoryError("Out of SSD capacity!")

        # Generate LBA list directly in C/NumPy memory
        page_start_lbas = np.arange(start_lba, end_lba + 1, sector_step, dtype=np.uint64)
        
        # Convert all LBAs to 64-bit uint64 PPAs in < 3 milliseconds
        mapped_ppas = self.lba_to_ppa_uint64_vectorized(page_start_lbas)

        self.tensor_table[file_id] = {
            "file_id": file_id,
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "num_elements": total_elements,
            "element_size": element_size,
            "total_bytes": total_bytes,
            "sector_size": bytes_per_page,  # Pass 4096
            "start_lba": start_lba,
            "end_lba": end_lba,
            "num_sectors": len(page_start_lbas), # Number of 4KB transfers
            "lba_list": page_start_lbas,
            "ppa_uint64_list": mapped_ppas, # NumPy array of uint64
        }

        self.next_free_lba = end_lba + 1

        print(
            f"[FTL] Registered '{file_id}': LBAs {start_lba}..{end_lba} "
            f"({sectors_needed} sectors, {total_bytes / (1024**2):.2f} MB)"
        )
        return self.tensor_table[file_id]

    def get_boundary_row_info(self, file_id, row_idx):
        if file_id not in self.tensor_table:
            raise KeyError(f"Tensor '{file_id}' not found in FTL table.")
        
        meta = self.tensor_table[file_id]

        shape = meta["shape"]
        num_cols = shape[1] if len(shape) > 1 else 1
        row_size_bytes = num_cols * meta["element_size"]

        start_byte = row_idx * row_size_bytes
        end_byte = start_byte + row_size_bytes - 1

        bytes_per_page = meta["sector_size"]  # 4096
        start_page_idx = start_byte // bytes_per_page
        end_page_idx = end_byte // bytes_per_page

        target_ppas = meta["ppa_uint64_list"][start_page_idx : end_page_idx + 1]
        in_page_offset = start_byte % bytes_per_page

        return {
            "row_idx": row_idx,
            "ppas": target_ppas,
            "ppa_tensor": self._convert_ppas_to_tensor(target_ppas),
            "num_pages": len(target_ppas),
            "in_page_offset": in_page_offset,
            "row_size_bytes": row_size_bytes,
        }

    def get_tensor_ppa_map(self, file_id):
        if file_id not in self.tensor_table:
            raise KeyError(f"Tensor '{file_id}' not found in FTL table.")

        meta = self.tensor_table[file_id]
        return list(zip(meta["lba_list"], meta["ppa_uint64_list"]))

    def delete_tensor(self, file_id):
        if file_id in self.tensor_table:
            del self.tensor_table[file_id]
            print(f"[FTL] Unmapped tensor '{file_id}'")

    def _convert_ppas_to_tensor(self, ppa_array):
        if isinstance(ppa_array, torch.Tensor):
            return ppa_array
        if isinstance(ppa_array, list):
            ppa_array = np.array(ppa_array, dtype=np.uint64)
        np_ppas = ppa_array.view(np.int64)
        return torch.from_numpy(np_ppas)

    def device_write(self, meta, tensor, file_id):
        assert not tensor.is_cuda, "device write uses cpu tensor"
        assert tensor.is_contiguous(), "device write requires contiguous tensor"

        ppa_tensor = self._convert_ppas_to_tensor(meta['ppa_uint64_list'])
        self._ops.device_write(self.cuFD, tensor, ppa_tensor, self.ssd_data[-1])

    def verify_write(self, meta, tensor, file_id):
        assert not tensor.is_cuda, "device_read_and_verify requires a CUDA tensor"
        assert tensor.is_contiguous(), "device_read_and_verify requires contiguous tensor"

        ppa_tensor = self._convert_ppas_to_tensor(meta['ppa_uint64_list'])

        is_correct = self._ops.device_read_and_verify(
            self.cuFD,
            tensor,
            ppa_tensor,
            self.ssd_data[-1]  # sector size
        )
        return is_correct