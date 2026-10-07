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

class PPA_Bitfield(ctypes.LittleEndianStructure):
    _fields_ = [
        ("blk", ctypes.c_uint64, BLK_BITS),
        ("pg",  ctypes.c_uint64, PG_BITS),
        ("sec", ctypes.c_uint64, SEC_BITS),
        ("pl",  ctypes.c_uint64, PL_BITS),
        ("lun", ctypes.c_uint64, LUN_BITS),
        ("ch",  ctypes.c_uint64, CH_BITS),
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
        self.ch = data[0]
        self.lun = data[1]
        self.pl = data[2]
        self.blk = data[3]
        self.pg = data[4]
        self.sec = data[5]
        self.rsv = data[6] if len(data) > 6 else 1

        self._c_ppa = StructPPA()

        self._c_ppa.blk = self.blk
        self._c_ppa.pg = self.pg
        self._c_ppa.sec = self.sec
        self._c_ppa.pl = self.pl
        self._c_ppa.lun = self.lun
        self._c_ppa.ch = self.ch
        self._c_ppa.rsv = self.rsv

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
        return [
            self.ch,
            self.lun,
            self.pl,
            self.blk,
            self.pg,
            self.sec,
            self.rsv,
        ]

    def to_uint64(self):
        """
        Convert PPA to FEMU's 64-bit PPA representation.
        """
        return self._c_ppa.ppa

    @classmethod
    def from_uint64(cls, raw_ppa, ssd_data):
        """
        Convert a 64-bit FEMU PPA back into a PPA object.
        """

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
            ssd_data,
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

    def get_page_index(self):

        index = (
            (self.ch * self.pages_per_channel)
            + (self.lun * self.pages_per_lun)
            + (self.pl * self.pages_per_plane)
            + (self.blk * self.npg)
            + self.pg
        )

        if index >= self.total_pages:
            raise ValueError(
                f"Invalid physical page index: {index}"
            )

        return index

    def same_page(self, other):
        return (
            self.ch == other.ch
            and self.lun == other.lun
            and self.pl == other.pl
            and self.blk == other.blk
            and self.pg == other.pg
        )

    def __repr__(self):
        return (
            f"PPA(ch={self.ch}, lun={self.lun}, pl={self.pl}, "
            f"blk={self.blk}, pg={self.pg}, sec={self.sec})"
        )


class FTL:
    def __init__(self, device="/dev/nvme0n1"):

        '''
            1 sector = 512 B
            8 sectors = 4096 B
            NAND page = 4 KiB

            1 LPN = 1 NAND page
            1 LPN = 8 LBAs/sectors

            LPN -> physical page PPA
        '''
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

        self.nch = self.ssd_data[0]
        self.nlun = self.ssd_data[1]
        self.npl = self.ssd_data[2]
        self.nblk = self.ssd_data[3]
        self.npg = self.ssd_data[4]
        self.nsec = self.ssd_data[5]

        self.sector_size = 512
        self.page_size = (self.nsec * self.sector_size)
        self.pages_per_block = self.npg

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

        self.total_sectors = (
            self.total_pages * self.nsec
        )

        self.total_lpn = self.total_pages

        self.total_lba = (
            self.total_lpn * self.nsec
        )

        self.l2p = {}
        self.p2l = {}
        self.valid = {} # page valid/invalid
        self.next_page = 0

        self.tensor_table = {}
        self.next_free_lba = 0

        self._ops = _load_ops()
        self.device_fd = self.open_device(device)
        self.cuFD = self._ops.register_fd(self.device_fd) if self._ops is not None else None

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

    def lba_to_lpn(self, lba):
        if lba < 0 or lba >= self.total_lba:
            raise ValueError(
                f"Invalid LBA {lba}. "
                f"Valid range: 0-{self.total_lba - 1}"
            )

        lpn = lba // self.nsec
        sec = lba % self.nsec

        return lpn, sec

    def get_ppa(self, lpn):
        if lpn < 0 or lpn >= self.total_lpn:
            raise ValueError(
                f"Invalid LPN {lpn}"
            )

        return self.l2p.get(lpn)

    def get_ppa_for_lba(self, lba):

        lpn, sec = self.lba_to_lpn(lba)

        page_ppa = self.get_ppa(lpn)

        if page_ppa is None:
            return None

        return PPA(
            [
                page_ppa.ch,
                page_ppa.lun,
                page_ppa.pl,
                page_ppa.blk,
                page_ppa.pg,
                sec,
                page_ppa.rsv,
            ],
            self.ssd_data,
        )

    def page_index_to_ppa(self, page_index):

        if page_index < 0 or page_index >= self.total_pages:
            raise ValueError(
                f"Invalid physical page index {page_index}"
            )

        ch = (
            page_index // self.pages_per_channel
        )

        rem = (
            page_index % self.pages_per_channel
        )

        lun = (
            rem // self.pages_per_lun
        )

        rem %= self.pages_per_lun

        pl = (
            rem // self.pages_per_plane
        )

        rem %= self.pages_per_plane

        blk = (
            rem // self.npg
        )

        pg = (
            rem % self.npg
        )

        return PPA(
            [
                ch,
                lun,
                pl,
                blk,
                pg,
                0,
                1,
            ],
            self.ssd_data,
        )

    def allocate_page(self):
        if self.next_page >= self.total_pages:
            raise RuntimeError(
                "FTL has no free physical pages. "
                "GC is not implemented yet."
            )

        page_index = self.next_page

        self.next_page += 1

        ppa = self.page_index_to_ppa(
            page_index
        )

        self.valid[page_index] = True

        return ppa

    def write_lpn(self, lpn):
        if lpn < 0 or lpn >= self.total_lpn:
            raise ValueError(
                f"Invalid LPN {lpn}")

        old_ppa = self.l2p.get(lpn)
        new_ppa = self.allocate_page()

        new_page_index = (
            new_ppa.get_page_index()
        )

        if old_ppa is not None:

            old_page_index = (
                old_ppa.get_page_index()
            )

            self.valid[old_page_index] = False

            self.p2l.pop(
                old_page_index,
                None
            )

        self.l2p[lpn] = new_ppa

        self.p2l[new_page_index] = lpn

        self.valid[new_page_index] = True

        return new_ppa

    def write_page(self, lpn):
        return self.write_lpn(lpn)

    def write_lba(self, lba):
        lpn, sec = self.lba_to_lpn(lba)

        if sec != 0:
            raise ValueError(
                f"LBA {lba} is not 4 KiB/page aligned. "
                f"Expected LBA % {self.nsec} == 0."
            )

        return self.write_lpn(lpn)

    def set_mapping(self, lpn, ppa):
        if lpn < 0 or lpn >= self.total_lpn:
            raise ValueError(
                f"Invalid LPN {lpn}"
            )

        if ppa.sec != 0:
            raise ValueError(
                "set_mapping() expects a page PPA "
                "with sec == 0."
            )

        page_index = ppa.get_page_index()

        existing_lpn = self.p2l.get(
            page_index
        )

        if (
            existing_lpn is not None
            and existing_lpn != lpn
        ):
            raise ValueError(
                f"Physical page {page_index} is already "
                f"mapped to LPN {existing_lpn}"
            )

        old_ppa = self.l2p.get(lpn)

        if old_ppa is not None:

            old_index = (
                old_ppa.get_page_index()
            )

            self.valid[old_index] = False

            self.p2l.pop(
                old_index,
                None
            )

        self.l2p[lpn] = ppa
        self.p2l[page_index] = lpn
        self.valid[page_index] = True

    def invalidate_lpn(self, lpn):

        old_ppa = self.l2p.pop(
            lpn,
            None
        )

        if old_ppa is None:
            return None

        page_index = (
            old_ppa.get_page_index()
        )

        self.valid[page_index] = False

        self.p2l.pop(
            page_index,
            None
        )

        return old_ppa

    def get_lpn(self, ppa):

        page_index = (
            ppa.get_page_index()
        )

        return self.p2l.get(
            page_index,
            None
        )

    def get_mapping(self, lpn):

        ppa = self.get_ppa(lpn)

        if ppa is None:
            return None

        return (
            lpn,
            ppa,
            ppa.get_page_index()
        )

    def lba_to_ppa_uint64_vectorized(
        self,
        lba_array: np.ndarray,
    ) -> np.ndarray:
        lbas = np.asarray(lba_array,dtype=np.uint64)

        result = np.empty(len(lbas),dtype=np.uint64)

        for i, lba in enumerate(lbas):

            lpn = int(lba) // self.nsec
            sec = int(lba) % self.nsec

            ppa = self.get_ppa(lpn)

            if ppa is None:
                raise KeyError(
                    f"LPN {lpn} has no physical mapping"
                )

            sector_ppa = PPA(
                [
                    ppa.ch,
                    ppa.lun,
                    ppa.pl,
                    ppa.blk,
                    ppa.pg,
                    sec,
                    ppa.rsv,
                ],
                self.ssd_data,
            )

            result[i] = sector_ppa.to_uint64()

        return result

    def map_tensor(
        self,
        file_id: str,
        tensor: torch.Tensor,
    ):
        if file_id in self.tensor_table:
            raise KeyError(
                f"Tensor '{file_id}' is already registered!"
            )

        tensor = tensor.contiguous()

        element_size = tensor.element_size()
        total_elements = tensor.numel()
        total_bytes = (
            total_elements * element_size
        )

        sectors_per_page = self.nsec

        bytes_per_page = (
            sectors_per_page * self.sector_size
        )

        pages_needed = (
            total_bytes + bytes_per_page - 1
        ) // bytes_per_page

        sectors_needed = (
            pages_needed * sectors_per_page
        )

        if self.next_free_lba % sectors_per_page != 0:

            self.next_free_lba += (
                sectors_per_page
                - (
                    self.next_free_lba
                    % sectors_per_page
                )
            )

        start_lba = self.next_free_lba

        end_lba = (
            start_lba
            + sectors_needed
            - 1
        )

        if end_lba >= self.total_lba:
            raise MemoryError(
                "Out of SSD logical capacity!"
            )

        page_start_lbas = np.arange(
            start_lba,
            end_lba + 1,
            sectors_per_page,
            dtype=np.uint64,
        )

        mapped_ppas = np.empty(
            pages_needed,
            dtype=np.uint64
        )

        for i, lba in enumerate(page_start_lbas):

            lpn = int(lba) // sectors_per_page

            # Allocate physical page and establish:
            #
            #     L2P[lpn] = physical PPA
            #
            ppa = self.write_lpn(lpn)

            mapped_ppas[i] = (
                ppa.to_uint64()
            )

        self.tensor_table[file_id] = {
            "file_id": file_id,
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "num_elements": total_elements,
            "element_size": element_size,
            "total_bytes": total_bytes,
            "sector_size": bytes_per_page,
            "start_lba": start_lba,
            "end_lba": end_lba,
            "num_sectors": pages_needed,
            "lba_list": page_start_lbas,
            "ppa_uint64_list": mapped_ppas,
        }

        self.next_free_lba = (
            end_lba + 1
        )

        print(
            f"[FTL] Registered '{file_id}': "
            f"LBAs {start_lba}..{end_lba} "
            f"({sectors_needed} sectors, "
            f"{total_bytes / (1024**2):.2f} MB)"
        )

        return self.tensor_table[file_id]

    def _get_tensor_current_ppas(
        self,
        file_id,
    ):
        if file_id not in self.tensor_table:
            raise KeyError(
                f"Tensor '{file_id}' not found in FTL table."
            )

        meta = self.tensor_table[file_id]

        lba_list = meta["lba_list"]

        ppas = np.empty(
            len(lba_list),
            dtype=np.uint64
        )

        for i, lba in enumerate(lba_list):

            lpn = int(lba) // self.nsec

            ppa = self.get_ppa(lpn)

            if ppa is None:
                raise RuntimeError(
                    f"LPN {lpn} has no physical mapping"
                )

            ppas[i] = ppa.to_uint64()

        return ppas

    def get_boundary_row_info(
        self,
        file_id,
        row_idx,
    ):
        if file_id not in self.tensor_table:
            raise KeyError(
                f"Tensor '{file_id}' not found in FTL table."
            )

        meta = self.tensor_table[file_id]

        shape = meta["shape"]

        if len(shape) > 1:
            num_cols = shape[1]
        else:
            num_cols = 1

        row_size_bytes = (
            num_cols
            * meta["element_size"]
        )

        start_byte = (
            row_idx * row_size_bytes
        )

        end_byte = (
            start_byte
            + row_size_bytes
            - 1
        )

        bytes_per_page = (
            meta["sector_size"]
        )

        start_page_idx = (
            start_byte // bytes_per_page
        )

        end_page_idx = (
            end_byte // bytes_per_page
        )

        current_ppas = (
            self._get_tensor_current_ppas(
                file_id
            )
        )

        target_ppas = (
            current_ppas[
                start_page_idx:
                end_page_idx + 1
            ]
        )

        in_page_offset = (
            start_byte
            % bytes_per_page
        )

        return {
            "row_idx": row_idx,

            "ppas": target_ppas,

            "ppa_tensor": (
                self._convert_ppas_to_tensor(
                    target_ppas
                )
            ),

            "num_pages": len(target_ppas),

            "in_page_offset": in_page_offset,

            "row_size_bytes": row_size_bytes,
        }

    def get_tensor_ppa_map(
        self,
        file_id,
    ):
        if file_id not in self.tensor_table:
            raise KeyError(
                f"Tensor '{file_id}' not found in FTL table."
            )

        meta = self.tensor_table[file_id]

        # Return CURRENT physical mapping.
        current_ppas = (
            self._get_tensor_current_ppas(
                file_id
            )
        )

        return list(
            zip(
                meta["lba_list"],
                current_ppas,
            )
        )

    def delete_tensor(
        self,
        file_id,
    ):

        if file_id not in self.tensor_table:
            return

        meta = self.tensor_table[file_id]

        # Invalidate every LPN belonging to this tensor.
        for lba in meta["lba_list"]:

            lpn = (
                int(lba)
                // self.nsec
            )

            self.invalidate_lpn(lpn)

        del self.tensor_table[file_id]

        print(
            f"[FTL] Unmapped tensor '{file_id}'"
        )

    def _convert_ppas_to_tensor(
        self,
        ppa_array,
    ):
        if isinstance(
            ppa_array,
            torch.Tensor,
        ):
            return ppa_array

        if isinstance(
            ppa_array,
            list,
        ):
            ppa_array = np.array(
                ppa_array,
                dtype=np.uint64,
            )

        # Preserve the exact 64-bit bit pattern.
        np_ppas = ppa_array.view(
            np.int64
        )

        return torch.from_numpy(
            np_ppas
        )

    def device_write(
        self,
        meta,
        tensor,
        file_id,
    ):
        assert not tensor.is_cuda, (
            "device write uses cpu tensor"
        )

        assert tensor.is_contiguous(), (
            "device write requires contiguous tensor"
        )

        # Use metadata's file_id when possible so that
        # the mapping can be refreshed after GC.
        if file_id in self.tensor_table:

            ppa_array = (
                self._get_tensor_current_ppas(
                    file_id
                )
            )

        else:
            ppa_array = (
                meta["ppa_uint64_list"]
            )

        ppa_tensor = (
            self._convert_ppas_to_tensor(
                ppa_array
            )
        )

        self._ops.device_write(
            self.cuFD,
            tensor,
            ppa_tensor,
            self.ssd_data[-1],
        )

    def verify_write(
        self,
        meta,
        tensor,
        file_id,
    ):
        assert not tensor.is_cuda, (
            "device_read_and_verify requires "
            "a CUDA tensor"
        )

        assert tensor.is_contiguous(), (
            "device_read_and_verify requires "
            "contiguous tensor"
        )

        if file_id in self.tensor_table:

            ppa_array = (
                self._get_tensor_current_ppas(
                    file_id
                )
            )

        else:
            ppa_array = (
                meta["ppa_uint64_list"]
            )

        ppa_tensor = (
            self._convert_ppas_to_tensor(
                ppa_array
            )
        )

        is_correct = (
            self._ops.device_read_and_verify(
                self.cuFD,
                tensor,
                ppa_tensor,
                self.ssd_data[-1],
            )
        )

        return is_correct

    def stats(self):

        mapped_pages = len(
            self.l2p
        )

        valid_pages = sum(
            1
            for value in self.valid.values()
            if value
        )

        invalid_pages = sum(
            1
            for value in self.valid.values()
            if not value
        )

        free_pages = (
            self.total_pages
            - self.next_page
        )

        return {
            "total_pages": self.total_pages,

            "total_lpn": self.total_lpn,

            "total_lba": self.total_lba,

            "mapped_pages": mapped_pages,

            "valid_pages": valid_pages,

            "invalid_pages": invalid_pages,

            "free_pages": free_pages,

            "next_page": self.next_page,

            "num_tensors": len(
                self.tensor_table
            ),
        }

    def get_flash_layout(
        self,
        device,
    ):

        return {
            "ch": self.nch,
            "lun": self.nlun,
            "plane": self.npl,
            "block": self.nblk,
            "page": self.npg,
            "sector": self.nsec,
            "sector_size": self.sector_size,
        }

    def open_device(self, device="/dev/nvme0n1"):

        return os.open(
            device,
            os.O_RDWR,
        )

    def close_device(self):

        if (
            hasattr(self, "device_fd")
            and self.device_fd is not None
        ):
            os.close(
                self.device_fd
            )

            self.device_fd = None
    