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
        self.rsv = data[6] if len(data) > 6 else 0

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


    def get_page_index(self):

        x = self.blk
        x = x * self.npg  + self.pg
        x = x * self.npl  + self.pl
        x = x * self.nlun + self.lun
        x = x * self.nch  + self.ch

        if not (0 <= self.ch < self.nch and 0 <= self.lun < self.nlun and 0 <= self.pl < self.npl
                and 0 <= self.blk < self.nblk and 0 <= self.pg < self.npg and 0 <= self.sec < self.nsec):
            raise ValueError(f"PPA field out of range: {self}")
        
        return x


    def get_index(self):
        return self.get_page_index() * self.nsec + self.sec

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
    def __init__(self, device="/dev/nvme0n1", row_size=4096):

        '''
            1 sector = 512 B
            32 sectors
            NAND page = 16 KiB

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

        self.row_size = row_size
        self.rows_per_page = self.page_size // self.row_size

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

        # Page mapping as flat arrays (a few bytes per page) instead of dicts
        # of PPA objects (~1.3 KB per page). PPA objects are built on demand
        # (get_ppa etc.); the I/O paths only need page indices.
        #   l2p[lpn]          physical page index, -1 = unmapped
        #   p2l[page_index]   LPN, -1 = none
        #   valid[page_index] page holds live data
        idx_dtype = np.int32 if self.total_pages < 2**31 else np.int64
        self.l2p = np.full(self.total_lpn, -1, dtype=idx_dtype)
        self.p2l = np.full(self.total_pages, -1, dtype=idx_dtype)
        self.valid = np.zeros(self.total_pages, dtype=np.bool_)
        self.next_page = 0
        self._check_ppa_packing()

        self.tensor_table = {}
        self.next_free_lba = 0

        # Bumped on every L2P change; cached per-tensor device LBA arrays
        # (get_tensor_dev_lbas) are rebuilt when it moved.
        self._map_version = 0

        self._ops = _load_ops()
        self.device_fd = self.open_device(device)
        self.cuFD = self._ops.register_fd(self.device_fd) if self._ops is not None else None

    def get_flash_layout(self, device):
        return {
            "ch" : 8,
            "lun" : 8,
            "plane" : 1,
            "block" : 1024,
            "page" : 512,
            "sector" : 32,
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

    # ------------------------------------------------------------------
    # Array helpers (vectorized mapping, no per-page Python objects)
    # ------------------------------------------------------------------

    def _pack_ppa(self, page_idx, sec=0):
        """FEMU 64-bit PPAs of physical page indices (+ sector), vectorized.
        Same bits as PPA.to_uint64(); checked by _check_ppa_packing()."""
        x = np.asarray(page_idx, dtype=np.uint64).copy()
        u = np.uint64
        ch = x % u(self.nch);  x //= u(self.nch)
        lun = x % u(self.nlun); x //= u(self.nlun)
        pl = x % u(self.npl);  x //= u(self.npl)
        pg = x % u(self.npg);  x //= u(self.npg)
        blk = x
        sec = np.asarray(sec, dtype=np.uint64)
        # Bitfield order of PPA_Bitfield, from the least significant bit.
        o_pg = BLK_BITS
        o_sec = o_pg + PG_BITS
        o_pl = o_sec + SEC_BITS
        o_lun = o_pl + PL_BITS
        o_ch = o_lun + LUN_BITS
        return (
            blk
            | (pg << u(o_pg))
            | (sec << u(o_sec))
            | (pl << u(o_pl))
            | (lun << u(o_lun))
            | (ch << u(o_ch))
        )

    def _check_ppa_packing(self):
        samples = sorted({0, 1, self.nch, self.total_pages // 3, self.total_pages - 1})
        fast = self._pack_ppa(np.array(samples))
        for i, idx in enumerate(samples):
            ref = self.page_index_to_ppa(idx).to_uint64()
            if int(fast[i]) != int(ref):
                raise RuntimeError(
                    f"FTL PPA packing mismatch at page {idx}: {int(fast[i]):#x} != {int(ref):#x}"
                )

    def _lpns_of(self, meta):
        return (np.asarray(meta["lba_list"], dtype=np.int64) // self.nsec)

    def _write_lpns(self, lpns):
        """Map each LPN to a fresh physical page (out-of-place, like
        write_lpn) and invalidate its old page. Returns the page indices."""
        lpns = np.asarray(lpns, dtype=np.int64)
        n = lpns.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int64)
        if lpns.min() < 0 or lpns.max() >= self.total_lpn:
            raise ValueError(f"Invalid LPN in range {lpns.min()}..{lpns.max()}")
        if self.next_page + n > self.total_pages:
            raise RuntimeError(
                "FTL has no free physical pages. "
                "GC is not implemented yet."
            )
        new = np.arange(self.next_page, self.next_page + n, dtype=np.int64)
        self.next_page += n

        old = self.l2p[lpns]
        old = old[old >= 0]
        if old.size:
            self.valid[old] = False
            self.p2l[old] = -1
        self.l2p[lpns] = new
        self.p2l[new] = lpns
        self.valid[new] = True
        self._map_version += 1
        return new

    def _invalidate_lpns(self, lpns):
        lpns = np.asarray(lpns, dtype=np.int64)
        old = self.l2p[lpns]
        old = old[old >= 0]
        if old.size:
            self.valid[old] = False
            self.p2l[old] = -1
        self.l2p[lpns] = -1
        self._map_version += 1

    def _mapped_pages(self, lpns, what):
        """Physical page indices of LPNs; error if any is unmapped."""
        idx = self.l2p[np.asarray(lpns, dtype=np.int64)].astype(np.int64)
        if idx.size and idx.min() < 0:
            bad = int(np.asarray(lpns)[np.argmin(idx)])
            raise RuntimeError(f"LPN {bad} of {what} has no physical mapping")
        return idx

    # ------------------------------------------------------------------

    def get_ppa(self, lpn):
        if lpn < 0 or lpn >= self.total_lpn:
            raise ValueError(
                f"Invalid LPN {lpn}"
            )

        idx = int(self.l2p[lpn])
        return None if idx < 0 else self.page_index_to_ppa(idx)

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

        x = page_index
        ch  = x % self.nch;  x //= self.nch
        lun = x % self.nlun; x //= self.nlun
        pl  = x % self.npl;  x //= self.npl
        pg  = x % self.npg;  x //= self.npg
        blk = x

        return PPA([ch, lun, pl, blk, pg, 0, 0], self.ssd_data)

    def allocate_page(self):
        if self.next_page >= self.total_pages:
            raise RuntimeError(
                "FTL has no free physical pages. "
                "GC is not implemented yet."
            )

        page_index = self.next_page

        self.next_page += 1

        self.valid[page_index] = True

        return self.page_index_to_ppa(page_index)

    def write_lpn(self, lpn):
        if lpn < 0 or lpn >= self.total_lpn:
            raise ValueError(
                f"Invalid LPN {lpn}")

        new_idx = self._write_lpns([lpn])[0]
        return self.page_index_to_ppa(int(new_idx))

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

        existing_lpn = int(self.p2l[page_index])

        if (
            existing_lpn >= 0
            and existing_lpn != lpn
        ):
            raise ValueError(
                f"Physical page {page_index} is already "
                f"mapped to LPN {existing_lpn}"
            )

        self._map_version += 1
        old_index = int(self.l2p[lpn])

        if old_index >= 0:
            self.valid[old_index] = False
            self.p2l[old_index] = -1

        self.l2p[lpn] = page_index
        self.p2l[page_index] = lpn
        self.valid[page_index] = True

    def invalidate_lpn(self, lpn):

        self._map_version += 1
        old_index = int(self.l2p[lpn])

        if old_index < 0:
            return None

        self.l2p[lpn] = -1
        self.valid[old_index] = False
        self.p2l[old_index] = -1

        return self.page_index_to_ppa(old_index)

    def get_lpn(self, ppa):

        page_index = (
            ppa.get_page_index()
        )

        lpn = int(self.p2l[page_index])
        return None if lpn < 0 else lpn

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
        lbas = np.asarray(lba_array, dtype=np.int64)
        lpns = lbas // self.nsec
        idx = self.l2p[lpns].astype(np.int64)
        if idx.size and idx.min() < 0:
            raise KeyError(
                f"LPN {int(lpns[np.argmin(idx)])} has no physical mapping"
            )
        return self._pack_ppa(idx, sec=lbas % self.nsec)

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
        total_bytes = total_elements * element_size

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
            
            # move LBA to start of next free page
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

        # One fresh physical page per LPN (same allocation order as calling
        # write_lpn page by page). Device LBA of a page = page_index * nsec.
        lpns = page_start_lbas.astype(np.int64) // sectors_per_page
        page_idx = self._write_lpns(lpns)
        dev_lbas = page_idx.astype(np.uint64) * np.uint64(sectors_per_page)
        mapped_ppas = self._pack_ppa(page_idx)

        self.tensor_table[file_id] = {
            "file_id": file_id,
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "num_elements": total_elements,
            "element_size": element_size,
            "total_bytes": total_bytes,
            "physical_bytes": pages_needed * bytes_per_page,
            "page_bytes": bytes_per_page,          
            "num_pages": pages_needed,             
            "num_sectors": sectors_needed,         
            "start_lba": start_lba,                
            "end_lba": end_lba,
            "lba_list": page_start_lbas,           
            "dev_lba_list": dev_lbas,              
            "io_runs": self._build_io_runs(dev_lbas),
            "ppa_uint64_list": mapped_ppas,        
        }

        self.next_free_lba = end_lba + 1

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
        idx = self._mapped_pages(self._lpns_of(meta), f"'{file_id}'")
        return self._pack_ppa(idx)

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
        row_byte_offset = row_idx * self.row_size
        page_idx = row_byte_offset // self.page_size  # flash page of row tensor

        # offset inside page converted to logical index : (0 = 0, 1 = 4096/row_size, 2 = 8192,...)
        offset = (row_byte_offset % self.page_size) // self.row_size  

        start_lpn = meta["start_lba"] // self.nsec
        lpn = start_lpn + page_idx
        mapping = self.get_mapping(lpn)
        _, ppa, _ = mapping

        ppa_uint64 = np.array([ppa.to_uint64()], dtype=np.uint64)


        return {
            "row_idx": row_idx,
            "lpn": lpn,
            "ppa": ppa.to_uint64(),
            "ppa_tensor": self._convert_ppas_to_tensor(ppa_uint64),
            "offset": offset,
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
        self._invalidate_lpns(self._lpns_of(meta))

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

        assert tensor.is_contiguous(), ("device write requires contiguous tensor")

        assert not tensor.is_cuda and tensor.is_contiguous()
        runs = self.get_tensor_io_runs(file_id)           
        runs_t = torch.tensor(runs, dtype=torch.int64)      
        self._ops.device_write(self.device_fd, tensor, runs_t, self.sector_size, self.page_size)

        

    def gpu_write(
        self,
        meta,
        tensor,
        file_id,
        async_=False,
    ):

        assert tensor.is_cuda and tensor.is_contiguous(), (
            "gpu_write requires a contiguous CUDA tensor"
        )

        # Same LBA runs as device_write; read from the current mapping.
        runs = self.get_tensor_io_runs(file_id)
        runs_t = torch.tensor(runs, dtype=torch.int64).reshape(-1, 3)
        if os.environ.get("GRINNDER_GDS_DIRECT") == "1":
            # cuFile batch writes (real GDS path)
            job = self._ops.gpu_write_async(
                self.cuFD, tensor, runs_t, self.sector_size, self.page_size
            )
        else:
            # Copies to pinned host memory + NVMe writes, no cuFile
            job = self._ops.gpu_write_nvme_async(
                self.device_fd, tensor, runs_t, self.sector_size, self.page_size
            )
        if async_:
            # Finish it with gpu_write_wait(id).
            return job
        self._ops.gpu_write_wait(job)
        return None

    def gpu_write_wait(self, job_id):
        self._ops.gpu_write_wait(job_id)

    def gpu_write_wait_all(self):
        self._ops.gpu_write_wait_all()


    def verify_write(
        self,
        meta,
        tensor,
        file_id,
    ):
        assert not tensor.is_cuda and tensor.is_contiguous(), (
            "device_read_and_verify requires a contiguous CPU tensor"
        )

        runs = self.get_tensor_io_runs(file_id)
        runs_t = torch.tensor(runs, dtype=torch.int64)
        return self._ops.device_read_and_verify(
            self.device_fd, tensor, runs_t, self.sector_size, self.page_size
        )

    def stats(self):

        mapped_pages = int(np.count_nonzero(self.l2p >= 0))

        valid_pages = int(np.count_nonzero(self.valid))

        # Pages handed out (below next_page) whose data was overwritten or
        # deleted; GC would reclaim these.
        invalid_pages = int(np.count_nonzero(~self.valid[: self.next_page]))

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
    
    def _build_io_runs(self, dev_lbas: np.ndarray):
        # merge contiguous pages
        n = len(dev_lbas)
        if n == 0:
            return []
        
        breaks = np.flatnonzero(np.diff(dev_lbas.astype(np.int64)) != self.nsec) + 1
        run_starts = np.concatenate(([0], breaks))
        run_lens = np.diff(np.concatenate((run_starts, [n])))

        return [
            (int(dev_lbas[s]) * self.sector_size,  
            int(s) * self.page_size,              
            int(k) * self.page_size)              
            for s, k in zip(run_starts, run_lens)
        ]
    

    def get_row_file_offsets(self, file_id, rows, row_bytes):
        """Device byte offsets (cuFile file_offset) for the given row indices"""
        if self.page_size % row_bytes != 0:
            raise ValueError(f"row_bytes={row_bytes} must divide page size {self.page_size}")
        if row_bytes % self.sector_size != 0:
            raise ValueError(f"row_bytes={row_bytes} must be a multiple of {self.sector_size}")

        rows = np.asarray(rows, dtype=np.int64)
        start = rows * row_bytes
        page_idx = start // self.page_size
        in_page = start % self.page_size

        dev_lbas = self.get_tensor_dev_lbas(file_id).astype(np.int64)
        return dev_lbas[page_idx] * self.sector_size + in_page

    

    def get_tensor_dev_lbas(self, file_id):
        """Device LBA of each page of a tensor, from the current mapping.

        Cached per tensor: recomputed only after the L2P mapping changed
        (_map_version).
        """
        meta = self.tensor_table[file_id]
        cached = meta.get("_dev_lba_cache")
        if cached is not None and cached[0] == self._map_version:
            return cached[1]
        out = self._walk_tensor_dev_lbas(file_id, meta)
        out.setflags(write=False)   # shared by all callers
        meta["_dev_lba_cache"] = (self._map_version, out)
        return out

    def _walk_tensor_dev_lbas(self, file_id, meta):
        # Device LBA of a page = physical page index * sectors per page
        # (PPA.get_index() with sec = 0).
        idx = self._mapped_pages(self._lpns_of(meta), f"'{file_id}'")
        return idx.astype(np.uint64) * np.uint64(self.nsec)

    def get_tensor_io_runs(self, file_id):
        return self._build_io_runs(self.get_tensor_dev_lbas(file_id))

    

    def open_device(self, device="/dev/nvme0n1"):
        return os.open(device, os.O_RDWR | os.O_DIRECT)

    def close_device(self):

        if (
            hasattr(self, "device_fd")
            and self.device_fd is not None
        ):
            os.close(
                self.device_fd
            )

            self.device_fd = None
    