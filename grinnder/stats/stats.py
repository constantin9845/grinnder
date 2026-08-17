import time
from typing import List
import json
from itertools import zip_longest

class Stat:
    def __init__(self):
        self.running: bool = False
        self.start_time = 0

        # forward
        self.forward_time = 0
        self.forward_GPU_start = 0
        self.forward_compute_start = 0
        self.forward_partition_load_CPU_timesteps: List[float] = []

        self.forward_partition_load_GPU_timesteps_gather: List[float] = []
        self.forward_partition_load_GPU_timesteps_copy: List[float] = []

        self.forward_compute_timesteps: List[float] = []

        self.forward_gds_writes: List[float] = []

        # loss
        self.loss_time = 0
        self.loss_GPU_start = 0
        self.loss_start = 0
        self.loss_partition_load_CPU_timesteps: List[float] = []
        self.loss_partition_load_GPU_timesteps_gather: List[float] = []
        self.loss_partition_load_GPU_timesteps_copy: List[float] = []

        # backward
        self.backward_time = 0
        self.backward_start = 0
        self.backward_GPU_start = 0
        self.backward_compute_start = 0
        self.backward_partition_load_CPU_timesteps: List[float] = []
        self.backward_gradient_load_CPU_timesteps: List[float] = []

        self.backward_partition_load_GPU_timesteps_gather: List[float] = []
        self.backward_partition_load_GPU_timesteps_copy: List[float] = []

        self.backward_gradient_load_GPU_timesteps_gather: List[float] = []
        self.backward_gradient_load_GPU_timesteps_copy: List[float] = []

        self.backward_direct_load_timesteps: List[float] = []
        self.backward_compute_timesteps: List[float] = []

        self.backward_gradient_to_CPU_writes: List[float] = []
        self.backward_gradient_to_SSD_writes: List[float] = []

        self.weights_time = 0


        # determine how much is wasted when loading full partitions
        self.partition_utilization: List[float] = [] 

        # determine at gather step
        # compare across different numbers of partitions
        self.actual_partition_size: List[float] = [] 


    def started(self):
        return self.running == True
    
    def start(self):
        if self.running:
            return
        
        self.running = True
        self.start_time = time.perf_counter_ns()

    def start_loss(self):
        self.loss_start = time.perf_counter_ns()

    def start_backward(self):
        self.backward_start = time.perf_counter_ns()

    def load_CPU_timestamp(self, stage, start, end):

        start = (start - self.start_time) / 1000000000.0
        end = (end - self.start_time) / 1000000000.0

        start = round(start,4)
        end = round(end,4)
        if stage == "forward":
            self.forward_partition_load_CPU_timesteps.extend([start,end])
        elif stage == "loss":
            self.loss_partition_load_CPU_timesteps.extend([start,end])
        elif stage == "gradient":
            self.backward_gradient_load_CPU_timesteps.extend([start,end])
        else:
            self.backward_partition_load_CPU_timesteps.extend([start,end])

    # track: stage
    #   - Time to gather boundary nodes from host buffer
    #   - Time to copy tensors to GPU
    def load_GPU_timestamp(self, phase, stage, start, end):

        start = (start - self.start_time) / 1000000000.0
        end = (end - self.start_time) / 1000000000.0

        start = round(start,4)
        end = round(end,4)


        if phase == "forward":
            if stage == "gather":
                self.forward_partition_load_GPU_timesteps_gather.extend(
                    [start, end]
                )
            else:
                self.forward_partition_load_GPU_timesteps_copy.extend(
                    [start, end]
                )

        elif phase == "loss":
            if stage == "gather":
                self.loss_partition_load_GPU_timesteps_gather.extend(
                    [start, end]
                )
            else:
                self.loss_partition_load_GPU_timesteps_copy.extend(
                    [start, end]
                )

        elif phase == "gradient":
            if stage == "gather":
                self.backward_gradient_load_GPU_timesteps_gather.extend(
                    [start, end]
                )
            else:
                self.backward_gradient_load_GPU_timesteps_copy.extend(
                    [start, end]
                )

        else:
            if stage == "gather":
                self.backward_partition_load_GPU_timesteps_gather.extend(
                    [start, end]
                )
            else:
                self.backward_partition_load_GPU_timesteps_copy.extend(
                    [start, end]
                )

    def load_GDS_timestamp(self):
        t = time.perf_counter_ns()
        t = t - self.start_time
        t = t / 1000000000.0
        t = round(t,2)
        self.backward_direct_load_timesteps.append(time.perf_counter_ns())

    def begin_compute(self, stage):
        t = time.perf_counter_ns()

        if stage == "foward":
            if self.forward_compute_start == 0:
                t = t - self.start_time
                t = t / 1000000000.0
                t = round(t,2)
                self.forward_compute_start = t

        else:
            if self.backward_compute_start == 0:
                t = t - self.start_time
                t = t / 1000000000.0
                t = round(t,2)
                self.backward_compute_start = t

    def compute_timestamp(self, stage, start, end):
        start = start - self.start_time
        end = end - self.start_time

        start = start / 1000000000.0
        end = end / 1000000000.0

        start = round(start,4)
        end = round(end,4)

        if stage == "forward":
            self.forward_compute_timesteps.extend([start,end])
        else:
            self.backward_compute_timesteps.extend([start,end])

    def write_timestamp(self, phase, stage, start, end):
        start = start - self.start_time
        end = end - self.start_time

        start = start / 1000000000.0
        end = end / 1000000000.0

        start = round(start,4)
        end = round(end,4)

        if phase == "forward":
            self.forward_gds_writes.append(start)
            self.forward_gds_writes.append(end)
        else:
            if stage == "CPU":
                self.backward_gradient_to_CPU_writes.append(start)
                self.backward_gradient_to_CPU_writes.append(end)
            else:
                self.backward_gradient_to_SSD_writes.append(start)
                self.backward_gradient_to_SSD_writes.append(end)

    def forward_done(self):
        self.forward_time = time.perf_counter_ns()

    def loss_done(self):
        self.loss_time = time.perf_counter_ns()

    def backward_done(self):
        self.backward_time = time.perf_counter_ns()

    def weights_done(self):
        self.weights_time = time.perf_counter_ns()

    def reset(self):
        self.running = False
        self.start_time = 0

        # forward
        self.forward_time = 0
        self.forward_GPU_start = 0
        self.forward_compute_start = 0
        self.forward_partition_load_CPU_timesteps.clear()
        self.forward_partition_load_GPU_timesteps_gather.clear()
        self.forward_partition_load_GPU_timesteps_copy.clear()
        self.forward_compute_timesteps.clear()
        self.forward_gds_writes.clear()

        # loss
        self.loss_time = 0
        self.loss_GPU_start = 0
        self.loss_start = 0
        self.loss_partition_load_CPU_timesteps.clear()
        self.loss_partition_load_GPU_timesteps_gather.clear()
        self.loss_partition_load_GPU_timesteps_copy.clear()

        # backward
        self.backward_time = 0
        self.backward_start = 0
        self.backward_GPU_start = 0
        self.backward_compute_start = 0
        self.backward_partition_load_CPU_timesteps.clear()
        self.backward_gradient_load_CPU_timesteps.clear()
        self.backward_partition_load_GPU_timesteps_gather.clear()
        self.backward_partition_load_GPU_timesteps_copy.clear()
        self.backward_gradient_load_GPU_timesteps_gather.clear()
        self.backward_gradient_load_GPU_timesteps_copy.clear()
        self.backward_direct_load_timesteps.clear()
        self.backward_compute_timesteps.clear()
        self.backward_gradient_to_CPU_writes.clear()
        self.backward_gradient_to_SSD_writes.clear()

        self.weights_time = 0

    def add_actual_size(self, part_size, working_size):
        self.actual_partition_size.extend([part_size, working_size])

    def add_boundary_utilization(self, percentage):
        self.partition_utilization.append(percentage)

    @staticmethod
    def _format_pairs(flat_list: List[float]) -> List[str]:
        """Converts flat list [s1, e1, s2, e2] -> ['(s1 - e1)', '(s2 - e2)']"""
        if not flat_list:
            return []
        pairs = []
        for i in range(0, len(flat_list), 2):
            if i + 1 < len(flat_list):
                pairs.append(f"({flat_list[i]} - {flat_list[i+1]})")
            else:
                pairs.append(f"({flat_list[i]} - ?)")
        return pairs

    
    def print_timeline(self):
        print("========== TIMELINE ==========")
        print(f"Start = {self.start_time}\n")

        W = 20
        FILL = "-"  # Placeholder when lists have unequal lengths

        # --- FORWARD STEP ---
        print("[FORWARD STEP]")
        header_fwd = (
            f"\t| {'SD --> CPU':<{W}} "
            f"| {'CPU->GPU (Gather)':<{W}} "
            f"| {'CPU->GPU (Copy)':<{W}} "
            f"| {'Compute':<{W}} |"
        )
        divider_fwd = f"\t+{'-' * (W + 2)}" * 4 + "+"
        print(header_fwd)
        print(divider_fwd)

        cnt = 1
        fwd_gpu_gather = self._format_pairs(
            self.forward_partition_load_GPU_timesteps_gather
        )
        fwd_gpu_copy = self._format_pairs(
            self.forward_partition_load_GPU_timesteps_copy
        )
        fwd_cmp = self._format_pairs(self.forward_compute_timesteps)

        for i, j, k, l in zip_longest(
            self.forward_partition_load_CPU_timesteps,
            fwd_gpu_gather,
            fwd_gpu_copy,
            fwd_cmp,
            fillvalue=FILL,
        ):
            print(
                f"{cnt}\t| {str(i):<{W}} | {str(j):<{W}} | {str(k):<{W}} | {str(l):<{W}} |"
            )
            cnt += 1

        # --- LOSS STEP ---
        cnt = 1
        print("\n\n[LOSS STEP]")
        header_loss = (
            f"\t| {'SD --> CPU':<{W}} "
            f"| {'CPU->GPU (Gather)':<{W}} "
            f"| {'CPU->GPU (Copy)':<{W}} |"
        )
        divider_loss = f"\t+{'-' * (W + 2)}" * 3 + "+"
        print(header_loss)
        print(divider_loss)

        loss_gpu_gather = self._format_pairs(
            self.loss_partition_load_GPU_timesteps_gather
        )
        loss_gpu_copy = self._format_pairs(
            self.loss_partition_load_GPU_timesteps_copy
        )

        for i, j, k in zip_longest(
            self.loss_partition_load_CPU_timesteps,
            loss_gpu_gather,
            loss_gpu_copy,
            fillvalue=FILL,
        ):
            print(f"{cnt}\t| {str(i):<{W}} | {str(j):<{W}} | {str(k):<{W}} |")
            cnt += 1

        # --- BACKWARD STEP ---
        cnt = 1
        print("\n\n[BACKWARD STEP]")
        header_bwd = (
            f"\t| {'SD->CPU (Parts)':<{W}} | {'SD->CPU (Grad)':<{W}} "
            f"| {'GPU Parts (Gather)':<{W}} | {'GPU Parts (Copy)':<{W}} "
            f"| {'GPU Grad (Gather)':<{W}} | {'GPU Grad (Copy)':<{W}} "
            f"| {'Compute':<{W}} | {'GDS':<{W}} |"
        )
        divider_bwd = f"\t+{'-' * (W + 2)}" * 8 + "+"
        print(header_bwd)
        print(divider_bwd)

        bwd_parts_gpu_gather = self._format_pairs(
            self.backward_partition_load_GPU_timesteps_gather
        )
        bwd_parts_gpu_copy = self._format_pairs(
            self.backward_partition_load_GPU_timesteps_copy
        )
        bwd_grad_gpu_gather = self._format_pairs(
            self.backward_gradient_load_GPU_timesteps_gather
        )
        bwd_grad_gpu_copy = self._format_pairs(
            self.backward_gradient_load_GPU_timesteps_copy
        )
        bwd_cmp = self._format_pairs(self.backward_compute_timesteps)

        for i, j, k, l, m, n, o, p in zip_longest(
            self.backward_partition_load_CPU_timesteps,
            self.backward_gradient_load_CPU_timesteps,
            bwd_parts_gpu_gather,
            bwd_parts_gpu_copy,
            bwd_grad_gpu_gather,
            bwd_grad_gpu_copy,
            bwd_cmp,
            self.backward_direct_load_timesteps,
            fillvalue=FILL,
        ):
            print(
                f"{cnt}\t| {str(i):<{W}} | {str(j):<{W}} | {str(k):<{W}} | "
                f"{str(l):<{W}} | {str(m):<{W}} | {str(n):<{W}} | "
                f"{str(o):<{W}} | {str(p):<{W}} |"
            )
            cnt += 1

        # --- SUMMARY ---
        print(
            f"\n\n\tStart = {self.start_time} "
            f"\n\tforward done = {self.forward_time} "
            f"\n\tloss done = {self.loss_time} "
            f"\n\tbackward done = {self.backward_time} "
            f"\n\tweights done = {self.weights_time}"
        )

        print("\n\n")
        print(f"Boundary partition utilization = {self.partition_utilization}\n")
        print(f"Working partition size = {self.actual_partition_size}\n")
        print("==============================")

        attrs = [
            (
                "forward_partition_load_CPU_timesteps",
                self.forward_partition_load_CPU_timesteps,
            ),
            (
                "forward_partition_load_GPU_timesteps_gather",
                self.forward_partition_load_GPU_timesteps_gather,
            ),
            (
                "forward_partition_load_GPU_timesteps_copy",
                self.forward_partition_load_GPU_timesteps_copy,
            ),
            ("forward_compute_timesteps", self.forward_compute_timesteps),
            ("forward_gds_writes", self.forward_gds_writes),
            (
                "loss_partition_load_CPU_timesteps",
                self.loss_partition_load_CPU_timesteps,
            ),
            (
                "loss_partition_load_GPU_timesteps_gather",
                self.loss_partition_load_GPU_timesteps_gather,
            ),
            (
                "loss_partition_load_GPU_timesteps_copy",
                self.loss_partition_load_GPU_timesteps_copy,
            ),
            (
                "backward_partition_load_CPU_timesteps",
                self.backward_partition_load_CPU_timesteps,
            ),
            (
                "backward_gradient_load_CPU_timesteps",
                self.backward_gradient_load_CPU_timesteps,
            ),
            (
                "backward_partition_load_GPU_timesteps_gather",
                self.backward_partition_load_GPU_timesteps_gather,
            ),
            (
                "backward_partition_load_GPU_timesteps_copy",
                self.backward_partition_load_GPU_timesteps_copy,
            ),
            (
                "backward_gradient_load_GPU_timesteps_gather",
                self.backward_gradient_load_GPU_timesteps_gather,
            ),
            (
                "backward_gradient_load_GPU_timesteps_copy",
                self.backward_gradient_load_GPU_timesteps_copy,
            ),
            ("backward_compute_timesteps", self.backward_compute_timesteps),
            (
                "backward_direct_load_timesteps",
                self.backward_direct_load_timesteps,
            ),
            (
                "backward_gradient_to_CPU_writes",
                self.backward_gradient_to_CPU_writes,
            ),
            (
                "backward_gradient_to_SSD_writes",
                self.backward_gradient_to_SSD_writes,
            ),
        ]

        with open("timesteps_output.py", "w") as f:
            f.write("# --- READY TO COPY PASTE ---\n")
            for name, data in attrs:
                clean_vals = (
                    [x.item() if hasattr(x, "item") else x for x in list(data)]
                    if data
                    else []
                )
                f.write(f"{name} = [\n")
                for item in clean_vals:
                    f.write(f"    {item},\n")
                f.write("]\n\n")

        print("Successfully written to timesteps_output.py!")
    
    
    def forward_timeline(self, data):
        """Expects data = [ssd_cpu, cpu_gpu_gather, compute, gds_writes]"""
        import matplotlib.pyplot as plt

        ssd_cpu = data[0] if len(data) > 0 and data[0] else []
        cpu_gpu = data[1] if len(data) > 1 and data[1] else []
        compute = data[2] if len(data) > 2 and data[2] else []
        gds_writes = data[3] if len(data) > 3 and data[3] else []

        def extract_blocks(flat_pairs):
            starts, durations = [], []
            for i in range(0, len(flat_pairs) - 1, 2):
                start = flat_pairs[i]
                end = flat_pairs[i + 1]
                starts.append(start)
                durations.append(max(0.0, end - start))
            return starts, durations

        cg_starts, cg_durations = extract_blocks(cpu_gpu)
        cmp_starts, cmp_durations = extract_blocks(compute)
        write_starts, write_durations = extract_blocks(gds_writes)

        fig, ax = plt.subplots(figsize=(16, 6))

        tracks = [
            ("CPU --> GPU (Gather)", 2, cg_starts, cg_durations, "#F5A623"),
            ("Compute", 1, cmp_starts, cmp_durations, "#7ED321"),
            ("GDS Writes", 0, write_starts, write_durations, "#4A90E2"),
        ]

        for label, y_pos, starts, durations, single_color in tracks:
            for start, duration in zip(starts, durations):
                if duration > 0:
                    ax.barh(
                        y=y_pos,
                        width=duration,
                        left=start,
                        height=0.55,
                        color=single_color,
                        edgecolor="black",
                        linewidth=0.3,
                    )

        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(
            ["CPU --> GPU\n(Gathering)", "Compute", "GDS Writes"],
            fontsize=12,
            fontweight="bold",
        )

        ax.set_xlabel("Time (s)", fontsize=16, fontweight="bold")
        ax.set_title(
            "Forward Pass Timeline", fontsize=18, fontweight="bold", pad=15
        )
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        all_ends = [
            flat_list[i]
            for flat_list in [cpu_gpu, compute, gds_writes]
            for i in range(1, len(flat_list), 2)
        ]
        max_x = max(all_ends) if all_ends else 10
        ax.set_xlim(0, max_x + 1)

        plt.tight_layout()
        plt.show()

    def backward_timeline(self, data):
        """Expects data = [ssd_parts, ssd_grad, gpu_parts, gpu_grad, compute, gds, cpu_writes, ssd_writes]"""
        import matplotlib.pyplot as plt

        ssd_parts = data[0] if len(data) > 0 and data[0] else []
        ssd_grad = data[1] if len(data) > 1 and data[1] else []
        gpu_parts = data[2] if len(data) > 2 and data[2] else []
        gpu_grad = data[3] if len(data) > 3 and data[3] else []
        compute = data[4] if len(data) > 4 and data[4] else []
        gds = data[5] if len(data) > 5 and data[5] else []
        cpu_writes = data[6] if len(data) > 6 and data[6] else []
        ssd_writes = data[7] if len(data) > 7 and data[7] else []

        def extract_blocks(flat_pairs):
            starts, durations = [], []
            for i in range(0, len(flat_pairs) - 1, 2):
                start = flat_pairs[i]
                end = flat_pairs[i + 1]
                starts.append(start)
                durations.append(max(0.0, end - start))
            return starts, durations

        s2_starts, s2_durations = extract_blocks(gpu_parts)
        s3_starts, s3_durations = extract_blocks(gpu_grad)
        s4_starts, s4_durations = extract_blocks(compute)
        cw_starts, cw_durations = extract_blocks(cpu_writes)
        sw_starts, sw_durations = extract_blocks(ssd_writes)

        fig, ax = plt.subplots(figsize=(16, 8))

        tracks = [
            ("CPU --> GPU (Parts)", 4, s2_starts, s2_durations, "#F5A623"),
            ("CPU --> GPU (Grad)", 3, s3_starts, s3_durations, "#9013FE"),
            ("Backward Compute", 2, s4_starts, s4_durations, "#7ED321"),
            ("Grad --> CPU Writes", 1, cw_starts, cw_durations, "#4A90E2"),
            ("Grad --> SSD Writes", 0, sw_starts, sw_durations, "#D0021B"),
        ]

        for label, y_pos, starts, durations, single_color in tracks:
            for start, duration in zip(starts, durations):
                if duration > 0:
                    ax.barh(
                        y=y_pos,
                        width=duration,
                        left=start,
                        height=0.55,
                        color=single_color,
                        edgecolor="black",
                        linewidth=0.3,
                    )

        ax.set_yticks([4, 3, 2, 1, 0])
        ax.set_yticklabels(
            [
                "CPU --> GPU\n(Gathering partitions)",
                "CPU --> GPU\n(Gathering gradients)",
                "Backward Compute",
                "Gradient --> CPU Writes",
                "Gradient --> SSD Writes",
            ],
            fontsize=11,
            fontweight="bold",
        )

        ax.set_xlabel("Time (s)", fontsize=16, fontweight="bold")
        ax.set_title(
            "Backward Pass Timeline", fontsize=18, fontweight="bold", pad=15
        )
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        all_lists = [gpu_parts, gpu_grad, compute, cpu_writes, ssd_writes]
        all_ends = [
            flat_list[i]
            for flat_list in all_lists
            for i in range(1, len(flat_list), 2)
        ]
        all_starts = [
            flat_list[i]
            for flat_list in all_lists
            for i in range(0, len(flat_list), 2)
        ]

        min_x = min(all_starts) if all_starts else 0
        max_x = max(all_ends) if all_ends else 10
        ax.set_xlim(max(0, min_x - 0.5), max_x + 1)

        plt.tight_layout()
        plt.show()

stat = Stat()