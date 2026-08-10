import time
from typing import List

class Stat:
    def __init__(self):
        self.running: bool = False
        self.start_time = 0

        #forward
        self.forward_time = 0
        self.forward_GPU_start = 0
        self.forward_compute_start = 0
        self.forward_partition_load_CPU_timesteps: List[int] = []
        self.forward_partition_load_GPU_timesteps: List[int] = []
        self.forward_compute_timesteps: List[int] = []

        # loss
        self.loss_time = 0
        self.loss_GPU_start = 0
        self.loss_start = 0
        self.loss_partition_load_CPU_timesteps: List[int] = []
        self.loss_partition_load_GPU_timesteps: List[int] = []

        # backward
        self.backward_start = 0
        self.backward_GPU_start = 0
        self.backward_compute_start = 0
        self.backward_partition_load_CPU_timesteps: List[int] = []
        self.backward_partition_load_GPU_timesteps: List[int] = []
        self.backward_direct_load_timesteps: List[int] = []
        self.backward_compute_timesteps: List[int] = []

        self.weights_time = 0


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

    def load_CPU_timestamp(self, stage):
        t = time.perf_counter_ns()
        if stage == "forward":
            t = t - self.start_time
            t = t / 1000000000.0
            t = round(t,2)
            self.forward_partition_load_CPU_timesteps.append(t)
        elif stage == "loss":
            t = t - self.loss_start
            t = t / 1000000000.0
            t = round(t,2)
            self.loss_partition_load_CPU_timesteps.append(t)
        else:
            t = t - self.backward_start
            t = t / 1000000000.0
            t = round(t,2)
            self.backward_partition_load_CPU_timesteps.append(t)

    def load_GPU_timestamp(self, stage):
        t = time.perf_counter_ns()
        if stage == "forward":
            if len(self.forward_partition_load_GPU_timesteps) == 0:
                self.forward_GPU_start = t

            t = t - self.forward_GPU_start
            t = t / 1000000000.0
            t = round(t,2) + self.forward_partition_load_CPU_timesteps[-1]
            self.forward_partition_load_GPU_timesteps.append(t)

        elif stage == "loss":
            if len(self.loss_partition_load_GPU_timesteps) == 0:
                self.loss_GPU_start = t

            t = t - self.loss_GPU_start
            t = t / 1000000000.0
            t = round(t,2) + self.loss_partition_load_CPU_timesteps[-1]
            self.loss_partition_load_GPU_timesteps.append(t)

        else:
            if len(self.backward_partition_load_GPU_timesteps) == 0:
                self.backward_GPU_start = t

            t = t - self.backward_GPU_start
            t = t / 1000000000.0
            # top layer uses losses instead of gradients, losses are already in HOST
            if len(self.backward_partition_load_CPU_timesteps) > 0:
                t = round(t,2) + self.backward_partition_load_CPU_timesteps[-1]
            self.backward_partition_load_GPU_timesteps.append(t)

    def load_GDS_timestamp(self):
        t = time.perf_counter_ns()
        t = t - self.backward_start
        t = t / 1000000000.0
        t = round(t,2)
        self.backward_direct_load_timesteps.append(time.perf_counter_ns())

    def begin_compute(self, stage):
        t = time.perf_counter_ns()
        if stage == "foward":
            if self.forward_compute_start == 0:
                t = t - self.start_time
                self.forward_compute_start = t

        else:
            if self.backward_compute_start == 0:
                t = t - self.backward_start
                self.backward_compute_start = t

    def compute_timestamp(self, stage):
        t = time.perf_counter_ns()

        if stage == "forward":
            t = t - self.forward_compute_start
            t = t / 1000000000.0
            t = round(t,2)
            self.forward_compute_timesteps.append(t)
        else:
            t = t - self.backward_compute_start
            t = t / 1000000000.0
            t = round(t,2)
            self.backward_compute_timesteps.append(t)

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

        #forward
        self.forward_time = 0
        self.forward_GPU_start = 0
        self.forward_compute_start = 0
        self.forward_partition_load_CPU_timesteps.clear()
        self.forward_partition_load_GPU_timesteps.clear()
        self.forward_compute_timesteps.clear()

        # loss
        self.loss_time = 0
        self.loss_GPU_start = 0
        self.loss_start = 0
        self.loss_partition_load_CPU_timesteps.clear()
        self.loss_partition_load_GPU_timesteps.clear()

        # backward
        self.backward_start = 0
        self.backward_GPU_start = 0
        self.backward_compute_start = 0
        self.backward_partition_load_CPU_timesteps.clear()
        self.backward_partition_load_GPU_timesteps.clear()
        self.backward_direct_load_timesteps.clear()
        self.backward_compute_timesteps.clear()

        self.weights_time = 0


    def print_timeline(self):
        from itertools import zip_longest
        print("========== TIMELINE ==========")
        print(f"Start = {self.start_time}\n")

        W = 14
        FILL = "-"  # Placeholder when lists have unequal lengths

        # --- FORWARD STEP ---
        print("[FORWARD STEP]")
        header_fwd = (
            f"\t| {'SD --> CPU':<{W}} | {'CPU --> GPU':<{W}} | {'Compute':<{W}} |"
        )
        divider_fwd = f"\t+{'-' * (W + 2)}+{'-' * (W + 2)}+{'-' * (W + 2)}+"
        print(header_fwd)
        print(divider_fwd)

        for i, j, k in zip_longest(
            self.forward_partition_load_CPU_timesteps,
            self.forward_partition_load_GPU_timesteps,
            self.forward_compute_timesteps,
            fillvalue=FILL,
        ):
            print(f"\t| {str(i):<{W}} | {str(j):<{W}} | {str(k):<{W}} |")

        # --- LOSS STEP ---
        print("\n\n[LOSS STEP]")
        header_loss = f"\t| {'SD --> CPU':<{W}} | {'CPU --> GPU':<{W}} |"
        divider_loss = f"\t+{'-' * (W + 2)}+{'-' * (W + 2)}+"
        print(header_loss)
        print(divider_loss)

        for i, j in zip_longest(
            self.loss_partition_load_CPU_timesteps,
            self.loss_partition_load_GPU_timesteps,
            fillvalue=FILL,
        ):
            print(f"\t| {str(i):<{W}} | {str(j):<{W}} |")

        # --- BACKWARD STEP ---
        print("\n\n[BACKWARD STEP]")
        header_bwd = f"\t| {'SD --> CPU':<{W}} | {'CPU --> GPU':<{W}} | {'Compute':<{W}} | {'GDS':<{W}} |"
        divider_bwd = f"\t+{'-' * (W + 2)}+{'-' * (W + 2)}+{'-' * (W + 2)}+{'-' * (W + 2)}+"
        print(header_bwd)
        print(divider_bwd)

        for i, j, k, l in zip_longest(
            self.backward_partition_load_CPU_timesteps,
            self.backward_partition_load_GPU_timesteps,
            self.backward_compute_timesteps,
            self.backward_direct_load_timesteps,
            fillvalue=FILL,
        ):
            print(
                f"\t| {str(i):<{W}} | {str(j):<{W}} | {str(k):<{W}} | {str(l):<{W}} |"
            )

        # --- SUMMARY ---
        print(
            f"\n\n\tStart = {self.start_time} "
            f"\n\tforward done = {self.forward_time} "
            f"\n\tloss done = {self.loss_time} "
            f"\n\tbackward done = {self.backward_time} "
            f"\n\tweights done = {self.weights_time}"
        )
        print("==============================")

stat = Stat()