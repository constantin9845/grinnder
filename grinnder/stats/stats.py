import time
from typing import List

class Stat:
    def __init__(self):
        self.running: bool = False
        self.start_time = 0
        self.compute_start = 0
        self.forward_time = 0
        self.loss_time = 0
        self.weights_time = 0
        self.partition_load_CPU_timesteps: List[int] = []
        self.partition_load_GPU_timesteps: List[int] = []
        self.compute_timesteps: List[int] = []

        self.backward_start = 0
        self.backward_direct_load_timesteps: List[int] = []
        self.recompute_timesteps: List[int] = []


    def started(self):
        return self.running == True
    
    def start(self):
        if self.running:
            return
        
        self.running = True
        self.start_time = time.perf_counter_ns()

    def load_CPU_timestamp(self):
        #print("Loaded SSD --> CPU")
        self.partition_load_CPU_timesteps.append(time.perf_counter_ns())

    def load_GPU_timestamp(self):
        #print("Loaded CPU --> GPU")
        self.partition_load_GPU_timesteps.append(time.perf_counter_ns())

    def load_GDS_timestamp(self):
        #print("Loaded CPU --> GPU")
        self.partition_load_GPU_timesteps.append(time.perf_counter_ns())

    def begin_compute(self):
        if self.compute_start == 0:
            self.compute_start = time.perf_counter_ns()

    def compute_timestamp(self):
        #print("Computed forward step for partition")
        t = time.perf_counter_ns()
        self.compute_timesteps.append(t)

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
        self.compute_start = 0
        self.partition_load_CPU_timesteps.clear()
        self.partition_load_GPU_timesteps.clear()
        self.compute_timesteps.clear()

    def print_timeline(self):
        print(f'========== TIMELINE ==========')
        print(f'Start = {self.start_time}\n')

        print(f'SSD --> CPU :')
        for i in self.partition_load_CPU_timesteps:
            print(f'\t{i}')

        print(f'CPU --> GPU :')
        for i in self.partition_load_GPU_timesteps:
            print(f'\t{i}')

        print(f'SSD --> GPU (GDS):')
        for i in self.backward_direct_load_timesteps:
            print(f'\t{i}')

        print(f'Compute: start = {self.compute_start}')
        for i in self.compute_timesteps:
            print(f'\t{i}')

        print(f'\n\n\tStart = {self.start_time} \n\tforward done = {self.forward_time} \n\tloss done = {self.loss_time} \n\tbackward done = {self.backward_time} \n\t weights done = {self.weights_time}')

        print(f'==============================')

stat = Stat()