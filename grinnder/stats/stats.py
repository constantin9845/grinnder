import time
from typing import List

class Stat:
    def __init__(self):
        self.running: bool = False
        self.start_time = 0
        self.compute_start = 0
        self.partition_load_CPU_timesteps: List[int] = []
        self.partition_load_GPU_timesteps: List[int] = []
        self.compute_timesteps: List[int] = []

    def started(self):
        return self.running == True
    
    def start(self):
        if self.running:
            return
        
        self.running = True
        self.start_time = time.perf_counter_ns()

    def load_CPU_timestamp(self):
        self.partition_load_CPU_timesteps.append(time.perf_counter_ns())

    def load_GPU_timestamp(self):
        self.partition_load_GPU_timesteps.append(time.perf_counter_ns())

    def begin_compute(self):
        if self.compute_start == 0:
            self.compute_start = time.perf_counter_ns()

    def compute_timestamp(self):
        t = time.perf_counter_ns()
        self.compute_timesteps.append(t)

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
        print(f'SSD --> CPU : \n{self.partition_load_CPU_timesteps}\n')
        print(f'CPU --> GPU : \n{self.partition_load_GPU_timesteps}\n')
        print(f'Compute: \n {self.compute_start} | {self.compute_timesteps}\n')
        print(f'==============================')

stat = Stat()