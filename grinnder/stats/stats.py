import time
from typing import List

class Stat:
    def __init__(self):
        self.running: bool = False
        self.start_time = 0
        self.start_time_compute = 0
        self.partition_load_timesteps: List[int] = []
        self.compute_timesteps: List[int] = []

    def started(self):
        return self.running == True
    
    def start(self):
        if self.running:
            return
        
        self.running = True
        self.start_time = time.perf_counter_ns()

    def load_timestamp(self):
        self.partition_load_timesteps.append(time.perf_counter_ns())

    def compute_timestamp(self):
        t = time.perf_counter_ns()
        self.compute_timesteps.append(t)

        if self.start_time_compute == 0:
            self.start_time_compute = t

    def reset(self):
        self.running = False
        self.start_time = 0
        self.start_time_compute = 0
        self.partition_load_timesteps.clear()
        self.compute_timesteps.clear()

    