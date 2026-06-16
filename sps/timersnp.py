import time
import torch
from datetime import datetime
import csv
from pathlib import Path

class TimerSNP:
    DIR_NAME = "times"
    
    def __init__(self, expected_steps=100, FILENAME="performance_times.csv", use_cuda=False):
        self._buffer = [None] * expected_steps
        self._index = 0
        self._start_times = [0.0] * expected_steps
        self._step_names = [None] * expected_steps
        self.FILENAME = FILENAME
        self.use_cuda = use_cuda
        
        if use_cuda:
            self._start_events = [None] * expected_steps
            self._end_events = [None] * expected_steps
    
    def start_step(self, step_name):
        self._step_names[self._index] = step_name
        
        if self.use_cuda and torch.cuda.is_available():
            self._start_events[self._index] = torch.cuda.Event(enable_timing=True)
            self._end_events[self._index] = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            self._start_events[self._index].record()
        else:
            self._start_times[self._index] = time.perf_counter()
    
    def end_step(self):
        if self._step_names[self._index] is not None:
            if self.use_cuda and torch.cuda.is_available():
                self._end_events[self._index].record()
                torch.cuda.synchronize()
                elapsed = self._start_events[self._index].elapsed_time(self._end_events[self._index])
                self._buffer[self._index] = elapsed
            else:
                elapsed = time.perf_counter() - self._start_times[self._index]
                self._buffer[self._index] = elapsed * 1000
            self._index += 1
    
    def export_to_csv(self):
        base_dir = Path(__file__).parent.parent 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = base_dir / self.DIR_NAME / f"{self.FILENAME}_{timestamp}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Step', 'Time (ms)']) 
            
            for i in range(self._index):
                if self._buffer[i] is not None:
                    writer.writerow([self._step_names[i], f"{self._buffer[i]:.3f}"])