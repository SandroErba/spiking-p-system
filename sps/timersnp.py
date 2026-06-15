import time
import csv
from pathlib import Path
from contextlib import contextmanager

class TimerSNP:

    DIR_NAME = "times"
    
    def __init__(self, expected_steps=100, FILENAME="performance_times.csv"):
        self._buffer = [None] * expected_steps
        self._index = 0
        self._start_times = [0.0] * expected_steps
        self._step_names = [None] * expected_steps
        self.FILENAME = FILENAME
    
    def start_step(self, step_name):
        self._step_names[self._index] = step_name
        self._start_times[self._index] = time.perf_counter()
    
    def end_step(self):
        if self._step_names[self._index] is not None:
            elapsed = time.perf_counter() - self._start_times[self._index]
            self._buffer[self._index] = elapsed * 1000
            self._index += 1  
    
    def export_to_csv(self):
        base_dir = Path(__file__).parent.parent 
        csv_path = base_dir / self.DIR_NAME / self.FILENAME  # ← self.DIR_NAME
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Step', 'Time (ms)']) 
            
            for i in range(self._index):
                if self._buffer[i] is not None:
                    writer.writerow([self._step_names[i], f"{self._buffer[i]:.3f}"])