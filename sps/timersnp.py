import time
import torch
from datetime import datetime
import csv
from pathlib import Path
from sps.config import Config

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
    
    def export_to_csv(self, putInQfolder=False):
        base_dir = Path(__file__).parent.parent 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if putInQfolder:
            csv_path = base_dir / self.DIR_NAME / f"Q_{Config.Q_RANGE}" / f"{self.FILENAME}_{timestamp}.csv"
        else:
            csv_path = base_dir / self.DIR_NAME / f"{self.FILENAME}_{timestamp}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Step', 'Time (ms)']) 
            
            for i in range(self._index):
                if self._buffer[i] is not None:
                    writer.writerow([self._step_names[i], f"{self._buffer[i]:.3f}"])
    
    def export_training_times(self, system_name):
        """
        Esporta i tempi di training in un CSV organizzato per Q_RANGE e TEST_NUM.
        Le colonne rappresentano Q2-T0, Q2-T1, Q2-T2, Q3-T0, ...
        Le righe rappresentano SVM e LogReg per ciascuna dimensione (TRAIN_SIZE).
        """
        base_dir = Path(__file__).parent.parent
        csv_path = base_dir / self.DIR_NAME / f"training_times_{system_name}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Raccogli i dati dal buffer
        training_data = {}  # {Q_RANGE: {TEST_NUM: {TRAIN_SIZE: {'SVM': time, 'LogReg': time}}}}
        
        for i in range(self._index):
            if self._buffer[i] is not None:
                step_name = self._step_names[i]
                time_ms = self._buffer[i]
                
                # Estrai le informazioni dal nome dello step
                # Formato atteso: "Q:{Q_RANGE}_T:{TEST_NUM}_S{TRAIN_SIZE}_{SYSTEM}_TRAINING_SVM" o "_LogReg"
                parts = step_name.split('_')
                if len(parts) >= 5 and 'TRAINING' in step_name:
                    try:
                        q_range = int(parts[0].split(':')[1])
                        test_num = int(parts[1].split(':')[1])
                        train_size = int(parts[2].split('S')[1])
                        model_type = parts[-1]  # SVM o LogReg
                        
                        if q_range not in training_data:
                            training_data[q_range] = {}
                        if test_num not in training_data[q_range]:
                            training_data[q_range][test_num] = {}
                        if train_size not in training_data[q_range][test_num]:
                            training_data[q_range][test_num][train_size] = {}
                        
                        training_data[q_range][test_num][train_size][model_type] = time_ms
                    except:
                        continue
        
        # Organizza i dati per il CSV
        if not training_data:
            print("No training data to export")
            return
        
        # Trova tutte le combinazioni uniche
        q_ranges = sorted(training_data.keys())
        test_nums = sorted(set(tn for q in training_data for tn in training_data[q]))
        train_sizes = sorted(set(ts for q in training_data for tn in training_data[q] for ts in training_data[q][tn]))
        
        # Crea le intestazioni delle colonne
        columns = []
        for q in q_ranges:
            for t in test_nums:
                columns.append(f"Q{q}-T{t}")
        
        # Scrivi il CSV
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Intestazione
            header = ['Size', 'Model'] + columns
            writer.writerow(header)
            
            # Dati per ogni size e modello
            for size in train_sizes:
                for model in ['SVM', 'LogReg']:
                    row = [f"S{size}", model]
                    for q in q_ranges:
                        for t in test_nums:
                            time_value = training_data.get(q, {}).get(t, {}).get(size, {}).get(model, '')
                            row.append(f"{time_value:.3f}" if time_value != '' else '')
                    writer.writerow(row)
        
        print(f"Training times saved to: {csv_path}")
        return csv_path
        
    def export_step_times(self, system_name, timer_type, phase="TRAIN"):
        """
        Esporta i tempi degli step in un CSV organizzato per Q_RANGE e TEST_NUM.
        
        Args:
            system_name: nome del sistema (es. "MSNPSystem_GPU", "MSNPSystem_CPU")
            timer_type: "InStep" o "PerStep"
            phase: "TRAIN" o "TEST"
        """
        base_dir = Path(__file__).parent.parent
        size_info = f"{Config.TRAIN_SIZE}-{Config.TEST_SIZE}"
        
        # Il nome del file ora include la fase
        if timer_type == "InStep":
            filename = f"times_{phase}_InStep_{system_name}_S{size_info}.csv"
        else:
            filename = f"times_{phase}_PerStep_{system_name}_S{size_info}.csv"
        
        csv_path = base_dir / self.DIR_NAME / filename
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Raccogli i dati dal buffer
        step_data = {}
        current_q = Config.Q_RANGE
        current_t = Config.TIME_TEST_NUM
        column_key = f"Q{current_q}-T{current_t}"
        
        for i in range(self._index):
            if self._buffer[i] is not None:
                step_name = str(self._step_names[i])
                time_ms = self._buffer[i]
                
                if step_name not in step_data:
                    step_data[step_name] = {}
                
                step_data[step_name][column_key] = time_ms
        
        # Se il file esiste già, carica i dati esistenti
        existing_data = {}
        if csv_path.exists():
            try:
                with open(csv_path, 'r', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    existing_columns = reader.fieldnames[1:] if reader.fieldnames else []
                    for row in reader:
                        step = row['Step']
                        existing_data[step] = {}
                        for col in existing_columns:
                            if col and row[col]:  # Verifica che col non sia vuoto
                                try:
                                    existing_data[step][col] = float(row[col])
                                except ValueError:
                                    pass
            except Exception as e:
                print(f"Error loading existing step times: {e}")
                existing_data = {}
        
        # Unisci i dati esistenti con i nuovi
        for step, times in existing_data.items():
            if step not in step_data:
                step_data[step] = {}
            step_data[step].update(times)
        
        # Trova tutte le colonne (Q-T combinations) ordinate
        all_columns = set()
        for step_times in step_data.values():
            all_columns.update(step_times.keys())
        
        # Rimuovi eventuali stringhe vuote
        all_columns.discard('')
        
        # Ordina le colonne: prima per Q, poi per T
        def sort_key(col):
            try:
                # Gestisci il formato "Q2-T0"
                parts = col.replace('Q', '').replace('T', '-').split('-')
                if len(parts) == 2 and parts[0] and parts[1]:
                    return (int(parts[0]), int(parts[1]))
            except (ValueError, IndexError):
                pass
            return (9999, 9999)  # Metti in fondo gli elementi non parsabili
        
        sorted_columns = sorted(all_columns, key=sort_key)
        
        # Ordina gli step in modo intelligente
        def step_sort_key(step_name):
            try:
                step_name_str = str(step_name)
                # Estrae il numero dello step dal nome (es. "0> Image Input" -> 0)
                if '>' in step_name_str:
                    step_num = int(step_name_str.split('>')[0])
                    if timer_type == "InStep":
                        sub_step = step_name_str.split('>')[1].strip()
                        sub_step_order = {
                            'Image Input': 0,
                            'Extended Config + Spiking Vector construction': 1,
                            'NetGain Vector update: smpi @ spikingVec': 2,
                            'Configuration Vector update': 3,
                            'Pooling image update': 4
                        }
                        sub_order = sub_step_order.get(sub_step, 99)
                        return (step_num, sub_order)
                    else:
                        return (step_num, 0)
                else:
                    # Per PerStep, il nome è solo un numero
                    return (int(step_name_str), 0)
            except (ValueError, IndexError):
                return (999999, 0)
        
        sorted_steps = sorted(step_data.keys(), key=step_sort_key)
        
        # Scrivi il CSV
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Intestazione
            header = ['Step'] + sorted_columns
            writer.writerow(header)
            
            # Dati per ogni step
            for step in sorted_steps:
                row = [step]
                for col in sorted_columns:
                    time_value = step_data[step].get(col, '')
                    row.append(f"{time_value:.3f}" if time_value != '' else '')
                writer.writerow(row)
        
        print(f"Step times ({timer_type}) saved to: {csv_path}")
        return csv_path