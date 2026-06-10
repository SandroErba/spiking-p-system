import numpy as np
import torch
from sps.spike_utils import TransformationRule
from sps.snp_system import SNPSystem  
from sps.config import Config


class MSNPSystemExactGPU:
    
    def __init__(self, configurationVector, spikingVector, spikingTransitionMatrix, 
                 synapsesMatrix, ruleVector, max_steps=1000, deterministic=True, 
                 single_spike_train=None, input_neurons=None, output_neurons=None,
                 applyingRuleVector=None, device='cpu', testsize=1, use_sparse=False):
        
        # Set device e dtype
        if device == 'gpu' or device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.dtype = torch.float32
                self._use_cuda = True
                print(f"MSNPSystemGPU using device: CUDA GPU with dtype={self.dtype}")
            else:
                print("Warning: CUDA not available. Falling back to CPU.")
                self.device = torch.device('cpu')
                self.dtype = torch.int32
                self._use_cuda = False
                print(f"MSNPSystemGPU using device: CPU with dtype={self.dtype}")
        else:
            self.device = torch.device('cpu')
            self.dtype = torch.int32
            self._use_cuda = False
            print(f"MSNPSystemGPU using device: CPU with dtype={self.dtype}")

        if applyingRuleVector is None or configurationVector is None or \
            spikingTransitionMatrix is None or ruleVector is None:
            raise ValueError("ApplyingRuleVector, configurationVector, spikingTransitionMatrix and ruleVector cannot be None")
        
        if max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        
        # Gestione automatica matrici sparse/dense
        self._is_sparse = use_sparse
        self._sparse_stm = None
        self._sparse_sm = None
        
        if use_sparse:
            rule_num = spikingTransitionMatrix.shape[0]
            neuron_num = spikingTransitionMatrix.shape[1]
            
            # Converte matrici NumPy in PyTorch sparse COO
            self.spikingTransitionMatrix = self._numpy_to_torch_sparse(
                spikingTransitionMatrix
            )
            self.synapsesMatrix = self._numpy_to_torch_sparse(
                synapsesMatrix
            )
            
            # Calcola sMpi come prodotto sparse (element-wise)
            self.sMpi = self._sparse_element_wise_mul(
                self.spikingTransitionMatrix, 
                self.synapsesMatrix
            )
            
            sparsity = self.spikingTransitionMatrix._nnz() / (rule_num * neuron_num)
            print(f"Using PyTorch sparse tensors on {self.device}: "
                  f"{self.spikingTransitionMatrix._nnz()} nonzeros ({sparsity:.2%} density)")
        else:
            rule_num = len(spikingTransitionMatrix)
            neuron_num = len(configurationVector)
            
            # Tensori densi normali
            self.spikingTransitionMatrix = torch.as_tensor(
                spikingTransitionMatrix, dtype=self.dtype, device=self.device
            ).contiguous()
            self.synapsesMatrix = torch.as_tensor(
                synapsesMatrix, dtype=self.dtype, device=self.device
            ).contiguous()
            
            # sMpi denso
            self.sMpi = torch.mul(self.spikingTransitionMatrix, self.synapsesMatrix)
        
        self.testsize = testsize
        
        # Pooling image
        if output_neurons is not None:
            self.pooling_image = torch.zeros(
                (len(output_neurons), self.testsize), 
                dtype=self.dtype, 
                device='cpu'
            )
        else:
            self.pooling_image = None
        
        # Converte input/output neurons in tensori
        if input_neurons is not None:
            self.input_neurons = torch.as_tensor(
                input_neurons, dtype=torch.int64, device=self.device
            )
        else:
            self.input_neurons = None
            
        if output_neurons is not None:
            self.output_neurons = torch.as_tensor(
                output_neurons, dtype=torch.int64, device=self.device
            )
        else:
            self.output_neurons = None

        self.max_steps = max_steps
        self.deterministic = deterministic
        
        # Tensori principali
        self.configurationVector = torch.as_tensor(
            configurationVector, dtype=self.dtype, device=self.device
        ).contiguous()
        
        self.ruleVector = torch.as_tensor(
            ruleVector, dtype=self.dtype, device=self.device
        ).contiguous()
        
        self.applyingRuleVector = torch.as_tensor(
            applyingRuleVector, dtype=torch.int32, device=self.device
        ).contiguous()
        
        # Vettori intermedi pre-allocati
        self.netGainVector = torch.zeros(neuron_num, dtype=self.dtype, device=self.device)
        
        # Indici per extended configuration
        self.ruleCountPerNeuron = torch.bincount(
            self.applyingRuleVector, minlength=neuron_num
        )
        self.repeat_indices = torch.repeat_interleave(
            torch.arange(neuron_num, device=self.device), 
            self.ruleCountPerNeuron
        ).contiguous()
        
        # Buffer temporanei pre-allocati
        self._extended_config = torch.empty(rule_num, dtype=self.dtype, device=self.device)
        self._diff_buffer = torch.empty(rule_num, dtype=self.dtype, device=self.device)
        
        # Valori costanti
        if self.dtype == torch.int32:
            self._spike_value = torch.tensor(1, dtype=self.dtype, device=self.device)
            self._zero_value = torch.tensor(0, dtype=self.dtype, device=self.device)
        else:
            self._spike_value = torch.tensor(1.0, dtype=self.dtype, device=self.device)
            self._zero_value = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        
        # Spiking vector
        if spikingVector is None:
            self.spikingVector = torch.zeros(rule_num, dtype=self.dtype, device=self.device)
        else:
            self.spikingVector = torch.as_tensor(
                spikingVector, dtype=self.dtype, device=self.device
            ).contiguous()
        
        # Spike train
        if single_spike_train is not None:
            self.single_spike_train = torch.as_tensor(
                single_spike_train, dtype=self.dtype, device=self.device
            ).contiguous()
            self._single_spike_length = len(single_spike_train)
        else:
            self.single_spike_train = torch.empty(0, dtype=self.dtype, device=self.device)
            self._single_spike_length = 0
        
        # Immagini
        self.img_spike_train = None
        self._img_spike_length = 0
        
        # Time step
        self.t_step = 0
        
        # Modalità
        self._cnn_mode = (Config.MODE == "CNN")
        
        # Pre-calcola range pooling
        if self.pooling_image is not None:
            self._pooling_start = Config.NUM_LAYERS - 2
            self._pooling_end = self.testsize + Config.NUM_LAYERS - 3
        else:
            self._pooling_start = None
            self._pooling_end = None
        
        # Stream CUDA
        if self._use_cuda:
            self._stream = torch.cuda.Stream()
        else:
            self._stream = None
        
        # Compilazione JIT
        if hasattr(torch, 'compile') and self._use_cuda:
            try:
                self.step = torch.compile(self.step, mode='reduce-overhead')
                print("Using torch.compile for step method")
            except Exception as e:
                print(f"torch.compile not available: {e}")
    
    @staticmethod
    def _numpy_to_torch_sparse(np_array):
        """Converte array NumPy in tensore PyTorch sparse COO"""
        if not isinstance(np_array, np.ndarray):
            np_array = np_array.toarray() if hasattr(np_array, 'toarray') else np.array(np_array)
        
        # Trova indici non-zero
        rows, cols = np.nonzero(np_array)
        values = np_array[rows, cols]
        
        # Crea tensore sparse COO
        indices = torch.tensor([rows, cols], dtype=torch.int64)
        values = torch.tensor(values, dtype=torch.float32)
        size = np_array.shape
        
        return torch.sparse_coo_tensor(indices, values, size)
    
    @staticmethod
    def _sparse_element_wise_mul(sparse_tensor1, sparse_tensor2):
        """Moltiplicazione element-wise per tensori sparsi"""
        # Converte entrambi a COO e moltiplica i valori
        t1 = sparse_tensor1.coalesce()
        t2 = sparse_tensor2.coalesce()
        
        # Per semplicità, se entrambi hanno la stessa struttura sparsa
        if torch.equal(t1.indices(), t2.indices()):
            return torch.sparse_coo_tensor(
                t1.indices(),
                t1.values() * t2.values(),
                t1.size()
            )
        else:
            # Altrimenti converti a denso per la moltiplicazione
            # (per strutture diverse, potrebbe non essere efficiente)
            return torch.mul(t1.to_dense(), t2.to_dense())
    
    def loadImages(self, img_spike_train):
        """Load images as spike trains"""
        if len(img_spike_train.shape) == 3:
            img_spike_train = img_spike_train.reshape(img_spike_train.shape[0], -1)
        
        self.img_spike_train = torch.as_tensor(
            img_spike_train, dtype=self.dtype, device=self.device
        ).contiguous()
        self._img_spike_length = img_spike_train.shape[0]
    
    def step(self, verbose=False):
        """Execute one step - ottimizzato per GPU con supporto sparse"""
        
        # Input handling
        if self._cnn_mode and self.t_step < self._img_spike_length:
            self.configurationVector.index_add_(
                0,
                self.input_neurons,
                self.img_spike_train[self.t_step]
            )
        elif self._single_spike_length > 0 and self.t_step < self._single_spike_length:
            if self.single_spike_train[self.t_step] == self._spike_value:
                self.configurationVector.index_add_(
                    0,
                    self.input_neurons,
                    self._spike_value.expand(len(self.input_neurons))
                )
        
        # Extended configuration
        torch.index_select(
            self.configurationVector, 0, self.repeat_indices,
            out=self._extended_config
        )
        
        # Calcolo spiking vector
        if self.dtype == torch.int32:
            torch.sub(self._extended_config, self.ruleVector, out=self._diff_buffer)
            torch.clamp(self._diff_buffer, min=0, out=self._diff_buffer)
            torch.add(self._diff_buffer, 1, out=self._diff_buffer)
            torch.floor_divide(
                torch.tensor(1, dtype=self.dtype, device=self.device),
                self._diff_buffer,
                out=self.spikingVector
            )
        else:
            torch.ge(self._extended_config, self.ruleVector, out=self.spikingVector)
            self.spikingVector.mul_(1.0)
        
        # Calcolo net gain vector (supporta sparse)
        if self._is_sparse and self.sMpi.is_sparse:
            # Moltiplicazione sparse-dense ottimizzata
            torch.sparse.mm(
                self.sMpi.t(),  # Trasponi per mv
                self.spikingVector.unsqueeze(1),
                out=self.netGainVector.unsqueeze(1)
            )
            self.netGainVector = self.netGainVector.squeeze()
        else:
            # Moltiplicazione densa standard
            torch.mv(self.sMpi, self.spikingVector, out=self.netGainVector)
        
        # Aggiornamento configuration vector
        self.configurationVector.add_(self.netGainVector)
        
        # Pooling
        if self.pooling_image is not None and self._pooling_start is not None:
            step_in_pooling = self.t_step - self._pooling_start
            if 0 <= step_in_pooling < self._pooling_end:
                self.pooling_image[:, step_in_pooling] = \
                    self.configurationVector[self.output_neurons].cpu()
        
        self.t_step += 1
        return True
    
    def execute(self, verbose=False, startAgain=True):
        """Execute with optimized GPU stream"""
        if startAgain:
            self.t_step = 0
        
        input_length = self._img_spike_length if self._cnn_mode else self._single_spike_length
        
        if self._use_cuda and self._stream is not None:
            with torch.cuda.stream(self._stream):
                return self._execute_loop(input_length, verbose)
        else:
            return self._execute_loop(input_length, verbose)
    
    def _execute_loop(self, input_length, verbose):
        """Internal execution loop"""
        zero_vector = self._zero_value.expand_as(self.spikingVector)
        
        while self.step(verbose=False) and self.t_step < self.max_steps:
            if self.t_step >= input_length and torch.equal(self.spikingVector, zero_vector):
                if verbose:
                    print("Computation halts: spiking vector is zero, input is accepted")
                return True
        
        if verbose:
            print("Computation halts: maximum number of steps reached, input is rejected")
        return False
    
    def get_configuration_vector(self):
        return self.configurationVector.cpu().numpy() if self.device.type != 'cpu' else self.configurationVector.numpy()
    
    def get_spiking_vector(self):
        return self.spikingVector.cpu().numpy() if self.device.type != 'cpu' else self.spikingVector.numpy()
    
    def get_net_gain_vector(self):
        return self.netGainVector.cpu().numpy() if self.device.type != 'cpu' else self.netGainVector.numpy()
    
    def get_spiking_transition_matrix(self):
        if self._is_sparse:
            return self.sMpi.to_dense().cpu().numpy()
        return self.sMpi.cpu().numpy() if self.device.type != 'cpu' else self.sMpi.numpy()
    
    def get_rule_vector(self):
        return self.ruleVector.cpu().numpy() if self.device.type != 'cpu' else self.ruleVector.numpy()
    
    def get_applying_rule_vector(self):
        return self.applyingRuleVector.cpu().numpy() if self.device.type != 'cpu' else self.applyingRuleVector.numpy()
    
    def to(self, device):
        """Move system to device"""
        new_device = torch.device(device)
        
        if new_device.type == 'cuda':
            new_dtype = torch.float32
            self._use_cuda = True
            if self._stream is None:
                self._stream = torch.cuda.Stream()
        else:
            new_dtype = torch.int32
            self._use_cuda = False
            self._stream = None
        
        tensor_attrs = [
            'configurationVector', 'ruleVector', 'netGainVector',
            'spikingVector', 'single_spike_train', 'repeat_indices',
            '_extended_config', '_diff_buffer', '_spike_value', '_zero_value',
            'input_neurons', 'output_neurons'
        ]
        
        with torch.no_grad():
            for attr in tensor_attrs:
                if hasattr(self, attr):
                    tensor = getattr(self, attr)
                    if tensor is not None:
                        if new_dtype != self.dtype:
                            setattr(self, attr, tensor.to(dtype=new_dtype, device=new_device))
                        else:
                            setattr(self, attr, tensor.to(device=new_device))
            
            # Gestione tensori sparsi
            if self._is_sparse:
                self.spikingTransitionMatrix = self.spikingTransitionMatrix.to(new_device)
                self.synapsesMatrix = self.synapsesMatrix.to(new_device)
                self.sMpi = self.sMpi.to(new_device)
            else:
                if new_dtype != self.dtype:
                    self.spikingTransitionMatrix = self.spikingTransitionMatrix.to(dtype=new_dtype, device=new_device)
                    self.synapsesMatrix = self.synapsesMatrix.to(dtype=new_dtype, device=new_device)
                    self.sMpi = self.sMpi.to(dtype=new_dtype, device=new_device)
                else:
                    self.spikingTransitionMatrix = self.spikingTransitionMatrix.to(new_device)
                    self.synapsesMatrix = self.synapsesMatrix.to(new_device)
                    self.sMpi = self.sMpi.to(new_device)
        
        self.applyingRuleVector = self.applyingRuleVector.to(new_device)
        self.dtype = new_dtype
        self.device = new_device
        
        if hasattr(torch, 'compile') and self._use_cuda:
            try:
                self.step = torch.compile(self.step, mode='reduce-overhead')
            except Exception:
                pass
        
        return self
    
    def __str__(self):
        memory_format = "sparse" if self._is_sparse else "dense"
        return (f"Device: {self.device} (dtype: {self.dtype}, format: {memory_format})\n"
                f"Deterministic: {self.deterministic}\n"
                f"Input Neurons: {self.input_neurons}\n"
                f"Output Neurons: {self.output_neurons}\n"
                f"Configuration Vector: {self.get_configuration_vector()}\n"
                f"Spiking Vector: {self.get_spiking_vector()}\n"
                f"Net Gain Vector: {self.get_net_gain_vector()}\n"
                f"Rule Vector: {self.get_rule_vector()}\n"
                f"Applying Rule Vector: {self.get_applying_rule_vector()}\n")