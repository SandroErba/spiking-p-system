# File: sps/m_opt_snp_pytorch_exact_GPU_and_CPU.py

import numpy as np
import torch
from sps.config import Config


class MSNPSystemExactGPU:
    
    def __init__(self, configurationVector, spikingVector, spikingTransitionMatrix, 
                 synapsesMatrix, ruleVector, max_steps=1000, deterministic=True, 
                 single_spike_train=None, input_neurons=None, output_neurons=None,
                 applyingRuleVector=None, device='cpu', testsize=1):
        
        # Setup dispositivo
        if device in ('gpu', 'cuda'):
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.dtype = torch.float32
            else:
                print("CUDA non disponibile. Uso CPU.")
                self.device = torch.device('cpu')
                self.dtype = torch.int32
        else:
            self.device = torch.device('cpu')
            self.dtype = torch.int32
        
        print(f"MSNPSystemGPU using device: {self.device} with dtype={self.dtype}")
        
        # Validazione
        if any(x is None for x in [applyingRuleVector, configurationVector, 
                                    spikingTransitionMatrix, ruleVector]):
            raise ValueError("Parametri obbligatori mancanti")
        
        if max_steps <= 0:
            raise ValueError("max_steps deve essere positivo")
        
        # Determina se le matrici sono sparse PyTorch
        self._is_sparse = isinstance(spikingTransitionMatrix, torch.Tensor) and spikingTransitionMatrix.is_sparse
        
        # Dimensioni
        if self._is_sparse:
            self.rule_num = spikingTransitionMatrix.shape[0]
            self.neuron_num = spikingTransitionMatrix.shape[1]
        else:
            self.rule_num = len(spikingTransitionMatrix)
            self.neuron_num = len(configurationVector)
        
        self.max_steps = max_steps
        self.deterministic = deterministic
        self.testsize = testsize
        
        # Input/Output neurons
        self.input_neurons = torch.as_tensor(
            input_neurons if input_neurons is not None else [], 
            dtype=torch.int64, device=self.device
        )
        self.output_neurons = torch.as_tensor(
            output_neurons if output_neurons is not None else [], 
            dtype=torch.int64, device=self.device
        )
        
        # Pooling image
        if output_neurons is not None and len(output_neurons) > 0:
            self.pooling_image = torch.zeros(
                (len(output_neurons), testsize), 
                dtype=torch.int64, device='cpu'
            )
        else:
            self.pooling_image = None
        
        # Conversione tensori principali
        self.configurationVector = self._to_tensor(configurationVector, self.dtype)
        self.ruleVector = self._to_tensor(ruleVector, self.dtype)
        self.applyingRuleVector = self._to_tensor(applyingRuleVector, torch.int32)
        
        # Gestione matrici (sparse o dense)
        if self._is_sparse:
            # Già tensori PyTorch sparse
            self.spikingTransitionMatrix = spikingTransitionMatrix.to(device=self.device)
            self.synapsesMatrix = synapsesMatrix.to(device=self.device)
            
            # Calcola sMpi per tensori sparsi
            if torch.equal(self.spikingTransitionMatrix._indices(), self.synapsesMatrix._indices()):
                # Stessa struttura sparsa: moltiplica solo i valori
                self.sMpi = torch.sparse_coo_tensor(
                    self.spikingTransitionMatrix._indices(),
                    self.spikingTransitionMatrix._values() * self.synapsesMatrix._values(),
                    self.spikingTransitionMatrix.shape
                ).coalesce()
            else:
                # Struttura diversa: converti a denso
                print("Warning: Different sparse structures, converting to dense")
                stm_dense = self.spikingTransitionMatrix.to_dense()
                sm_dense = self.synapsesMatrix.to_dense()
                self.sMpi = torch.mul(stm_dense, sm_dense)
                self._is_sparse = False
        else:
            # Array NumPy o tensori densi
            self.spikingTransitionMatrix = self._to_tensor(spikingTransitionMatrix, self.dtype)
            self.synapsesMatrix = self._to_tensor(synapsesMatrix, self.dtype)
            self.sMpi = torch.mul(self.spikingTransitionMatrix, self.synapsesMatrix)
        
        # Pre-calcolo indici per extended configuration
        self.ruleCountPerNeuron = torch.bincount(
            self.applyingRuleVector, minlength=self.neuron_num
        )
        self.repeat_indices = torch.repeat_interleave(
            torch.arange(self.neuron_num, device=self.device), 
            self.ruleCountPerNeuron
        ).contiguous()
        
        # Buffer pre-allocati
        self._extended_config = torch.empty(self.rule_num, dtype=self.dtype, device=self.device)
        self._diff_buffer = torch.empty(self.rule_num, dtype=self.dtype, device=self.device)
        self.netGainVector = torch.zeros(self.neuron_num, dtype=self.dtype, device=self.device)
        
        # Spiking vector
        self.spikingVector = self._to_tensor(
            spikingVector if spikingVector is not None else np.zeros(self.rule_num, dtype=np.int32), 
            self.dtype
        )
        
        # Spike train
        if single_spike_train is not None:
            self.single_spike_train = self._to_tensor(single_spike_train, self.dtype)
            self._input_len = len(single_spike_train) if hasattr(single_spike_train, '__len__') else single_spike_train.shape[0]
        else:
            self.single_spike_train = torch.zeros(0, dtype=self.dtype, device=self.device)
            self._input_len = 0
        
        # Stato
        self.t_step = 0
        self._cnn_mode = (Config.MODE == "CNN")
        self.img_spike_train = None
        
        # Pooling range
        if self.pooling_image is not None:
            self._pooling_start = Config.NUM_LAYERS - 2
            self._pooling_end = Config.NUM_LAYERS - 2 + self.testsize
        
        # Disabilita controlli sparse invariants (API corretta)
        if self._is_sparse:
            try:
                # Nuova API (PyTorch >= 2.0)
                torch.sparse.check_sparse_tensor_invariants.enable(False)
            except TypeError:
                # Vecchia API
                try:
                    torch.sparse.set_sparse_tensor_invariant_checks(False)
                except AttributeError:
                    pass
    
    def _to_tensor(self, data, dtype):
        """Converte dati in tensore PyTorch, gestendo più formati"""
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype, device=self.device).contiguous()
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(dtype=dtype, device=self.device).contiguous()
        else:
            return torch.as_tensor(data, dtype=dtype, device=self.device).contiguous()
    
    def loadImages(self, img_spike_train):
        """Carica immagini come spike train"""
        if len(img_spike_train.shape) == 3:
            img_spike_train = img_spike_train.reshape(img_spike_train.shape[0], -1)
        self.img_spike_train = self._to_tensor(img_spike_train, self.dtype)
        self._input_len = img_spike_train.shape[0]
    
    @torch.no_grad()
    def step(self, verbose=False):
        """Singolo passo di esecuzione ottimizzato"""
        
        # 1. Gestione input
        if self._cnn_mode and self.t_step < self._input_len:
            if self.input_neurons.numel() > 0:
                self.configurationVector.index_add_(
                    0, 
                    self.input_neurons, 
                    self.img_spike_train[self.t_step].to(dtype=self.dtype)
                )
        elif not self._cnn_mode and self.t_step < self._input_len:
            if self.input_neurons.numel() > 0 and self.single_spike_train[self.t_step] == 1:
                ones = torch.ones(len(self.input_neurons), dtype=self.dtype, device=self.device)
                self.configurationVector.index_add_(0, self.input_neurons, ones)
        
        # 2. Extended configuration vector
        torch.index_select(
            self.configurationVector, 0, self.repeat_indices, 
            out=self._extended_config
        )
        
        # 3. Calcolo spiking vector (ottimizzato)
        if self.dtype == torch.int32:
            # Versione intera
            torch.sub(self._extended_config, self.ruleVector, out=self._diff_buffer)
            self._diff_buffer.clamp_(min=0)
            self._diff_buffer.add_(1)
            torch.div(1, self._diff_buffer, rounding_mode='floor', out=self.spikingVector)
        else:
            # Versione float
            torch.ge(self._extended_config, self.ruleVector, out=self.spikingVector)
            self.spikingVector.mul_(1.0)
        
        # 4. Net gain vector
        # CORREZIONE: sMpi ha forma (rule_num, neuron_num)
        # spikingVector ha forma (rule_num,)
        # netGainVector deve avere forma (neuron_num,)
        # Quindi: netGainVector = spikingVector @ sMpi  (1 x rule_num @ rule_num x neuron_num = 1 x neuron_num)
        if self._is_sparse:
            # Per sparse: mv fa M @ v, quindi dobbiamo trasporre
            # sMpi.t() ha forma (neuron_num, rule_num)
            # spikingVector ha forma (rule_num,)
            # mv(sMpi.t(), spikingVector) -> (neuron_num,)
            torch.mv(self.sMpi.t(), self.spikingVector, out=self.netGainVector)
        else:
            # Versione densa: spikingVector @ sMpi
            torch.mv(self.sMpi.t(), self.spikingVector, out=self.netGainVector)
        
        # 5. Aggiornamento configurazione
        self.configurationVector.add_(self.netGainVector)
        
        # 6. Pooling
        if self.pooling_image is not None and self.output_neurons.numel() > 0:
            idx = self.t_step - self._pooling_start
            if 0 <= idx < self._pooling_end:
                self.pooling_image[:, idx] = self.configurationVector[self.output_neurons].cpu()
        
        self.t_step += 1
        return True
    
    def execute(self, verbose=False, startAgain=True):
        """Esecuzione completa"""
        if startAgain:
            self.t_step = 0
        
        input_length = self._input_len
        
        while self.t_step < self.max_steps:
            self.step(verbose=False)
            
            # Halt condition ottimizzata
            if self.t_step >= input_length and not self.spikingVector.any():
                if verbose:
                    print(f"Computation halts at step {self.t_step}: input accepted")
                return True
        
        if verbose:
            print(f"Computation halts at step {self.t_step}: max steps reached")
        return False
    
    def get_configuration_vector(self):
        return self.configurationVector.cpu().numpy()
    
    def get_spiking_vector(self):
        return self.spikingVector.cpu().numpy()
    
    def get_net_gain_vector(self):
        return self.netGainVector.cpu().numpy()
    
    def get_spiking_transition_matrix(self):
        if self._is_sparse:
            return self.sMpi.to_dense().cpu().numpy()
        return self.sMpi.cpu().numpy()
    
    def get_rule_vector(self):
        return self.ruleVector.cpu().numpy()
    
    def get_applying_rule_vector(self):
        return self.applyingRuleVector.cpu().numpy()
    
    def to(self, device):
        """Sposta su altro dispositivo"""
        new_device = torch.device(device)
        if new_device != self.device:
            # Sposta tutti i tensori
            for attr in ['configurationVector', 'ruleVector', 'applyingRuleVector',
                        'spikingVector', 'single_spike_train', 'netGainVector',
                        'repeat_indices', '_extended_config', '_diff_buffer',
                        'input_neurons', 'output_neurons']:
                if hasattr(self, attr):
                    tensor = getattr(self, attr)
                    if tensor is not None:
                        setattr(self, attr, tensor.to(new_device))
            
            if self._is_sparse:
                self.spikingTransitionMatrix = self.spikingTransitionMatrix.to(new_device)
                self.synapsesMatrix = self.synapsesMatrix.to(new_device)
                self.sMpi = self.sMpi.to(new_device)
            else:
                self.spikingTransitionMatrix = self.spikingTransitionMatrix.to(new_device)
                self.synapsesMatrix = self.synapsesMatrix.to(new_device)
                self.sMpi = self.sMpi.to(new_device)
            
            self.device = new_device
        return self
    
    def __str__(self):
        matrix_type = "sparse" if self._is_sparse else "dense"
        return (f"MSNPSystemGPU(device={self.device}, dtype={self.dtype}, "
                f"format={matrix_type}, rules={self.rule_num}, "
                f"neurons={self.neuron_num})")