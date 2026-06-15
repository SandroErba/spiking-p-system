import numpy as np
import torch
from sps.config import Config
from sps.timersnp import TimerSNP

class MSNPSystemExactGPU:

    def __init__(self, configurationVector, spikingVector, sMpi_sparse,
                 ruleVector, max_steps=1000, deterministic=True,
                 single_spike_train=None, input_neurons=None, output_neurons=None,
                 applyingRuleVector=None, device='cpu', testsize=1):

        if device == 'gpu' or device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.dtype = torch.float32
                print(f"MSNPSystemGPU using device: CUDA GPU with dtype={self.dtype}")
            else:
                print("Warning: CUDA not available. Falling back to CPU.")
                self.device = torch.device('cpu')
                self.dtype = torch.int32
                print(f"MSNPSystemGPU using device: CPU with dtype={self.dtype}")
        else:
            self.device = torch.device('cpu')
            self.dtype = torch.int32
            print(f"MSNPSystemGPU using device: CPU with dtype={self.dtype}")

        if applyingRuleVector is None or configurationVector is None or ruleVector is None:
            raise ValueError("ApplyingRuleVector, configurationVector and ruleVector cannot be None")

        if max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

        #rule_num = len(spikingTransitionMatrix)
        rule_num = sMpi_sparse.shape[0]
        neuron_num = len(configurationVector)

        self.testsize = testsize
        self.pooling_image = torch.zeros((len(output_neurons), self.testsize), dtype=self.dtype, device='cpu') if output_neurons is not None else None

        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.max_steps = max_steps
        self.deterministic = deterministic

        self.timer = TimerSNP(self.max_steps,"time_MSNPSystem.csv")

        self.configurationVector = torch.tensor(configurationVector, dtype=self.dtype, device=self.device)
        #self.spikingTransitionMatrix = torch.tensor(spikingTransitionMatrix, dtype=self.dtype, device=self.device)
        #self.synapsesMatrix = torch.tensor(synapsesMatrix, dtype=self.dtype, device=self.device)
        self.netGainVector = torch.zeros(neuron_num, dtype=self.dtype, device=self.device)
        self.ruleVector = torch.tensor(ruleVector, dtype=self.dtype, device=self.device)
        self.applyingRuleVector = torch.tensor(applyingRuleVector, dtype=torch.int32, device=self.device)

        #self.sMpi = self.spikingTransitionMatrix * self.synapsesMatrix
        self.sMpi = sMpi_sparse.to(self.device)

        self.ruleCountPerNeuron = torch.bincount(self.applyingRuleVector, minlength=neuron_num)

        if spikingVector is None:
            self.spikingVector = torch.zeros(rule_num, dtype=self.dtype, device=self.device)
        else:
            self.spikingVector = torch.tensor(spikingVector, dtype=self.dtype, device=self.device)

        if single_spike_train is not None:
            self.single_spike_train = torch.tensor(single_spike_train, dtype=self.dtype, device=self.device)
        else:
            self.single_spike_train = torch.tensor([], dtype=self.dtype, device=self.device)

        self.t_step = 0
        self._debug_mode = None  # set to "TRAIN" or "TEST" in network.py before execute()

    def loadImages(self, img_spike_train):
        """Load images as spike trains for CNN mode"""
        if len(img_spike_train.shape) == 3:
            img_spike_train = img_spike_train.reshape(img_spike_train.shape[0], -1)
        self.img_spike_train = torch.tensor(img_spike_train, dtype=self.dtype, device=self.device)

    def step(self, verbose=False):
        """Execute one step of the system"""

        self.timer.start_step(self.t_step)
        # 1. Input
        if Config.MODE == "CNN":
            if self.t_step < self.img_spike_train.shape[0]:
                self.configurationVector[self.input_neurons] += self.img_spike_train[self.t_step]
        elif self.single_spike_train.size(0) > 0 and self.t_step < self.single_spike_train.shape[0]:
            spike_value = 1 if self.dtype == torch.int32 else 1.0
            if self.single_spike_train[self.t_step] == spike_value:
                self.configurationVector[self.input_neurons] += spike_value

        # 2. Extended config vector
        extendedConfigVector = torch.zeros_like(self.spikingVector, dtype=self.dtype, device=self.device)
        idx = 0
        for i in range(len(self.configurationVector)):
            count = self.ruleCountPerNeuron[i].item()
            extendedConfigVector[idx:idx+count] = self.configurationVector[i]
            idx += count

        # 3. Spiking vector
        diff = torch.abs(extendedConfigVector - self.ruleVector)
        self.spikingVector = torch.div(1, 1 + diff, rounding_mode='floor') if self.dtype == torch.int32 \
            else torch.floor(1.0 / (1.0 + diff))

        # Debug (first 5 steps only)
        #self._print_step_debug()

        # 4. Update configuration
        #self.netGainVector = self.spikingVector @ self.sMpi #dense
        self.netGainVector = torch.mv(self.sMpi.t(), self.spikingVector.float()).to(self.dtype) #sparse method
        self.configurationVector = self.configurationVector + self.netGainVector



        # 5. Save pooling
        if self.pooling_image is not None:
            # Nel test (10 classi) la propagazione richiede 1 step in più
            offset = 1 if len(self.output_neurons) == Config.CLASSES else 0

            if Config.NUM_LAYERS - 4 < self.t_step - offset <= self.testsize + Config.NUM_LAYERS - 4:
                col = (self.t_step - offset) - Config.NUM_LAYERS + 3
                self.pooling_image[:, col] = self.configurationVector[self.output_neurons]
                if len(self.output_neurons) == Config.CLASSES:
                    self.configurationVector[self.output_neurons] = 0


        self.timer.end_step()
        self.t_step += 1
        return True

    def execute(self, verbose=False, startAgain=True):
        if startAgain:
            self.t_step = 0

        if Config.MODE == "CNN":
            input_length = self.img_spike_train.shape[0]
        else:
            input_length = len(self.single_spike_train) if hasattr(self.single_spike_train, '__len__') else 0

        while self.step(verbose=verbose) and (self.t_step < self.max_steps or self.t_step < input_length):
            if torch.all(self.spikingVector == 0) and (self.t_step >= input_length):
                print("Computation halts: spiking vector is zero, input is accepted")
                np.save("/tmp/charge_map_gpu.npy", self.pooling_image.cpu().numpy())
                print(f"Saved charge_map_gpu: {self.pooling_image.shape}")
                self.timer.export_to_csv()
                return True

        self.timer.export_to_csv()
        print("Computation halts: maximum number of steps reached, input is rejected")
        return False

    def get_configuration_vector(self):
        return self.configurationVector.cpu().numpy()

    def get_spiking_vector(self):
        return self.spikingVector.cpu().numpy()

    def get_net_gain_vector(self):
        return self.netGainVector.cpu().numpy()

    def get_spiking_transition_matrix(self):
        return self.sMpi.cpu().numpy()

    def get_rule_vector(self):
        return self.ruleVector.cpu().numpy()

    def get_applying_rule_vector(self):
        return self.applyingRuleVector.cpu().numpy()

    def to(self, device):
        new_device = torch.device(device)
        new_dtype = torch.float32 if new_device.type == 'cuda' else torch.int32

        if new_dtype != self.dtype:
            self.dtype = new_dtype
            self.configurationVector = self.configurationVector.to(dtype=self.dtype, device=new_device)
            self.spikingTransitionMatrix = self.spikingTransitionMatrix.to(dtype=self.dtype, device=new_device)
            self.synapsesMatrix = self.synapsesMatrix.to(dtype=self.dtype, device=new_device)
            self.netGainVector = self.netGainVector.to(dtype=self.dtype, device=new_device)
            self.ruleVector = self.ruleVector.to(dtype=self.dtype, device=new_device)
            self.sMpi = self.sMpi.to(dtype=self.dtype, device=new_device)
            self.spikingVector = self.spikingVector.to(dtype=self.dtype, device=new_device)
            self.single_spike_train = self.single_spike_train.to(dtype=self.dtype, device=new_device)
            if hasattr(self, 'img_spike_train'):
                self.img_spike_train = self.img_spike_train.to(dtype=self.dtype, device=new_device)
        else:
            self.configurationVector = self.configurationVector.to(new_device)
            self.spikingTransitionMatrix = self.spikingTransitionMatrix.to(new_device)
            self.synapsesMatrix = self.synapsesMatrix.to(new_device)
            self.netGainVector = self.netGainVector.to(new_device)
            self.ruleVector = self.ruleVector.to(new_device)
            self.sMpi = self.sMpi.to(new_device)
            self.spikingVector = self.spikingVector.to(new_device)
            self.single_spike_train = self.single_spike_train.to(new_device)
            if hasattr(self, 'img_spike_train'):
                self.img_spike_train = self.img_spike_train.to(new_device)

        self.applyingRuleVector = self.applyingRuleVector.to(new_device)
        self.device = new_device
        return self

    def __str__(self):
        return (f"Device: {self.device} (dtype: {self.dtype})\n"
                f"Deterministic: {self.deterministic}\n"
                f"Input Neurons: {self.input_neurons}\n"
                f"Output Neurons: {self.output_neurons}\n"
                f"Configuration Vector: {self.get_configuration_vector()}\n"
                f"Spiking Vector: {self.get_spiking_vector()}\n"
                f"Net Gain Vector: {self.get_net_gain_vector()}\n"
                f"Rule Vector: {self.get_rule_vector()}\n"
                f"Applying Rule Vector: {self.get_applying_rule_vector()}\n")
