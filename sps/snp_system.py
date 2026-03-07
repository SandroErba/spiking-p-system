import numpy as np
from sps.config import Config

from .p_neuron import PNeuron
from .spike_utils import SpikeEvent, TransformationRule, History
import csv

from .charge_tracker import ChargeTracker


class SNPSystem:
    """Spiking Neural P System"""

    def __init__(self, max_delay, max_steps, input_type, output_type, deterministic):
        if Config.MODE in ("binarized", "quantized"):
            self.charge_map_l1 = np.zeros(Config.NEURONS_L1, dtype=float) # support array - for saving and showing the internal charge
            self.charge_map_l2 = np.zeros(Config.NEURONS_L12 - Config.NEURONS_L1, dtype=float)
            self.charge_map_l3 = np.zeros(Config.NEURONS_L3 - Config.NEURONS_L12, dtype=float)
            self.charge_map_l4 = np.zeros(Config.NEURONS_T - Config.NEURONS_L3, dtype=float)

        # init time step, history
        PNeuron.reset_nid()
        self.t_step = 0
        self.max_steps = max_steps
        self.input_type = input_type # can be none, spike_train, images
        self.output_type = output_type # can be halting, generative, prediction, images
        self.deterministic = deterministic # can be true or false

        self.history = None

        # init circular future spiking events based on max_delay
        self.max_delay = max_delay
        self.spike_events = [[] for _ in range(self.max_delay)]

        # init neuron container
        self.neurons = []

        # record energy used
        self.firing_applied = 0
        self.forgetting_applied = 0
        self.spike_fired = 0
        self.inhibition_fired = 0

        self.spike_train = None

        if Config.MODE in ("binarized", "quantized"): # record firing of layer 2 for the training phase
            self.labels = []
            self.old_layer_2_firing_counts = 0
            self.layer_2_firing_counts = 0
            self.layer_2_synapses = []
            # structure to memorize layer 3 synampses' weights for training
            self.layer_3_firing_counts = np.zeros(Config.NEURONS_L3 - Config.NEURONS_L12, dtype=int)
            self.old_layer_3_firing_counts = 0
            self.layer_3_synapses = []

        # record output
        if output_type == "generative": #TODO continue to delete output type
            self.output = [] # time between two spikes in the output neuron
        elif Config.MODE in ("binarized", "quantized"):
            self.output_array = np.zeros((self.max_steps, Config.CLASSES), dtype=int) # array of prediction

        if Config.MODE == "edge":
            self.feature_image = np.zeros((Config.SHAPE_FEATURE * Config.SHAPE_FEATURE, Config.TRAIN_SIZE), dtype=int)
        if Config.MODE == "cnn":
            self.feature_image = np.zeros((Config.NEURONS_FEATURE * Config.KERNEL_NUMBER, Config.TRAIN_SIZE), dtype=int)
            self.pooling_image = np.zeros((Config.NEURONS_POOL * Config.KERNEL_NUMBER, Config.TRAIN_SIZE), dtype=int)

        self.charge_tracker = None # I initialize the charge tracker 
    def init_history(self):
        """init tick history based on the system's neurons"""
        self.history = History(self.neurons)

    def start(self):
        """start sending and receiving spikes"""
        self.init_history()
        self.charge_tracker = ChargeTracker(filename="output_charges", mode="all_at_once", format="parquet", num_neurons=len(self.neurons))
        # keep on ticking until output condition is met or max number of ticks is exceeded
        while True:
            self.charge_tracker.record_charges(self.t_step, self.neurons) #takes the index of the current image and reads the charge of each neuron and saves it in self.history.
            if not self.tick():
                # calculate consumed energy
                """For more info, see latex document and chapter 5.5 of 'Beyond classification: directly training spiking
                neural networks for semantic segmentation' -> paper "https://arxiv.org/pdf/2110.07742" """
                total_synapses = sum(len(neuron.targets) for neuron in self.neurons)
                total_rules = sum(len(neuron.transf_rules) for neuron in self.neurons)
                w_energy = self.t_step * (total_synapses + total_rules * Config.WORST_REGEX)
                e_energy = int(self.spike_fired * Config.EXPECTED_SPIKE + (self.firing_applied + self.forgetting_applied) * Config.EXPECTED_REGEX)
                self.charge_tracker.finish()

                if self.output_type == "generative":
                    print("Spike fired at time step", self.output[0], "and time step", self.output[1], ". The output is", self.output[1] - self.output[0])
                return w_energy, e_energy

    def tick(self):
        """at each time step, first evolve and then receive spikes, cant do both in the same step, refractory will prevent it"""
        self.history.add_new_tick()

        any_rule_applied = False
        # evolve each neuron
        for neuron in self.neurons:
            if Config.MODE in ("binarized", "quantized"): self.save_charge(self, neuron) #DEBUGGING ONLY - for saving the internal charge
            used_rule = neuron.tick()
            if used_rule:
                any_rule_applied = True
            # fire event
            if used_rule and used_rule.target > 0:
                # Generate a firing event that will be received in the future, if it has delay
                self.spike_events[(self.t_step + used_rule.delay) % self.max_delay].append(SpikeEvent(neuron.nid, used_rule.target, neuron.targets))
                if neuron.neuron_type == 2 and self.output_type == "generative": # output neuron
                    self.output.append(self.t_step)
            self.history.record_rule(neuron, used_rule)

        # Stampa solo se c'è attività nel Layer 3 (utile per capire la decisione)
        #if Config.MODE in ("binarized", "quantized") and np.any(self.charge_map_l3):
        #    print(f"\n=== T_STEP {self.t_step} ===")
        #    print(f"Attivazioni Layer 3 (Neuroni Intermedi):")
        #    print(self.charge_map_l3) # Stampa l'array dei tuoi 24 neuroni
            

        #self.show_charge() #debug only - for saving the internal charge

        input_spike = False # check if there are more input for halting condition
        if self.spike_train is not None and self.t_step < len(self.spike_train):
            input_spike = True
            if Config.MODE in ("binarized", "quantized", "edge", "cnn"): # you have an array of images as input
                input_vector = self.spike_train[self.t_step].flatten() # input_vector should be a list with len = input neurons
                for i, neuron in enumerate(self.neurons):
                    if neuron.neuron_type == 0:
                        if input_vector[i] > 0:
                            neuron.charge =  int(neuron.charge) + int(input_vector[i]) # add charge to the corresponding neuron
                            #self.spike_fired += input_vector[i]
                            self.history.record_incoming(neuron, input_vector[i], "input")
            elif self.spike_train[self.t_step] == 1: # one boolean spike train for all the input neurons
                for neuron in self.neurons:
                    if neuron.neuron_type == 0:
                        neuron.charge += 1
                        self.spike_fired += 1
                        self.history.record_incoming(neuron, 1, "input")

        # consume current spiking events
        for spike_event in self.spike_events[self.t_step % self.max_delay]:
            for idx in spike_event.targets:
                if idx >= 0:
                    #print("targets", spike_event.targets, "idx", idx)
                    self.neurons[idx].receive(spike_event.charge)
                else:
                    self.neurons[-idx].inhibit(spike_event.charge)
                self.history.record_incoming(self.neurons[idx], spike_event.charge, spike_event.nid)

        # create the output images
        if Config.MODE == "edge":
            for input_id in range(Config.SHAPE_FEATURE * Config.SHAPE_FEATURE):
                offset = input_id + Config.NEURONS_L1 + (Config.SHAPE_FEATURE * Config.SHAPE_FEATURE * Config.KERNEL_NUMBER)
                if self.neurons[offset].charge > 0:
                    self.feature_image[input_id][self.t_step - 2] = 1

        if Config.MODE == "cnn" and 0 < self.t_step <= len(self.spike_train):
            for input_id in range(Config.NEURONS_L2): #generate faeture images
                offset = input_id + Config.NEURONS_L1
                self.feature_image[input_id][self.t_step - 1] = self.neurons[offset].charge
            for input_id in range(Config.NEURONS_LP): #generate pooling images
                offset = input_id + Config.NEURONS_L1 + Config.NEURONS_L2
                self.pooling_image[input_id][self.t_step - 2] = self.neurons[offset].charge

        # clear current spiking events
        self.spike_events[self.t_step % self.max_delay].clear()

        # set negative charge to 0 for the inhibitor spike
        for neuron in self.neurons:
            if neuron.charge < 0:
                neuron.charge = 0

        # synapses tuning, enter only in the image classification mode
        if Config.MODE in ("binarized", "quantized") and len(self.layer_2_synapses) > 0:
            if Config.QUANTIZATION and np.any(self.charge_map_l2):
                #print("MATRICE LAYER 2: ", charge_map_l2)
                label = self.labels[self.t_step - 2] # -2 because the P system requires 2 step for start the computation
                for idx in range(Config.NEURONS_L2):
                    self.layer_2_synapses[label][idx] = self.layer_2_synapses[label][idx] + (self.charge_map_l2[idx] * (Config.CLASSES - 1))
                    for wrong_label in range(Config.CLASSES):
                        if wrong_label != label:
                            self.layer_2_synapses[wrong_label][idx] -= self.charge_map_l2[idx]

            elif not Config.QUANTIZATION:
                fired_diff = self.layer_2_firing_counts - self.old_layer_2_firing_counts
                fired_indices = np.where(fired_diff > 0)[0]  
                if fired_indices.size > 0:
                    l3_neurons_count = len(self.layer_2_synapses)
                    label = self.labels[self.t_step - 2]
                    target_l3_indices = [label]
                    for idx in fired_indices:
                        for l3_idx in range(l3_neurons_count):
                            if l3_idx in target_l3_indices:
                                self.layer_2_synapses[l3_idx][idx] += Config.POSITIVE_REINFORCE
                            else:
                                self.layer_2_synapses[l3_idx][idx] -= Config.NEGATIVE_PENALIZATION
                    self.old_layer_2_firing_counts = self.layer_2_firing_counts.copy()
            
        # synapses tuning for layer 3
        if self.input_type == "images" and self.output_type == "prediction" and len(self.layer_3_synapses) > 0:
            current_label_idx = self.t_step - 2
            if 0 <= current_label_idx < len(self.labels): 
                if Config.QUANTIZATION and np.any(self.charge_map_l3):
                    label = self.labels[current_label_idx]
                    num_l3_neurons = self.layer_3_synapses.shape[1] 
                    for idx in range(num_l3_neurons):
                        if self.charge_map_l3[idx] > 0:
                            self.layer_3_synapses[label][idx] += self.charge_map_l3[idx]
                            for wrong_label in range(Config.CLASSES):
                                if wrong_label != label:
                                    self.layer_3_synapses[wrong_label][idx] -= self.charge_map_l3[idx]


        # check for halting computation
        any_in_delay = any(n.refractory > 0 for n in self.neurons)
        any_spike_in_transit = any(self.spike_events[i] for i in range(self.max_delay))
        if self.output_type == "generative" and len(self.output) == 2: # halt computation if output has fired 2 times
            return False
        if not any_rule_applied and not any_in_delay and not any_spike_in_transit and not input_spike and self.t_step > 1:
            if self.output_type == "halting":
                print("The computation halts because no further rules can be applied; the input is accepted")
            return False # end computation

        self.t_step += 1 # advance time
        if self.t_step > self.max_steps:
            if self.output_type == "halting":
                print("The system did not halt naturally within the given step bound; the input is rejected")
            return False # end computation
        else:
            return True # continue computation

    def load_neurons_from_csv(self, filename):
        """Read a CSV file and create the corresponding SNPS"""
        neurons = []

        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                if not any(row):
                    break

                initial_charge = int(row[1])
                targets = eval(row[2])
                neuron_type = int(row[3])
                # Read rules
                transf_rules = []
                for cell in row[4:]:
                    if not cell.strip():
                        break
                    try:
                        values = eval(cell)
                        if len(values) != 5:
                            raise ValueError(f"Wrong rules: {cell}")
                        rule = TransformationRule(values[0],values[1],values[2],values[3],values[4])
                        transf_rules.append(rule)
                    except Exception as e:
                        print(f"Error during rule reading {cell}: {e}")

                # Create neuron
                neuron = PNeuron(snp_system=self, charge = initial_charge, targets=targets, transf_rules=transf_rules, neuron_type=neuron_type)
                neurons.append(neuron)
        self.neurons = neurons
        return neurons




    # support array - for saving and showing the internal charge
    @staticmethod
    def save_charge(self, neuron):
        nid = neuron.nid
        if 0 <= nid < Config.NEURONS_L1:
            self.charge_map_l1[nid] = neuron.charge
            return
        l2_index = nid - Config.NEURONS_L1
        if 0 <= nid < Config.NEURONS_L12:
            self.charge_map_l2[l2_index] = neuron.charge
            return
        l3_index = nid - Config.NEURONS_L12
        if 0 <= nid < Config.NEURONS_L3:
            self.charge_map_l3[l3_index] = neuron.charge
            return
        l4_index = nid - Config.NEURONS_L3
        if 0 <= nid < Config.NEURONS_T:
            self.charge_map_l4[l4_index] = neuron.charge
            return

    def show_charge(self):
        print("-------------MATRIX LAYER 1 at time step ", self.t_step, "------------------")
        print(self.charge_map_l1)
        print("-------------MATRIX LAYER 2 at time step ", self.t_step, "------------------")
        print(self.charge_map_l2)
        print("-------------MATRIX LAYER 3 at time step ", self.t_step, "------------------")
        print(self.charge_map_l3)
        print("-------------MATRIX LAYER 4 at time step ", self.t_step, "------------------")
        print(self.charge_map_l4)