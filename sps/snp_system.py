import numpy as np
from sps.config import Config

from .p_neuron import PNeuron
from .perceptron import OnlineDiscretePerceptron
from .spike_utils import SpikeEvent, TransformationRule, History
import csv


class SNPSystem:
    """Spiking Neural P System"""

    def __init__(self, max_delay, max_steps, deterministic):

        PNeuron.reset_nid()
        self.t_step = 0
        self.max_steps = max_steps
        self.deterministic = deterministic # can be true or false

        self.history = None


        self.max_delay = max_delay # init circular future spiking events based on max_delay
        self.spike_events = [[] for _ in range(self.max_delay)]

        self.neurons = []
        self.spike_train = None

        # record output
        if Config.MODE == "generative":
            self.output = [] # time between two spikes in the output neuron
        elif Config.MODE == "edge":
            self.feature_image = np.zeros((Config.SHAPE_FEATURE * Config.SHAPE_FEATURE, Config.TRAIN_SIZE), dtype=int)
        elif Config.MODE == "cnn":
            self.feature_image = np.zeros((Config.NEURONS_FEATURE * Config.KERNEL_NUMBER, Config.TRAIN_SIZE), dtype=int)
            self.pooling_image = np.zeros((Config.NEURONS_L3, Config.TRAIN_SIZE), dtype=int)
            self.charge_map_pool = np.zeros(Config.NEURONS_L3, dtype=int) #values for the third layer of the cnn #TODO substitute with charge_pooling_image[t_step]
            self.model = OnlineDiscretePerceptron(Config.NEURONS_L3, Config.CLASSES, Config.LR, 1e-5) #sparsity = % of pruned synapses
            self.labels = []
            self.correct = 0
            self.charge_map_prediction = np.zeros((Config.CLASSES, Config.TEST_SIZE), dtype=int) #values for the last layer of the cnn


        self.firing_applied = 0 # record energy used
        self.forgetting_applied = 0
        self.spike_fired = 0
        self.inhibition_fired = 0 #those need the energy_tracker

    def init_history(self):
        """init tick history based on the system's neurons"""
        self.history = History(self.neurons)

    def start(self):
        """start sending and receiving spikes"""
        self.init_history()
        while True: # keep on ticking until output condition is met or max number of ticks is exceeded
            if not self.tick():
                # calculate consumed energy
                """For more info, see latex document and chapter 5.5 of 'Beyond classification: directly training spiking
                neural networks for semantic segmentation' -> paper "https://arxiv.org/pdf/2110.07742" """
                total_synapses = sum(len(neuron.targets) for neuron in self.neurons)
                total_rules = sum(len(neuron.transf_rules) for neuron in self.neurons)
                w_energy = self.t_step * (total_synapses + total_rules * Config.WORST_REGEX)
                e_energy = int(self.spike_fired * Config.EXPECTED_SPIKE + (self.firing_applied + self.forgetting_applied) * Config.EXPECTED_REGEX)

                if Config.MODE == "generative":
                    print("Spike fired at time step", self.output[0], "and time step", self.output[1], ". The output is", self.output[1] - self.output[0])
                return w_energy, e_energy

    def tick(self):
        """at each time step, first evolve and then receive spikes, cant do both in the same step, refractory will prevent it"""
        self.history.add_new_tick()

        any_rule_applied = False
        # evolve each neuron
        for neuron in self.neurons:
            if Config.MODE == "cnn": # perceptron tuning for cnn
                pool_offset = Config.NEURONS_L1 + Config.NEURONS_L2
                if pool_offset <= neuron.nid < (pool_offset + Config.NEURONS_L3):
                    self.charge_map_pool[neuron.nid - pool_offset] = neuron.charge

            used_rule = neuron.tick()
            if used_rule:
                any_rule_applied = True
            # fire event
            if used_rule and used_rule.target > 0:
                # Generate a firing event that will be received in the future, if it has delay
                self.spike_events[(self.t_step + used_rule.delay) % self.max_delay].append(SpikeEvent(neuron.nid, used_rule.target, neuron.targets))
                if neuron.neuron_type == 2 and Config.MODE == "generative": # output neuron
                    self.output.append(self.t_step)
            self.history.record_rule(neuron, used_rule)

        # perceptron tuning for cnn
        if Config.MODE == "cnn":
            if self.t_step >= 3 and self.t_step - 3 < len(self.labels): #3 is the number of layer where the perceptron is applied
                self.model.update(self.charge_map_pool, self.labels[self.t_step - 3])

        input_spike = False # check if there are more input for halting condition
        if Config.MODE in ("edge", "cnn"): # you have an array of images as input
            if self.spike_train.any() and self.t_step < len(self.spike_train):
                input_spike = True
                input_vector = self.spike_train[self.t_step].flatten() # input_vector should be a list with len = input neurons
                for i, neuron in enumerate(self.neurons):
                    if neuron.neuron_type == 0:
                        if input_vector[i] > 0:
                            neuron.charge =  int(neuron.charge) + int(input_vector[i]) # add charge to the corresponding neuron
                            self.history.record_incoming(neuron, input_vector[i], "input")
        elif self.spike_train and self.t_step < len(self.spike_train):
            if self.spike_train[self.t_step] == 1: # one boolean spike train for all the input neurons
                input_spike = True
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

        # fill charge maps
        if Config.MODE == "cnn":
            if 0 < self.t_step <= len(self.spike_train):
                for input_id in range(Config.NEURONS_L2): #generate feature images
                    offset = input_id + Config.NEURONS_L1
                    self.feature_image[input_id][self.t_step - 1] = self.neurons[offset].charge
            if 1 < self.t_step <= len(self.spike_train) + 1:
                for input_id in range(Config.NEURONS_L3): #generate pooling images
                    offset = input_id + Config.NEURONS_L1 + Config.NEURONS_L2
                    self.pooling_image[input_id][self.t_step - 2] = self.neurons[offset].charge
            if 2 < self.t_step <= len(self.spike_train) + 2:
                if len(self.labels) == 0: #there is a better method?
                    for input_id in range(Config.CLASSES): #generate output charge
                        offset = input_id + Config.NEURONS_L1 + Config.NEURONS_L2 + Config.NEURONS_L3 - 1
                        self.charge_map_prediction[input_id][self.t_step - 3] = self.neurons[offset].charge

        # clear current spiking events
        self.spike_events[self.t_step % self.max_delay].clear()

        # set negative charge to 0 for the inhibitor spike
        for neuron in self.neurons:
            if neuron.charge < 0: #TODO implicit RELU function
                neuron.charge = 0

        # check for halting computation
        any_in_delay = any(n.refractory > 0 for n in self.neurons)
        any_spike_in_transit = any(self.spike_events[i] for i in range(self.max_delay))
        if Config.MODE == "generative" and len(self.output) == 2: # halt computation if output has fired 2 times
            return False
        if not any_rule_applied and not any_in_delay and not any_spike_in_transit and not input_spike and self.t_step > 1:
            if Config.MODE == "halting":
                print("The computation halts because no further rules can be applied; the input is accepted")
            return False # end computation

        self.t_step += 1 # advance time
        if self.t_step > self.max_steps:
            if Config.MODE == "halting":
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