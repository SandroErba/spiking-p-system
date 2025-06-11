"""Spiking Neural P System"""
from .PNeuron import PNeuron
from .SpikeUtils import SpikeEvent, TransformationRule, History
import csv

class SNPSystem:
    """Spiking Neural P System"""
    def __init__(self, max_delay, max_steps, input_type):
        # init time step, history
        self.t_step = 0
        self.max_steps = max_steps
        self.input_type = input_type #generative, binary_spike_train, 8x8_spike_train
        self.history = None

        # init circular future spiking events based on max_delay
        self.max_delay = max_delay
        self.spike_events = [[] for x in range(self.max_delay)]

        # init neuron container
        self.neurons = []

        # record output
        self.output = []

    def init_history(self):
        """init tick history based on the system's neurons"""
        self.history = History(self.neurons)

    def start(self):
        """start sending and receiving spikes"""
        # init history
        self.init_history()
        # keep on ticking until output condition is met or max number of ticks is exceeded
        while True:
            if not self.tick():
                break

    def result(self):
        """system output as number of total ticks between 1st and 2nd spike of the output neuron"""
        return self.output[1] - self.output[0]

    def tick(self):
        """at each time step, first evolve and then receive spikes, cant do both in the same step, refractory will prevent it"""
        self.history.add_new_tick()

        any_rule_applied = False
        # evolve each neuron
        for neuron in self.neurons:
            used_rule = neuron.tick()
            if used_rule:
                any_rule_applied = True

            # fire event
            if used_rule and used_rule.target > 0:
                # Generate a firing event that will be received in the future, if it has delay
                self.spike_events[(self.t_step + used_rule.delay) % self.max_delay].append(SpikeEvent(neuron.nid, used_rule.target, neuron.targets))
                # record output if neuron belongs to output
                if neuron.neuron_type == 2:
                    self.output.append(self.t_step)
            self.history.record_rule(neuron, used_rule)

        input_spike = False
        if hasattr(self, "spike_train"):
            if self.t_step < len(self.spike_train):
                input_spike = True
                if self.input_type == "8x8_spike_train": # You have some 8x8 images as input
                    input_vector = self.spike_train[self.t_step] # input_vector should be a list with len = input neurons
                    for i, neuron in enumerate(self.neurons):
                        if neuron.neuron_type == 0:  # neurone input
                            if input_vector[i] == 1:
                                neuron.charge += 1
                                self.history.record_incoming(neuron, 1, "input")
                if (self.input_type == "binary_spike_train") and self.spike_train[self.t_step] == 1: # You have one spike train for all the input neurons
                    for neuron in self.neurons:
                        if neuron.neuron_type == 0:
                            neuron.charge += 1
                            self.history.record_incoming(neuron, 1, "input")


        # consume current spiking events
        for spike_event in self.spike_events[self.t_step % self.max_delay]:
            for idx in spike_event.targets:
                self.neurons[idx].receive(spike_event.charge) # A neuron can receive more than 1 spike at time only from different input neurons
                self.history.record_incoming(self.neurons[idx], spike_event.charge, spike_event.nid)

        # clear current spiking events
        self.spike_events[self.t_step % self.max_delay].clear()

        if self.max_steps == self.t_step:
            print("Time limit reached, the computation halts")

        #check for halting computation
        any_in_delay = any(n.refractory > 0 for n in self.neurons)
        any_spike_in_transit = any(self.spike_events[i] for i in range(self.max_delay))

        if not any_rule_applied and not any_in_delay and not any_spike_in_transit and not input_spike and self.t_step > 1:
            print("System halts at tick", self.t_step)
            return False
        # advance time
        self.t_step += 1
        # exit if closing condition is met, otherwise continue
        return False if (len(self.output) == 2 or self.t_step > self.max_steps) else True

    def load_neurons_from_csv(self, filename):
        """Read a CSV file and create the corrisponding SNPS"""
        neurons = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                if not any(row):
                    break

                initial_charge = int(row[1])
                targets = eval(row[2])  # example: "[1,3]" → [1,3]
                neuron_type = int(row[3])

                # Read rules
                transf_rules = []
                for cell in row[4:]:
                    if not cell.strip():
                        break
                    try:
                        values = eval(cell)
                        if values[0] == 0:
                            values[0] = 999
                        if len(values) != 5:
                            raise ValueError(f"Wrong rules: {cell}")
                        rule = TransformationRule(values[0], values[1],values[2],values[3],values[4])
                        transf_rules.append(rule)
                    except Exception as e:
                        print(f"Error during rule reading {cell}: {e}")

                # Create neuron
                neuron = PNeuron(charge = initial_charge, targets=targets, transf_rules=transf_rules, neuron_type=neuron_type)
                #print(neuron)
                neurons.append(neuron)
        self.neurons = neurons
        return neurons