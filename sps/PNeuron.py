"""P Neuron"""
from sps import Config

class PNeuron:
    """
    each neuron has a unique nid, connects to other neurons (targets),and follows a set of charge transformation rules
    """
    nid = 0

    @staticmethod
    def get_nid():
        return PNeuron.nid

    @staticmethod
    def increment_nid():
        PNeuron.nid += 1

    @staticmethod
    def reset_nid():
        PNeuron.nid = 0

    def __init__(self, snp_system, charge, targets, transf_rules, neuron_type=1):
        self.nid = PNeuron.get_nid()
        PNeuron.increment_nid()
        self.snp_system = snp_system
        self.targets = targets
        self.charge = charge
        self.refractory = 0 # can not fire or receive outside spikes between firing at t0 and spiking at t0+delay
        self.transf_rules = transf_rules
        self.neuron_type = neuron_type # can be 0 (input) - 1 (intermediate) - 2 (output)

    def receive(self, charge):
        if self.refractory == 0: # only receive input if outside the refractory period
            self.charge += charge
            self.snp_system.spike_fired += 1

    def inhibit(self, charge):
        if self.refractory == 0: # only receive input if outside the refractory period
            self.charge -= charge
            self.snp_system.inhibition_fired += 1

    def tick(self):
        """Tick and apply one transformation rule (either fire or consume), if possible."""
        # If the neuron is still in its refractory period, decrease the counter and skip this tick
        if self.refractory > 0:
            self.refractory = self.refractory -1
            return None

        # Rule are applied in order, so the priority is intrinsic and there isn't nondeterminism
        idxs = list(range(len(self.transf_rules)))
        #random.shuffle(idxs) # randomly shuffle the available transformation rules, firing and forgetting

        # Iterate through the shuffled rules and apply the first one that is valid
        for idx in idxs:
            rule = self.transf_rules[idx]
            if rule.check(self.charge):
                if rule.target > 0:
                    return self.fire(rule)
                else:
                    return self.consume(rule)

    def fire(self, rule):
        if Config.NEURONS_LAYER1 <= self.nid < Config.NEURONS_LAYER1_2:
            self.snp_system.layer_2_firing_counts[self.nid - Config.NEURONS_LAYER1] += 1 # for rules tuning
        if rule.source != 0:
            self.charge = self.charge - rule.source
        else:
            self.charge = 0 #TODO this is not theoretical possible
        self.refractory = rule.delay
        self.snp_system.firing_applied += 1
        return rule

    def consume(self, rule):
        if Config.NEURONS_LAYER1_2 <= self.nid < Config.NEURONS_TOTAL: # output array with predictions
            self.snp_system.output_array[self.snp_system.t_step][self.nid - Config.NEURONS_LAYER1_2] = self.charge
        self.charge = 0
        self.snp_system.forgetting_applied += 1
        return rule

    def __str__(self):
        type_map = {0: "Input", 1: "Intermediate", 2: "Output"}
        neuron_type = type_map.get(getattr(self, "neuron_type", 1), "Unknown")
        info = f"Neuron ID: {getattr(self, 'nid', '?')}\n"
        info += f"  Type: {neuron_type}\n"
        info += f"  Initial Charge: {self.charge}\n"
        info += f"  Output Targets: {self.targets}\n"
        info += f"  Rules ({len(self.transf_rules)}):\n"
        for i, rule in enumerate(self.transf_rules):
            info += f"    Rule {i+1}: div={rule.div}, mod={rule.mod}, source={rule.source}, target={rule.target}, delay={rule.delay}\n"
        return info