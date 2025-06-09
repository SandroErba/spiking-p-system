"""P Neuron"""
import random

class PNeuron:
    """
    each neuron has a unique nid, 
    connects to other neurons (targets),
    follows a set of charge transformation rules, 
    can be an internal or output neuron with output neurons terminating the computation
    """
    nid = 0

    @staticmethod
    def get_nid():
        return PNeuron.nid

    @staticmethod
    def increment_nid():
        PNeuron.nid += 1

    def __init__(self, charge, targets, transf_rules, neuron_type=1):
        self.nid = PNeuron.get_nid()
        PNeuron.increment_nid()
        self.targets = targets
        self.charge = charge
        self.refractory = 0 # can not fire or receive outside spikes between firing at t0 and spiking at t0+delay
        self.transf_rules = transf_rules
        self.neuron_type = neuron_type #can be 0 (input) - 1 (intermediate) - 2 (output)

    def receive(self, charge):
        # only receive input if outside the refractory period
        if self.refractory == 0:
            self.charge += charge

    def tick(self):
        """Tick and apply one transformation rule (either fire or consume), if possible."""

        # If the neuron is still in its refractory period, decrease the counter and skip this tick
        if self.refractory > 0:
            self.refractory = self.refractory -1
            return None

        # In this code there is no priority - randomly shuffle the indices of the available transformation rules
        # This introduces nondeterminism in which rule is selected if multiple match
        idxs = list(range(len(self.transf_rules)))
        random.shuffle(idxs)

        # Iterate through the shuffled rules and apply the first one that is valid for the current charge
        for idx in idxs:
            rule = self.transf_rules[idx]
            if rule.check(self.charge):
                # rule.target > 0 indicates this is a firing rule (spike sent to targets)
                # rule.target <= 0 probably means it's a forgetting or charge-reducing rule
                if rule.target > 0:
                    return self.fire(rule)
                else:
                    return self.consume(rule)

    def fire(self, rule):
        self.charge = self.charge - rule.source
        self.refractory = rule.delay
        return rule

    def consume(self, rule):
        self.charge = self.charge - rule.source
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

