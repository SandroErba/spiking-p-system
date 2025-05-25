"""Spike related utils classes"""
from texttable import Texttable

class SpikeEvent:
    """contains originator neuron, charge and target neurons"""
    def __init__(self, nid, charge, targets):
        self.nid = nid
        self.charge = charge
        self.targets = targets

class TransformationRule:
    """
    E1/a^r1->a;t1
    E2/a^r2->a;t2
    a+/a->a | a^k -> a | ...
    a^2 exact condition, a^2n+1 | a^2n condition, a^+ condition
    # match condition:
        - exact, div = 1, mod = 0
        - match, an+b, div = a, mod = b
    # target
        - value = 0, forgetting rule
        - value = x, firing rule to x neurons
    # source is always substracted from current charge
    """
    def __init__(self, div, mod, source, target, delay):
        self.div = div
        self.mod = mod
        self.source = source
        self.target = target
        self.delay = delay

    def check(self, charge):
        # with div and mod is possible to manage odd, even, and all value condition for "charge"
        return charge > 0 and charge % self.div == self.mod

    def exec(self, charge):
        return charge -self.source

    def __str__(self):
        # Build the condition part of the rule, e.g., "2a+1" if div=2 and mod=1
        # This describes when the rule is applicable based on charge (e.g., odd/even)
        rulecond = "{0}a{1}".format(self.div, "+{0}".format(self.mod) if self.mod > 0 else "")

        # Build the transformation part: "source->target"
        # If target > 0, append "!" to indicate a firing rule (sends spike)
        # If target == 0, it's a forgetting rule (consumes charge only)
        ruletransf = "{0}->{1}{2}".format(self.source, self.target, "!" if self.target > 0 else "")

        # Final string format: condition;transformation;delay
        # Example: "2a+1;3->1!;2"
        return "{0};{1};{2}".format(rulecond, ruletransf, self.delay)

class History:
    """
    responsible for recording each neuron status at each step
    takes of advantage of neuron ids starting from 0
    """
    def __init__(self, neurons):
        self.ticks = [] # every element is the state of a neuron
        self.n_len = len(neurons)

        # add initial charge
        self.add_new_tick()
        for neuron in neurons:
            self.ticks[-1][neuron.nid] = neuron.charge

    def add_new_tick(self):
        self.ticks.append([""] * self.n_len)

    def record_rule(self, neuron, used_rule):
        self.ticks[-1][neuron.nid] = "r:{0}\nc:{1}".format(str(used_rule) if used_rule else "-", neuron.charge)

    def record_incoming(self, neuron, charge, source_nid):
        self.ticks[-1][neuron.nid] += "\ni:{0}({2})\nc:{1}".format(charge, neuron.charge, source_nid)

    # Create and populate output table
    def __str__(self):
        ticks_wrapper = []
        for idx, tick in enumerate(self.ticks):
            pref = idx - 1 if idx > 0 else "initial\ncharge"
            ticks_wrapper.append([pref] + tick)

        table = Texttable()
        table.set_cols_width([7] + [20] * len(self.ticks[0]))
        table.header(["Step"] + ["Neuron " + str(nid) for nid in range(len(self.ticks[0]))])
        table.add_rows(ticks_wrapper[:], header=False)

        return str(table.draw())
        