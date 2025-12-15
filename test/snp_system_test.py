import unittest
from unittest.mock import MagicMock

from sps.config import Config
from sps.p_neuron import PNeuron
from sps.snp_system import SNPSystem
from sps.spike_utils import TransformationRule

class SNPSystemTest(unittest.TestCase):

    def setUp(self):
        PNeuron.nid = 0

    def test_tick_neurons_not_triggering_rules(self):
        Config.MODE = "test"
        snps = SNPSystem(5, 100, "none", "halting", True)

        pn1 = PNeuron(snps, 0, targets=[], transf_rules=[])
        pn2 = PNeuron(snps, 0, targets=[], transf_rules=[])

        pn1.tick = MagicMock(return_value=None)
        pn2.tick = MagicMock(return_value=None)

        snps.neurons = [pn1, pn2]
        snps.init_history()

        # after a system tick where no neuron fired/consume a rule, return True in order to keep ticking
        self.assertEqual(snps.tick(), True)

        # each neuron was "ticked"
        pn1.tick.assert_called_with()
        pn2.tick.assert_called_with()


    def test_tick_output_neuron(self):
        Config.MODE = "test"
        """ the final system output should be [startTick, endTick] - the number of ticks passed between the first two firing of the output neuron"""
        snps = SNPSystem(5, 100, "none", "generative", True)
        
        p1 = PNeuron(snps, 0, targets=[], transf_rules=[])
        p_output = PNeuron(snps, 0, targets=[], transf_rules=[], neuron_type=2)

        p1.tick = MagicMock(return_value=None)
        output_fire_rule = TransformationRule(div=1, mod=0, source=5, target=1, delay=3)
        p_output.tick = MagicMock(return_value=output_fire_rule)

        snps.neurons = [p1, p_output]
        snps.init_history()

        # output neuron hasn't fired so far
        self.assertEqual(len(snps.output), 0)

        # after a system tick where the output neuron fired once, keep firing
        self.assertEqual(snps.tick(), True)

        # the current system output is [startTick] - recording the 1st firing of the output neuron
        self.assertEqual(len(snps.output), 1)

        # system will complete the computation, the output value is the number of ticks between 1st and 2nd fire events of the output neuron
        p_output.tick = MagicMock(return_value=output_fire_rule)
        self.assertEqual(snps.tick(), False)

        self.assertEqual(len(snps.output), 2)

if __name__ == '__main__':
    unittest.main()
