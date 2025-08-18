import unittest
from sps.PNeuron import PNeuron
from sps.SNPSystem import SNPSystem
from sps.SpikeUtils import TransformationRule

class PNeuronTest(unittest.TestCase):
    def test_fire(self):
        fire_rule = TransformationRule(div=1, mod=0, source=5, target=1, delay=3)
        snps = SNPSystem(5, 5, "test", "test", True)
        pn = PNeuron(snps, 0, targets=[], transf_rules=[fire_rule])
        pn.charge = 8
        pn.fire(pn.transf_rules[0])

        # after a firing event, the neuron current charge is diminished by rule source condition
        self.assertEqual(pn.charge, 8-5)

        # after a firing event, the neuron refractory period is equal to rule delay, diminishing by 1 per tick
        self.assertEqual(pn.refractory, 3)
        for x in range(3):
            self.assertEqual(pn.tick(), None)


    def test_consume(self):
        fire_rule = TransformationRule(div=1, mod=0, source=5, target=1, delay=3)
        snps = SNPSystem(5, 5, "test", "test", True)
        pn = PNeuron(snps, 0, targets=[], transf_rules=[fire_rule])
        pn.charge = 8
        pn.fire(pn.transf_rules[0])

        # after a consume event, the neuron current charge is diminished by rule source condition
        self.assertEqual(pn.charge, 8-5)

if __name__ == '__main__':
    unittest.main()
