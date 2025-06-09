"""Some examples using spiking neural p systems"""
from sps.SNPSystem import SNPSystem

def compute_divisible_3():
    """SNPS that classify if a number is divisible by 3
    see Example 9 of paper https://link.springer.com/article/10.1007/s41965-020-00050-2?fromPaywallRec=false """
    snps = SNPSystem(max_delay=5, max_steps=100)
    snps.load_neurons_from_csv("neuronsDiv3.csv")
    snps.spike_train = [1, 0, 0, 0, 0, 0, 0, 1]  # example of an input spike train that create halting computation
    snps.start()
    print(snps.history)

def compute_gen_even():
    """SNPS that generate all possible even numbers
    see Figure 3 of paper https://www.researchgate.net/publication/220443792_Spiking_Neural_P_Systems """
    snps = SNPSystem(max_delay=5, max_steps=100)
    snps.load_neurons_from_csv("neuronsGenerateEven.csv")
    snps.start()
    print(snps.history)

compute_divisible_3()

'''How to read the output table:
r: rule applied. r:999a+2;3->0;1 mean "using 3 charge, 0 spike are fired, with 1 delay". ! = firing rules. 
1a is the condition part that follows: if c %999 == 2, the rule applies
c: internal charge of neuron. i:x(y) means "x charge received from neuron y" 

Remember: a neuron has rules of type E/r^x->a. E is a regular expression that should find an EXACT match with the number of spikes a in the neuron.
the rule consumes x spikes, or all the spikes if x is no defined. We use div and mod to describe E, and source is x. 
For the rules that want exact numbers, not regulars expressions, we are using div = k as "large number"
'''