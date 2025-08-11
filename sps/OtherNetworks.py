from sps import Config
from sps.MedMnist import process_dataset
from sps.SNPSystem import SNPSystem
import numpy as np
from sklearn.datasets import fetch_openml, load_digits
import matplotlib.pyplot as plt
from skimage.measure import block_reduce

# for testing different types of networks
def compute_divisible_3():
    #SNPS that classify if a number is divisible by 3
    #see Example 9 of paper https://link.springer.com/article/10.1007/s41965-020-00050-2?fromPaywallRec=false
    snps = SNPSystem(5, 100, "spike_train", "halting", True)
    snps.load_neurons_from_csv("neuronsDiv3.csv")
    snps.spike_train = [1, 0, 0, 0, 0, 0, 0, 1]  # example of an input spike train that create halting computation
    snps.start()
    print(snps.history)
    with open("historyDiv3.txt", "w", encoding="utf-8") as f:
        f.write(str(snps.history))

def compute_gen_even():
    #SNPS that generate all possible even numbers
    #see Figure 3 of paper https://www.researchgate.net/publication/220443792_Spiking_Neural_P_Systems
    #require nondeterminism, see method tick in class PNeuron
    snps = SNPSystem(5, 100, "none", "generative", False)
    snps.load_neurons_from_csv("neuronsGenerateEven.csv")
    snps.start()
    print(snps.history)