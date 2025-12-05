from sps.snp_system import SNPSystem

# for testing different types of networks
def compute_divisible_3():
    #SNPS that classify if a number is divisible by 3
    #see Example 9 of paper https://link.springer.com/article/10.1007/s41965-020-00050-2?fromPaywallRec=false
    snps = SNPSystem(5, 100, "spike_train", "halting", True)
    snps.load_neurons_from_csv("csv/" + "neuronsDiv3.csv")
    snps.spike_train = [1, 0, 0, 0, 0, 0, 0, 1]  # example of an input spike train that create halting computation
    snps.start()
    print(snps.history)
    with open("historyDiv3.txt", "w", encoding="utf-8") as f:
        f.write(str(snps.history))

def compute_gen_even():
    #SNPS that generate all possible even numbers
    #see Figure 3 of paper https://www.researchgate.net/publication/220443792_Spiking_Neural_P_Systems
    snps = SNPSystem(5, 100, "none", "generative", False)
    snps.load_neurons_from_csv("csv/" + "neuronsGenerateEven.csv")
    snps.start()
    print(snps.history)

def compute_extended():
    #SNPS that test the extended version of the rules
    #see https://www.researchgate.net/publication/31597157_Spiking_Neural_P_Systems_with_Extended_Rules
    snps = SNPSystem(5, 10, "none", "halting", False)
    snps.load_neurons_from_csv("csv/" + "ExampleExtended.csv")
    snps.start()
    print(snps.history)