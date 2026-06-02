import numpy as np
from sps.m_gpu import MSNPSystemGPU
from sps.config import Config

# Initial configuration: neuron charges
configurationVector = np.array([1, 1, 2], dtype=np.int32)

spikingTransitionMatrix = np.array([
    [-1, 1, 1],   # rule 0: fires from neuron 0
    [-2, 2, 2],   # rule 1: fires from neuron 0
    [1, -1, 1],   # rule 2: fires from neuron 1
    [0, 0, -1],   # rule 3: fires from neuron 2
], dtype=np.int32)

# Synapses matrix (4 rules x 3 neurons)
synapsesMatrix = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [-1, 1, 1],
    [0, 0, 1],
], dtype=np.int32)

# Rule vector (threshold for each rule) - row vector
ruleVector = np.array([1, 2, 1, 1], dtype=np.int32)

# Applying rule vector: maps each rule to its source neuron
# rule 0 -> neuron 0, rule 1 -> neuron 0, rule 2 -> neuron 1, rule 3 -> neuron 2
applyingRuleVector = np.array([0, 0, 1, 2], dtype=np.int32)

# Initial spiking vector (None to initialize to zeros)
spikingVector = None

# Input neurons and spike train (not used for this simple example)
input_neurons = np.array([], dtype=np.int32)
single_spike_train = np.array([], dtype=np.int32)

# Create GPU system
print("Creating MSNPSystemGPU...")
gpu_system = MSNPSystemGPU(
    configurationVector=configurationVector,
    spikingVector=spikingVector,
    spikingTransitionMatrix=spikingTransitionMatrix,
    synapsesMatrix=synapsesMatrix,
    ruleVector=ruleVector,
    max_steps=10,
    deterministic=True,
    single_spike_train=single_spike_train,
    input_neurons=input_neurons,
    applyingRuleVector=applyingRuleVector
)

print("\nInitial state:")
print(gpu_system)

print("\n" + "="*60)
print("Running 3 steps manually:")
print("="*60)

for step_num in range(3):
    print(f"\n--- Step {step_num + 1} ---")
    gpu_system.step(verbose=True)
    print(f"Configuration: {gpu_system.get_configuration_vector()}")
    print(f"Spiking: {gpu_system.get_spiking_vector()}")
    print(f"Net Gain: {gpu_system.get_net_gain_vector()}")

print("\n" + "="*60)
print("Rule count per neuron:", gpu_system.get_rule_count_per_neuron())
print("="*60)
