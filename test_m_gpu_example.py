import numpy as np
from sps.m_gpu import MSNPSystemGPU
from sps.config import Config


Config.WHITE_HOLE = False 
# Initial configuration: neuron charges
configurationVector = np.array([5, 2, 2,5,2,2,5,2,2,5,2,2,5,2], dtype=np.int32)

spikingTransitionMatrix = np.array([
    [-1, 1, 1,0,0,0,1,1,1,1,1,1,1,1], 
    [-2, 1, 1,0,0,0,1,1,1,1,1,1,1,1],   
    [-3, 1, 1,0,0,0,1,1,1,1,1,1,1,1],   
    [-4, 1, 1,0,0,0,1,1,1,1,1,1,1,1],
    [-5, 2, 2,0,0,0,2,2,2,2,2,2,2,2], 
    [-6, 2, 2,0,0,0,2,2,2,2,2,2,2,2],
    [-7, 2, 2,0,0,0,2,2,2,2,2,2,2,2],
    [-8, 2, 2,0,0,0,2,2,2,2,2,2,2,2],
    [0,0,0,-1,1,1,0,0,0,1,1,1,1,1],
    [0,0,0,-2,1,1,0,0,0,1,1,1,1,1],
    [0,0,0,-3,1,1,0,0,0,1,1,1,1,1],
    [0,0,0,-4,1,1,0,0,0,1,1,1,1,1],
    [0,0,0,-5,2,2,0,0,0,2,2,2,2,2],
    [0,0,0,-6,2,2,0,0,0,2,2,2,2,2],
    [0,0,0,-7,2,2,0,0,0,2,2,2,2,2],
    [0,0,0,-8,2,2,0,0,0,2,2,2,2,2], 
], dtype=np.int32)

spikingTransitionMatrix = np.repeat(spikingTransitionMatrix, repeats=1, axis=1)

# Synapses matrix (4 rules x 3 neurons)
synapsesMatrix = np.array([
    [1, -1, 1,0,0,0,1,1,1,1,1,1,1,1],
    [1, -1, 1,0,0,0,1,1,1,1,1,1,1,1],
    [1, -1, 1,0,0,0,1,1,1,1,1,1,1,1],
    [1, -1, 1,0,0,0,1,1,1,1,1,1,1,1],
    [1, -1, 1,0,0,0,1,1,1,1,1,1,1,1],
    [1, -1, 1,0,0,0,1,1,1,1,1,1,1,1],
    [1, -1, 1,0,0,0,1,1,1,1,1,1,1,1],
    [1, -1, 1,0,0,0,1,1,1,1,1,1,1,1],
    [0,0,0,1, -1, 1,0,0,0,1,1,1,1,1],
    [0,0,0,1, -1, 1,0,0,0,1,1,1,1,1],
    [0,0,0,1, -1, 1,0,0,0,1,1,1,1,1],
    [0,0,0,1, -1, 1,0,0,0,1,1,1,1,1],
    [0,0,0,1, -1, 1,0,0,0,1,1,1,1,1],
    [0,0,0,1, -1, 1,0,0,0,1,1,1,1,1],
    [0,0,0,1, -1, 1,0,0,0,1,1,1,1,1],
    [0,0,0,1, -1, 1,0,0,0,1,1,1,1,1],
], dtype=np.int32)

ruleVector = np.array([1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8], dtype=np.int32)

applyingRuleVector = np.array([0,0,0,0,0,0,0,0,3,3,3,3,3,3,3,3], dtype=np.int32)

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

