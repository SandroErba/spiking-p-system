
class MatrixExecutor:


    # Da capire per gil spike train
    def __init__(self, SNPSystem):
        self.SNPSystem = SNPSystem
        self.neurons = SNPSystem.neurons
        self.deterministic = SNPSystem.deterministic
        self.max_steps = SNPSystem.max_steps

    # Necessito di un:
    # - configuration vector
    # - spiking vector
    # - spiking transition matrix
    # - net gain vector

    def __str__(self):
        output = ""
        for neuron in self.neurons:
            output += neuron.__str__() + "\n"
        return output
    