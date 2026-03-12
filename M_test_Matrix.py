import os
import sys

# ensure the project root is on sys.path so that the `sps` package can be imported
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from sps.config import Config
from sps.snp_system import SNPSystem
from sps.M_matrix_executor import MatrixExecutor

# running this small script outside the main application; choose a non-cnn MODE so that
# the constructor doesn't try to allocate feature/pooling arrays. Tests use "test",
# but halting or generative are also fine depending on the network's semantics.
Config.MODE = "test"

# input_len is only required when MODE == "cnn", otherwise it can be any integer.
# use 0 here to make the intent clear.
snps = SNPSystem(0, 100, True)
snps.load_neurons_from_csv("csv/" + "neuronsDiv3.csv")
msnps = MatrixExecutor(snps)
print(msnps)