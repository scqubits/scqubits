import numpy as np
import scqubits as scq
# l = np.random.randint(100)
from scqubits.core.analyzeqcircuit import example_circuit
# print("hello", l)


circ = scq.AnalyzeQCircuit.from_input_string(example_circuit("transmon"))
print(scq.__file__)
# print(circ.H)
