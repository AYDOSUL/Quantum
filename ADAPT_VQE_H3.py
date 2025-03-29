#ADAPT-VQE on H3
import pennylane as qml
from pennylane import numpy as np

symbols = ["H", "H", "H"]
geometry = np.array([[0.01076341, 0.04449877, 0.0],
                     [0.98729513, 1.63059094, 0.0],
                     [1.87262415, -0.00815842, 0.0]], requires_grad=False)
H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge = 1)

n_electrons = 2
singles, doubles = qml.qchem.excitations(n_electrons, qubits)
singles_excitations = [qml.SingleExcitation(0.0, x) for x in singles]
doubles_excitations = [qml.DoubleExcitation(0.0, x) for x in doubles]
operator_pool = doubles_excitations + singles_excitations

hf_state = qml.qchem.hf_state(n_electrons, qubits)
dev = qml.device("default.qubit", wires=qubits)
@qml.qnode(dev)
def circuit():
    qml.BasisState(hf_state, wires=range(qubits))
    return qml.expval(H)

opt = qml.AdaptiveOptimizer()
for i in range(len(operator_pool)):
    circuit, energy, gradient = opt.step_and_cost(circuit, operator_pool, drain_pool=True)
    print('Energy:', energy)
    print(qml.draw(circuit, show_matrices=False)())
    print('Largest Gradient:', gradient)
    print()
    if gradient < 1e-8:
        break