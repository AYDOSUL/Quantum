#ADAPT-VQE on H2
import pennylane as qml
from pennylane import numpy as np

symbols = ["H", "H", "H"]
geometry = coordinates = np.array([[0.0102,0.0442,0.0],[0.9867,1.6303,0.0],
                        [1.8720,-0.0085,0.0]], )
H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=1)

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
prev_gradient = 1
gradient = 1
opt = qml.AdaptiveOptimizer()
for i in range(len(operator_pool)):
    prev_gradient = gradient
    circuit, energy, gradient = opt.step_and_cost(circuit, operator_pool, drain_pool=True)
    derivative_gradient = gradient/prev_gradient
    print('Energy:', energy)
    print(qml.draw(circuit, show_matrices=False)())
    print('Largest Gradient:', gradient)
    print('Gradient Prime:', derivative_gradient)
    print()
    if gradient < 1e-10:
        break