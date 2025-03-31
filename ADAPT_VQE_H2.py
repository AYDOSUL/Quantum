#ADAPT-VQE on H2
import pennylane as qml
from pennylane import numpy as np

#Generating the hamiltonian from scratch
symbols = ["H", "H"]
geometry = np.array([[-0.673,0,0],[0.673,0,0]], requires_grad=False)
H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry)

#Generating the excitation optimizer pool
n_electrons = 2
singles, doubles = qml.qchem.excitations(n_electrons, qubits)
singles_excitations = [qml.SingleExcitation(0.0, x) for x in singles]
doubles_excitations = [qml.DoubleExcitation(0.0, x) for x in doubles]
operator_pool = doubles_excitations + singles_excitations

#Generating the Hartree-Fock state for optimization
hf_state = qml.qchem.hf_state(n_electrons, qubits)
dev = qml.device("default.qubit", wires=qubits)

#Creating the base circuit
@qml.qnode(dev)
def circuit():
    qml.BasisState(hf_state, wires=range(qubits))
    return qml.expval(H)

#Creating the optimization loop
opt = qml.AdaptiveOptimizer()
for i in range(len(operator_pool)):
    circuit, energy, gradient = opt.step_and_cost(circuit, operator_pool, drain_pool=True)
    print('Energy:', energy)
    print(qml.draw(circuit, show_matrices=False)())
    print('Largest Gradient:', gradient)
    print()
    if gradient < 1e-8:
        break