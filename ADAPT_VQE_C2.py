#ADAPT-VQE on C2 with Pennylane QChem Database
import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem

#Loading data about the hamiltonian and number of qubits
part = qml.data.load("qchem", molname="C2", basis="STO-3G", bondlength=1.1,
                     attributes=["hamiltonian"])[0]
Hamiltonian = part.hamiltonian
qubits = 20

#Generating the Excitation operator pool
n_electrons = 12
singles, doubles = qml.qchem.excitations(n_electrons, qubits)
singles_excitations = [qml.SingleExcitation(0.0, x) for x in singles]
doubles_excitations = [qml.DoubleExcitation(0.0, x) for x in doubles]
operator_pool = doubles_excitations + singles_excitations

#Creating the Hartree-Fock state for the initialization
hf_state = qml.qchem.hf_state(n_electrons, qubits)

#Creating the Expected energy function
dev = qml.device("lightning.qubit", wires=qubits)
@qml.qnode(dev)
def exp_energy(state):
    qml.BasisState(np.array(state), wires=range(qubits))
    return qml.expval(Hamiltonian)

#Creating the circuit for the first initialization
@qml.qnode(dev)
def circuit():
    qml.BasisState(hf_state, wires=range(qubits))
    return qml.expval(Hamiltonian)

#Creating the optimization loop
opt = qml.AdaptiveOptimizer(0, 0.1)
for i in range(len(operator_pool)):
    circuit, energy, gradient = opt.step_and_cost(circuit, operator_pool, drain_pool=True)
    print('Energy:', energy)
    print(qml.draw(circuit, show_matrices=False)())
    print('Largest Gradient:', gradient)
    print()
    if gradient < 1e-8:
        break