#VQE for the ground state estimation of BH3
import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem

part = qml.data.load("qchem", molname="BH3", basis="STO-3G", attributes=["hamiltonian"])[0]
H = part.hamiltonian
qubits = 16

n_electrons = 8

singles, doubles = qchem.excitations(n_electrons, qubits)
singles_excitations = [qml.SingleExcitation(0.0, x) for x in singles]
doubles_excitations = [qml.DoubleExcitation(0.0, x) for x in doubles]
operator_pool = singles_excitations + doubles_excitations

hf_state = qchem.hf_state(n_electrons, qubits)
dev = qml.device("lightning.qubit", wires=qubits)

@qml.qnode(dev)
def circuit():
    qml.BasisState(hf_state, wires=range(qubits))
    return qml.expval(H)
gradient = 1
prev_gradient = 1
opt = qml.AdaptiveOptimizer(1, 0.1)
run = 0
for i in range(len(operator_pool)):
    run = run + 1
    prev_gradient = gradient
    circuit, energy, gradient = opt.step_and_cost(circuit, operator_pool, drain_pool=True)
    derivative_gradient = gradient/prev_gradient
    print("Energy", energy)
    print(qml.draw(circuit, show_matrices=False))
    print("Largest Gradient", gradient)
    print("Gradient Prime", derivative_gradient)
    print(f"Iteration number: {run}")
    print()
    if 1e-7 > gradient:
        break