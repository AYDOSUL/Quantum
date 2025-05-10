#Hartree-Fock ground state estimation of H2O
import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem

symbols = ["H", "O", "H"]
coordinates = np.array([[-0.0399, -0.0038, 0.0], [1.5780, 0.8540, 0.0], [2.7909, -0.5159, 0.0]])

molecule = qchem.Molecule(symbols, coordinates)
H, qubits = qchem.molecular_hamiltonian(molecule)

# Create quantum device
dev = qml.device("lightning.qubit", wires=qubits)

# Define QNode to compute energy
@qml.qnode(dev)
def exp_energy(state):
    qml.BasisState(np.array(state), wires=range(num_wires))
    return qml.expval(H)
print(qubits)
# Generate Hartree-Fock state for H2O (This uses the correct number of qubits and the Hartree-Fock occupation)
hf = qchem.hf_state(electrons=10, orbitals=qubits)

# Ensure that hf has the right length
print("HF state:", hf)

# Print HF energy
print("HF Energy:", exp_energy(hf))