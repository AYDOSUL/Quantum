#Hartree-Fock ground state estimation of H2O
import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem

symbols = ["H", "H"]
coordinates = np.array([[-0.673,0,0],[0.673,0,0]], requires_grad=False)

molecule = qchem.Molecule(symbols, coordinates)
H, qubits = qchem.molecular_hamiltonian(molecule)