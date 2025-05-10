#Applying a VQE for ground state estimation of H2O
import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem

# Define the molecule
symbols = ["H", "O", "H"]
coordinates = np.array([[-0.0399, -0.0038, 0.0], [1.5780, 0.8540, 0.0], [2.7909, -0.5159, 0.0]])

# Generate the molecular Hamiltonian
molecule = qchem.Molecule(symbols, coordinates)
H, qubits = qchem.molecular_hamiltonian(molecule)

# Number of qubits (the same as the number of orbitals used)
num_wires = qubits

# Hartree-Fock state for 10 electrons and 10 orbitals
hf = qchem.hf_state(electrons=10, orbitals=num_wires)

# VQE Setup
dev = qml.device("lightning.qubit", wires=num_wires)

# Energy measurement function
def exp_energy(state):
    qml.BasisState(np.array(state), wires=range(num_wires))
    return qml.expval(H)

# Ansatz function
def ansatz(params):
    qml.BasisState(hf, wires=range(num_wires))  # Start with the Hartree-Fock state
    # Apply double-excitation gates with valid qubit indices
    qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])  # DoubleExcitation on qubits 0, 1, 2, 3
    qml.DoubleExcitation(params[1], wires=[4, 5, 6, 7])  # DoubleExcitation on qubits 4, 5, 6, 7

@qml.qnode(dev)
def cost_function(params):
    ansatz(params)
    return qml.expval(H)

# Optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.1)
theta = np.array([0.0, 0.0], requires_grad=True)

# Energy and angle tracking
energy = [cost_function(theta)]
angle = [theta]
max_iterations = 2

# Optimization loop
for n in range(max_iterations + 1):
    theta, prev_energy = opt.step_and_cost(cost_function, theta)

    energy.append(cost_function(theta))
    angle.append(theta)

    print(f"Step = {n}, Energy = {energy[-1]:.8f} Ha")  # Log Prints

# Function to compute the final quantum state
@qml.qnode(dev)
def ground_state(params):
    ansatz(params)
    return qml.state()