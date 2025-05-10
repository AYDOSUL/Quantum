import pennylane as qml
from pennylane import qchem
from jax import numpy as jnp
import jax  # Import the jax module
import matplotlib.pyplot as plt
import optax
from pennylane.templates import AllSinglesDoubles

# Define the molecular system
symbols = ["H", "H", "H"]  # Use a standard Python list for symbols
multiplicity = 2
electrons = 3
orbitals = 6
singles, doubles = qchem.excitations(electrons, orbitals)
hf = qchem.hf_state(electrons, orbitals)

# Range of distances to scan for the PES
r_range = jnp.arange(1.0, 3.0, 0.1)
energies = []
pes_point = 0
params_old = jnp.zeros(len(singles) + len(doubles))  # Initialize previous parameters

for r in r_range:
    print(f"Calculating energy for H-H distance: {r:.2f} Bohr")
    coordinates = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, r], [0.0, 0.0, 4.0]])
    molecule = qchem.Molecule(symbols, coordinates, mult=multiplicity)

    H, qubits = qchem.molecular_hamiltonian(molecule, method='openfermion')

    dev = qml.device("lightning.qubit", wires=qubits)
    opt = optax.sgd(learning_rate=1.5)  # Stochastic Gradient Descent

    @qml.qnode(dev, interface='jax')
    def circuit(parameters):
        AllSinglesDoubles(parameters, range(qubits), hf, singles, doubles)
        return qml.expval(H)

    def cost_fn(params):
        return circuit(params)

    init_params = jnp.zeros(len(singles) + len(doubles))
    if pes_point > 0:
        init_params = params_old  # Use previous converged parameters as initial guess

    opt_state = opt.init(init_params)
    params = init_params
    prev_energy = 0.0
    convergence_threshold = 1e-6

    print("Starting optimization...")
    for i in range(60):
        grads = jax.grad(cost_fn)(params)  # Use jax.grad directly
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        energy = circuit(params)

        if jnp.abs(energy - prev_energy) < convergence_threshold:
            print(f"Converged at step {i+1} with energy: {energy:.8f} Hartree")
            break
        prev_energy = energy
        if (i + 1) % 10 == 0:
            print(f"Step {i+1}: Energy = {energy:.8f} Hartree")
    else:
        print(f"Optimization did not converge within {60} steps. Final energy: {energy:.8f} Hartree")

    # Store the converged parameters and energy
    params_old = params
    energies.append(energy)
    pes_point += 1

# Plot the Potential Energy Surface
fig, ax = plt.subplots()
ax.plot(r_range, energies)
ax.set(
    xlabel="Distance (Bohr)",
    ylabel="Total energy (Hartree)",
    title="Potential Energy Surface of H3 (Linear Geometry)",
)
ax.grid()
plt.show()