import pennylane as qml
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

def find_lowest_energy_bond_length(symbols, initial_bond_length, num_steps=100, learning_rate=0.1):

    bond_lengths = [initial_bond_length]
    energies = []

    for i in range(num_steps):
        geometry = jnp.array([[0.0, 0.0, -bond_lengths[-1] / 2],
                             [0.0, 0.0, bond_lengths[-1] / 2]])
        mol = qml.qchem.Molecule(symbols, geometry)

        # Compute the Hartree-Fock energy and its gradient with respect to geometry
        energy_val = qml.qchem.hf_energy(mol)()
        grad_geometry = jax.grad(qml.qchem.hf_energy(mol), argnums=0)(geometry, mol.coeff, mol.alpha)

        energies.append(energy_val)


        force = grad_geometry[1, 2] - grad_geometry[0, 2]

        # Update the bond length using gradient descent
        new_bond_length = bond_lengths[-1] - learning_rate * force
        bond_lengths.append(new_bond_length)

        if i % 1 == 0:
            print(f"Step {i}: Bond Length = {bond_lengths[-1]:.4f} Bohr, Energy = {energy_val:.8f} Ha")

    min_energy_index = np.argmin(energies)
    lowest_energy_bond_length = bond_lengths[min_energy_index]
    lowest_energy = energies[min_energy_index]

    print(f"\nLowest Energy Bond Length found: {lowest_energy_bond_length:.4f} Bohr")
    print(f"Corresponding Energy: {lowest_energy:.8f} Ha")

    plt.plot(bond_lengths[:-1], energies)
    plt.xlabel("Bond Length [Bohr]")
    plt.ylabel("Hartree-Fock Energy [Ha]")
    plt.title("Hartree-Fock Energy vs. Bond Length")
    plt.grid(True)
    plt.show()

    return lowest_energy_bond_length

if __name__ == "__main__":
    symbols = ["C", "C"]
    initial_bond_length = 1.3459  # Initial guess for H2 bond length in Bohr
    lowest_energy_bond_length = find_lowest_energy_bond_length(symbols, initial_bond_length)
    print(f"Estimated lowest energy bond length for H2: {lowest_energy_bond_length:.4f} Bohr")