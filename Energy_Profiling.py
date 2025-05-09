import pennylane as qml
from pennylane import qchem
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

Molecule = ["H", "H"]
def energy(symbols, bond_length):
    geometry = jnp.array([[0.0, 0.0, -bond_length[-1] / 2],
                            [0.0, 0.0, bond_length[-1] / 2]])
    mol = qchem.Molecule(symbols, geometry)
    energy_val = qchem.hf_energy(mol)()
    return energy_val

plot_range = [x * 0.1 for x in range(1, 50)]
energies = []

def plotting():
    for i in plot_range:
        current_energy = energy(Molecule, jnp.array([i]))
        energies.append(current_energy)
    return energies

plt.plot(plot_range, plotting())
plt.xlabel("Bond Length [Bohr]")
plt.ylabel("Hartree-Fock Energy [Ha]")
plt.title("Hartree-Fock Energy vs. Bond Length")
plt.grid(True)
plt.show()