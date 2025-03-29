##Applying a VQE for ground state estimation of H+3(Trihydrogen Cation)
import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem

symbols = ["H", "H", "H"]
coordinates = np.array([[0.0102,0.0442,0.0],[0.9867,1.6303,0.0],
                        [1.8720,-0.0085,0.0]])#Vertices of H_3 atom
hamiltonian, qubits = qchem.molecular_hamiltonian(symbols, coordinates, charge=1)
hf = qchem.hf_state(electrons=2, orbitals = 6)

num_wires = qubits
dev = qml.device("default.qubit", wires=num_wires)

@qml.qnode(dev)
#Calculates Energy
def exp_energy(state):
  qml.BasisState(np.array(state), wires=range(num_wires))
  return qml.expval(hamiltonian)
#Ansatz
def ansatz(params):
  qml.BasisState(hf,wires=range(num_wires))
  qml.DoubleExcitation(params[0],wires=[0,1,2,3])
  qml.DoubleExcitation(params[1],wires=[0,1,4,5])
#Cost function given parameters
@qml.qnode(dev)
def cost_function(params):
  ansatz(params)
  return qml.expval(hamiltonian)

#Optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.1)
theta = np.array([0.0,0.0], requires_grad = True)

energy = [cost_function(theta)]
angle = [theta]
max_iterations = 50

for n in range(max_iterations + 1):
  theta, prev_energy = opt.step_and_cost(cost_function, theta)

  energy.append(cost_function(theta))
  angle.append(theta)

  print(f"Step = {n}, Energy = {energy[-1]:.8f} Ha")#Log Prints

@qml.qnode(dev)
def ground_state(params):
  ansatz(params)
  return qml.state()
print(ground_state(theta))