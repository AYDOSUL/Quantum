import pennylane as qml
from pennylane import qchem
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import unitary_group

n_qubits = 2

pnp.random.seed(7)
np.random.seed(7)

# generate random weights

alphas = np.random.normal(0, 0.5, size=n_qubits)

hamiltonian = qml.sum(
    *[qml.PauliZ(wires=i) @ qml.PauliZ(wires=i + 1) for i in range(n_qubits - 1)]
) + qml.dot(alphas, [qml.PauliX(wires=i) for i in range(n_qubits)])

n_random_states = 100

# Generating random unitaries
random_unitaries = unitary_group.rvs(2**n_qubits, n_random_states)
# Take the first column of each unitary as a random state
random_states = [random_unitary[:, 0] for random_unitary in random_unitaries]

dev = qml.device("default.qubit")

@qml.qnode(dev)
def target_circuit(input_state):
    # prepare training state
    qml.StatePrep(input_state, wires=range(n_qubits))

    # evolve the Hamiltonian for time=2 in n=1 steps with the order 1 formula
    qml.TrotterProduct(hamiltonian, time=2, n=1, order=1)
    return qml.classical_shadow(wires=range(n_qubits))


qml.draw_mpl(target_circuit)(random_states[0])
plt.show()

n_measurements = 10000

shadows = []
for random_state in random_states:
    bits, recipes = target_circuit(random_state, shots=n_measurements)
    shadow = qml.ClassicalShadow(bits, recipes)
    shadows.append(shadow)

@qml.qnode(dev)
def model_circuit(params, random_state):
    qml.StatePrep(random_state, wires=range(n_qubits))
    # parameterized quantum circuit with the same gate structure as the target
    for i in range(n_qubits):
        qml.RX(params[i], wires=i)

    for i in reversed(range(n_qubits - 1)):
        qml.IsingZZ(params[n_qubits + i], wires=[i, i + 1])
    return [qml.density_matrix(i) for i in range(n_qubits)]


initial_params = pnp.random.random(size=n_qubits*2-1, requires_grad=True)

qml.draw_mpl(model_circuit)(initial_params, random_states[0])
plt.show()

def cost(params):
    cost = 0.0
    for idx, random_state in enumerate(random_states):
        # obtain the density matrices for each qubit
        observable_mats = model_circuit(params, random_state)
        # convert to a PauliSentence
        observable_pauli = [
            qml.pauli_decompose(observable_mat, wire_order=[qubit])
            for qubit, observable_mat in enumerate(observable_mats)
        ]
        # estimate the overlap for each qubit
        cost = cost + qml.math.sum(shadows[idx].expval(observable_pauli))
    cost = 1 - cost / n_qubits / n_random_states
    return cost


params = initial_params

optimizer = qml.GradientDescentOptimizer(stepsize=5)
steps = 50

costs = [None]*(steps+1)
params_list = [None]*(steps+1)

params_list[0]=initial_params
for i in range(steps):
    params_list[i + 1], costs[i] = optimizer.step_and_cost(cost, params_list[i])

costs[-1] = cost(params_list[-1])

print("Initial cost:", costs[0])
print("Final cost:", costs[-1])

# find the ideal parameters from the original Trotterized Hamiltonian
ideal_parameters = [
    op.decomposition()[0].parameters[0]
    for op in qml.TrotterProduct(hamiltonian, 2, 1, 1).decomposition()
]
ideal_parameters = ideal_parameters[:n_qubits][::-1] + ideal_parameters[n_qubits:]

ideal_cost = cost(ideal_parameters)

plt.plot(costs, label="Training")
plt.plot([0, steps], [ideal_cost, ideal_cost], "r--", label="Ideal parameters")
plt.ylabel("Cost")
plt.xlabel("Training iterations")
plt.legend()
plt.show()

target_matrix = qml.matrix(
    qml.TrotterProduct(hamiltonian, 2, 1, 1),
    wire_order=range(n_qubits),
)

zero_state = [1] + [0]*(2**n_qubits-1)

# model matrix using the all-|0> state to negate state preparation effects
model_matrices = [qml.matrix(model_circuit, wire_order=range(n_qubits))(params, zero_state) for params in params_list]
trace_distances = [qml.math.trace_distance(target_matrix, model_matrix) for model_matrix in model_matrices]

plt.plot(trace_distances)
plt.ylabel("Trace distance")
plt.xlabel("Training iterations")
plt.show()

print("The final trace distance is: \n", trace_distances[-1])