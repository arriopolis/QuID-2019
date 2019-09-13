import numpy as np
np.set_printoptions(linewidth=200)
import cirq
from functools import reduce

def apply_controls(gate, control_qubits):
    return reduce(lambda x,y : x.controlled_by(y), control_qubits, gate)

def normalize_and_remove_phase(v):
    return v / np.linalg.norm(v) / np.exp(1.j * np.angle(v[0]))

def construct_state_circuit(points, qubits, control_string = '', debug = False):
    k = len(control_string)
    n = len(qubits) - k
    assert len(points) == 2**n

    points = normalize_and_remove_phase(points)
    assert abs(np.linalg.norm(points) - 1.) < 1e-4
    assert abs(np.angle(points[0])) < 1e-4

    p1 = np.linalg.norm(points[:2**(n-1)])
    theta = 2 * np.arccos(p1) / np.pi
    phi = np.angle(points[2**(n-1)]) / np.pi

    if debug:
        print("Current control string:", control_string)
        print("state = ", points)
        print("p1 = ", p1)
        print("theta = ", theta)
        print("phi = ", phi)
        print("phi[0] = ", np.angle(points[0]))
        print("phi[2**(n-1)] = ", np.angle(points[2**(n-1)]))
        print()

    x_circuit = cirq.Circuit()
    for j,c in enumerate(control_string):
        if c == '0':
            x_circuit.append(cirq.X(qubits[j]))

    circuit = cirq.Circuit()
    circuit.append(x_circuit)
    circuit.append(apply_controls(cirq.Y(qubits[k])**theta, qubits[:k]))
    if k > 0: circuit.append(apply_controls(cirq.Z(qubits[k-1])**(-theta/2), qubits[:(k-1)]))
    circuit.append(apply_controls(cirq.Z(qubits[k])**phi, qubits[:k]))
    circuit.append(x_circuit)

    if debug:
        print("The Y rotation as a matrix:")
        print(apply_controls(cirq.Y(qubits[k])**theta, qubits[:k])._unitary_())
        print("The Z rotation as a matrix:")
        print(apply_controls(cirq.Z(qubits[k])**phi, qubits[:k])._unitary_())
        print()

    if n > 1:
        circuit.append(construct_state_circuit(points[:2**(n-1)], qubits, control_string + '0', debug = debug))
        circuit.append(construct_state_circuit(points[2**(n-1):], qubits, control_string + '1', debug = debug))
    return circuit

if __name__ == "__main__":
    n = 2
    state = np.array([ 0.24862956+0.34668762j, -0.44474569+0.34668762j, -0.44474569-0.34668762j, 0.24862956-0.34668762j])
    state /= np.linalg.norm(state)

    qubits = [cirq.GridQubit(0,i) for i in range(n)]
    circuit = construct_state_circuit(state, qubits)
    constructed_state = cirq.Simulator().simulate(circuit, qubit_order = qubits).final_state
    inp = np.sum(np.conj(state) * constructed_state)

    print("Construction circuit:")
    print(circuit)

    print("Constructed state:")
    print(constructed_state)

    print("Correct state:")
    print(state)

    print("Absolute value of the inner product:")
    print(abs(inp))
