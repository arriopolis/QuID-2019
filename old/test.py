import cirq
import numpy as np
np.set_printoptions(linewidth=200)
from functools import reduce

def apply_controls(gate, control_qubits):
    return reduce(lambda x,y : x.controlled_by(y), control_qubits, gate)

def normalize(v):
    return v / np.linalg.norm(v)

def construction_circuit(points, qubits, control_string = ''):
    n = int(np.log2(len(points)))
    assert len(points) == 2**n
    assert abs(np.linalg.norm(points) - 1.) < 1e-8

    p1 = np.linalg.norm(points[:2**(n-1)])
    theta = 2 * np.arccos(p1) / np.pi
    phi = (np.angle(points[2**(n-1)]) - np.angle(points[0])) / np.pi
    k = len(control_string)

    circuit = cirq.Circuit()

    for j,c in enumerate(control_string):
        if c == '0':
            circuit.append(cirq.X(qubits[j]))

    circuit.append(apply_controls(cirq.Y(qubits[k])**theta, qubits[:k]))
    circuit.append(apply_controls(cirq.Z(qubits[k])**phi, qubits[:k]))

    for j,c in enumerate(control_string):
        if c == '0':
            circuit.append(cirq.X(qubits[j]))

    if n > 1:
        circuit.append(construction_circuit(normalize(points[:2**(n-1)]), qubits, control_string + '0'))
        circuit.append(construction_circuit(normalize(points[2**(n-1):]), qubits, control_string + '1'))
    return circuit

f = lambda t : np.exp(2.j * np.pi * t)

M = 8
boundaries = np.linspace(0.,1.,M+1)
midpoints = .5 * (boundaries[:-1] + boundaries[1:])
points = np.array([f(t) for t in midpoints])
points = [ 0.24862956+0.34668762j, -0.44474569+0.34668762j, -0.44474569-0.34668762j, 0.24862956-0.34668762j]
points /= np.linalg.norm(points)

n = int(np.log2(M))
qubits = [cirq.GridQubit(0,i) for i in range(n)]
circuit = construction_circuit(points, qubits)

print(circuit)
sim = cirq.Simulator()
gen = sim.simulate_moment_steps(circuit)
for step_result in gen:
    print(dir(gen.gi_frame.f_globals['protocols'].apply_unitary))
    print(gen.gi_frame.f_globals['protocols'].apply_unitary.__code__)
    print(dir(gen.gi_frame.f_globals['protocols'].apply_unitary.__code__))
    print(gen.gi_frame.f_globals['protocols'].ApplyUnitaryArgs)
