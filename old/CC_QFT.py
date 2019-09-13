import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
import cirq
from functools import reduce

def points2coeffs(points):
    return np.fft.fft(points, norm = 'ortho')

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

def QFT(qubits):
    circuit = cirq.Circuit()
    n = len(qubits)
    for i in range(n):
        circuit.append(cirq.H(qubits[i]))
        for j in range(i+1,n):
            circuit.append(cirq.CZ(qubits[j],qubits[i])**(1/(2**(j-i))))
    for i in range(n//2):
        circuit.append(cirq.CNOT(qubits[i], qubits[n-i-1]))
        circuit.append(cirq.CNOT(qubits[n-i-1], qubits[i]))
        circuit.append(cirq.CNOT(qubits[i], qubits[n-i-1]))
    return circuit

def points2coeffs_quantum(points):
    n = int(np.log2(len(points)))
    qubits = [cirq.GridQubit(0,i) for i in range(n)]
    circuit = cirq.Circuit()
    # circuit.append(construction_circuit(points, qubits))
    circuit.append(QFT(qubits))
    print(circuit)
    return cirq.Simulator().simulate(circuit, qubit_order = qubits, initial_state = points.astype(np.complex64)).final_state
    # return cirq.Simulator().simulate(circuit, qubit_order = qubits).final_state

# Simple circle with quadratic speed
# f = lambda t : np.exp(2.j*np.pi*t**2)

# Weird function
f = lambda t : np.exp(2.j * np.pi * (t + .04)) + .2 * np.exp(2.j * np.pi * 12 * t)

n = 16
M = 32
boundaries = np.linspace(0.,1.,M+1)
midpoints = .5 * (boundaries[:-1] + boundaries[1:])
points = np.array([f(t) for t in midpoints])
points /= np.linalg.norm(points)

fcs1 = points2coeffs(points)
fcs2 = points2coeffs_quantum(points)
for i,fcs in [(0,fcs1),(1,fcs2)]:
    coeffs = {}
    for k in range(-n,n+1):
        coeffs[k] = np.exp(-np.pi * 1.j * k / M) / np.sqrt(M) * fcs[k]

    ts = np.linspace(0.,1.,2048)
    fs = sum(c * np.exp(2.j * np.pi * k * ts) for k,c in coeffs.items())
    if i == 1:
        fs1 = fs
    else:
        fs2 = fs

fig = plt.figure()
ax = fig.gca()
cycle = [*points, points[0]]
ax.plot(np.real(cycle), np.imag(cycle), marker = '.', color = 'red')
ax.plot(np.real(fs1), np.imag(fs1), color = 'blue')
ax.plot(np.real(fs2), np.imag(fs2), color = 'teal')
plt.show()
