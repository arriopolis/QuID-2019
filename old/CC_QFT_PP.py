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
    circuit.append(construction_circuit(points, qubits))
    circuit.append(QFT(qubits))
    # print(circuit)
    # return cirq.Simulator().simulate(circuit, qubit_order = qubits, initial_state = points.astype(np.complex64)).final_state
    return cirq.Simulator().simulate(circuit, qubit_order = qubits).final_state

def generate_circuit_normal_measurement(points):
    n = int(np.log2(len(points)))
    qubits = [cirq.GridQubit(0,i) for i in range(n)]
    circuit = cirq.Circuit()
    circuit.append(construction_circuit(points, qubits))
    circuit.append(QFT(qubits))
    circuit.append(cirq.measure(*qubits, key = 'a'))
    return circuit

def generate_circuit_normal_phase_measurement(points, i):
    n = int(np.log2(len(points)))
    qubits = [cirq.GridQubit(0,i) for i in range(n)]
    circuit = cirq.Circuit()
    circuit.append(construction_circuit(points, qubits))
    circuit.append(QFT(qubits))
    circuit.append(cirq.H(qubits[i]))
    circuit.append(cirq.measure(*qubits, key = 'a'))
    return circuit

def generate_circuit_imaginary_phase_measurement(points, i):
    n = int(np.log2(len(points)))
    qubits = [cirq.GridQubit(0,i) for i in range(n)]
    circuit = cirq.Circuit()
    circuit.append(construction_circuit(points, qubits))
    circuit.append(QFT(qubits))
    circuit.append(cirq.H(qubits[i]))
    circuit.append(cirq.S(qubits[i]))
    circuit.append(cirq.measure(*qubits, key = 'a'))
    return circuit

def get_coefficients(absolute_measurements, normal_phase_measurements, imaginary_phase_measurements):
    num_qubits = len(normal_phase_measurements)

    num_absolute_measurements = sum(absolute_measurements.values())
    rs = np.sqrt(np.array([absolute_measurements[i]/num_absolute_measurements for i in sorted(absolute_measurements.keys())]))

    for i in range(num_qubits):
        npms = normal_phase_measurements[i]
        ipms = imaginary_phase_measurements[i]
        for j in range()
        num_npms = sum(npms.values())
        num_ipms = sum(ipms.values())

    return rs

# Simple circle with quadratic speed
f = lambda t : np.exp(2.j*np.pi*t**2)

# Weird function
# f = lambda t : np.exp(2.j * np.pi * t) + .2 * np.exp(2.j * np.pi * 12 * t)

n = 16
M = 32
boundaries = np.linspace(0.,1.,M+1)
midpoints = .5 * (boundaries[:-1] + boundaries[1:])
points = np.array([f(t) for t in midpoints])
points /= np.linalg.norm(points)

num_qubits = int(np.log2(len(points)))

absolute_measurements = {j : 0 for j in range(2**num_qubits)}
normal_phase_measurements = [{j : 0 for j in range(2**num_qubits)} for _ in range(num_qubits)]
imaginary_phase_measurements = [{j : 0 for j in range(2**num_qubits)} for _ in range(num_qubits)]

circuit = generate_circuit_normal_measurement(points)
res = cirq.Simulator().run(circuit, repetitions = 1024)
for o in res.measurements['a']:
    j = np.sum(o.astype(int) * np.power(2, np.arange(num_qubits-1, -1, -1)))
    absolute_measurements[j] += 1

for i in range(num_qubits):
    circuit = generate_circuit_normal_phase_measurement(points, i)
    res = cirq.Simulator().run(circuit, repetitions = 1024)
    for o in res.measurements['a']:
        j = np.sum(o.astype(int) * np.power(2, np.arange(num_qubits-1, -1, -1)))
        normal_phase_measurements[i][j] += 1

for i in range(num_qubits):
    circuit = generate_circuit_imaginary_phase_measurement(points, i)
    res = cirq.Simulator().run(circuit, repetitions = 1024)
    for o in res.measurements['a']:
        j = np.sum(o.astype(int) * np.power(2, np.arange(num_qubits-1, -1, -1)))
        imaginary_phase_measurements[i][j] += 1

fcs = get_coefficients(absolute_measurements, normal_phase_measurements, imaginary_phase_measurements)
print(fcs)

fcs_old = points2coeffs_quantum(points)
print(fcs_old)

fig = plt.figure()
ax = fig.gca()
ax.plot(np.abs(fcs), label = 'measured')
ax.plot(np.abs(fcs_old), label = 'calculated')
ax.set_title('abs')
ax.legend()

fig = plt.figure()
ax = fig.gca()
ax.plot(np.real(fcs), label = 'measured')
ax.plot(np.real(fcs_old), label = 'calculated')
ax.set_title('real')
ax.legend()

fig = plt.figure()
ax = fig.gca()
ax.plot(np.imag(fcs), label = 'measured')
ax.plot(np.imag(fcs_old), label = 'calculated')
ax.set_title('imag')
ax.legend()
plt.show()

import sys
sys.exit()

coeffs = {}
for k in range(-n,n+1):
    coeffs[k] = np.exp(-np.pi * 1.j * k / M) / np.sqrt(M) * fcs[k]

ts = np.linspace(0.,1.,2048)
fs = sum(c * np.exp(2.j * np.pi * k * ts) for k,c in coeffs.items())

fig = plt.figure()
ax = fig.gca()
cycle = [*points, points[0]]
ax.plot(np.real(cycle), np.imag(cycle), marker = '.', color = 'red')
ax.plot(np.real(fs), np.imag(fs), color = 'blue')
plt.show()
