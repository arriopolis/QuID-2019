import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
import cirq

f = lambda coeffs, ts : sum(c * np.exp(2.j * np.pi * j * ts) / np.sqrt(len(coeffs)) for j,c in enumerate(coeffs))

def points2coeffs(points):
    return np.fft.fft(points, norm = 'ortho')

def points2construction_circuit(points, qubits):
    assert 2**len(qubits) == len(points)
    circuit = cirq.Circuit()
    circuit.append(cirq.X(qubits))

n = 4
points = np.array([np.exp(2.j*np.pi*t**2) for t in np.linspace(0.,1.,2**n,endpoint = False)])
points /= np.linalg.norm(points)

coeffs = points2coeffs(points)
fs = f(coeffs, np.linspace(0., 1., 200))

# qubits = [cirq.GridQubit(0,i) for i in range(4)]
# circuit = points2construction_circuit(points, qubits)
# print(circuit.controlled_by(cirq.GridQubit(0,5)))

fig = plt.figure()
ax = fig.gca()
cycle = [*points, points[0]]
ax.plot(np.real(cycle), np.imag(cycle), marker = '.', color = 'red')
ax.plot(np.real(fs), np.imag(fs), color = 'blue')
plt.show()
