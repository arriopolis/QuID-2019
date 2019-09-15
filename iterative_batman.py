import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
import cirq
from functools import reduce

from state_construction import construct_state_circuit
from tomography import perform_iterative_tomography
from QFT import QFT
from generate_points import points_from_file, generate_points

def points2coeffs(points):
    n = int(np.log2(len(points)))
    qubits = [cirq.GridQubit(0,i) for i in range(n)]
    circuit = cirq.Circuit()
    circuit.append(construct_state_circuit(points, qubits))
    circuit.append(QFT(qubits))
    for coeffs in perform_iterative_tomography(circuit, qubits, batch_size = 2048):
        yield coeffs

if __name__ == "__main__":
    # name = 'elephant'
    # global_phase = -1/10
    # spacing = 16

    name = 'batman'
    global_phase = -1/20
    spacing = 8

    points = points_from_file('img/{}.txt'.format(name), spacing = spacing)

    points /= np.linalg.norm(points)
    points /= np.exp(1.j * np.angle(points[0]))

    n = 16  # Number of terms
    M = len(points)

    fig = plt.figure()
    for x,fcs in enumerate(points2coeffs(points)):
        print("Iteration:", x)
        fcs /= np.exp(1.j * np.angle(fcs[0]))
        fcs *= np.exp(2.j * np.pi * global_phase)

        coeffs = {}
        for k in range(-n,n+1):
            coeffs[k] = np.exp(-np.pi * 1.j * k / M) / np.sqrt(M) * fcs[k]

        ts = np.linspace(0.,1.,2048)
        fs = sum(c * np.exp(2.j * np.pi * k * ts) for k,c in coeffs.items())

        fig.clf()
        ax = fig.gca()
        cycle = [*points, points[0]]
        ax.plot(np.real(cycle), np.imag(cycle), marker = '.', color = 'red')
        ax.plot(np.real(fs), np.imag(fs), color = 'blue')
        ax.set_title('iteration = {}'.format(x))

        plt.draw()
        plt.pause(.01)
    plt.show()
