import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
import cirq
from functools import reduce

from state_construction import construct_state_circuit
from tomography import perform_tomography
from QFT import QFT
from generate_points import points_from_file, generate_points

def points2coeffs(points, cc = True, qft = True, fst = True):
    if not cc and not qft and not fst:
        return np.fft.fft(points, norm = 'ortho')
    elif not cc and qft and not fst:
        n = int(np.log2(len(points)))
        qubits = [cirq.GridQubit(0,i) for i in range(n)]
        circuit = cirq.Circuit()
        circuit.append(QFT(qubits))
        return cirq.Simulator().simulate(circuit, qubit_order = qubits, initial_state = points.astype(np.complex64)).final_state
    elif cc and qft and not fst:
        n = int(np.log2(len(points)))
        qubits = [cirq.GridQubit(0,i) for i in range(n)]
        circuit = cirq.Circuit()
        circuit.append(construct_state_circuit(points, qubits))
        circuit.append(QFT(qubits))
        return cirq.Simulator().simulate(circuit, qubit_order = qubits).final_state
    elif not cc and qft and fst:
        n = int(np.log2(len(points)))
        qubits = [cirq.GridQubit(0,i) for i in range(n)]
        circuit = cirq.Circuit()
        circuit.append(QFT(qubits))
        return perform_tomography(circuit, qubits, initial_state = points.astype(np.complex64))
    elif cc and qft and fst:
        n = int(np.log2(len(points)))
        qubits = [cirq.GridQubit(0,i) for i in range(n)]
        circuit = cirq.Circuit()
        circuit.append(construct_state_circuit(points, qubits))
        circuit.append(QFT(qubits))
        return perform_tomography(circuit, qubits)
    else:
        raise NotImplementedError("This combination is not supported.")

if __name__ == "__main__":
    # name = 'elephant'
    # global_phase = -1/10
    # spacing = 16

    name = 'batman'
    global_phase = -1/20
    spacing = 8

    points = points_from_file('img/{}.txt'.format(name), spacing = spacing)

    points /= np.linalg.norm(points)

    n = 16  # Number of terms
    M = len(points)

    first_fcs = None
    for cc,qft,fst in [(False,False,False),(True,True,True)]:
        fcs = points2coeffs(points, cc = cc, qft = qft, fst = fst)
        if first_fcs is None:
            first_fcs = fcs
        else:
            inp = np.sum(np.conj(first_fcs) * fcs)
            if abs(inp) > 1e-4:
                fcs /= np.exp(1.j * np.angle(inp))

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
        ax.set_title('cc = {}, qft = {}, fst = {}'.format(cc,qft,fst))
    plt.show()
