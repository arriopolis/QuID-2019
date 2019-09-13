import cirq

def QFT(qubits):
    # Note: this is actually an implementation of the inverse quantum Fourier transform.
    # This is to agree with the classical interpretation of the Fourier transform.
    circuit = cirq.Circuit()
    n = len(qubits)
    for i in range(n):
        circuit.append(cirq.H(qubits[i]))
        for j in range(i+1,n):
            circuit.append(cirq.CZ(qubits[j],qubits[i])**(-1/(2**(j-i))))
    for i in range(n//2):
        circuit.append(cirq.CNOT(qubits[i], qubits[n-i-1]))
        circuit.append(cirq.CNOT(qubits[n-i-1], qubits[i]))
        circuit.append(cirq.CNOT(qubits[i], qubits[n-i-1]))
    return circuit

if __name__ == "__main__":
    n = 3
    print(QFT([cirq.GridQubit(0,i) for i in range(n)]))
