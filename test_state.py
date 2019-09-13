import numpy as np
np.set_printoptions(linewidth=200)
import cirq
from state_construction import construct_state_circuit

def real_world_state():
    f = lambda t : np.sin(4. * np.pi * t) + np.cos(6. * np.pi * t) * 1.j

    n = 6
    M = 2**n
    boundaries = np.linspace(0.,1.,M+1)
    midpoints = .5 * (boundaries[:-1] + boundaries[1:])
    points = np.array([f(t) for t in midpoints])
    return points / np.linalg.norm(points)

def test_state(n):
    state = np.random.rand(2**n) + 1.j * np.random.rand(2**n)
    return state / np.linalg.norm(state)

def bad_state():
    state = np.array([0.06632473+0.12046373j,0.60702249+0.50764989j,0.52591309+0.25776855j,0.07458765+0.07944992j])
    return state / np.linalg.norm(state)

n = 2
state = bad_state()

qubits = [cirq.GridQubit(0,i) for i in range(n)]
circuit = construct_state_circuit(state, qubits, debug = True)
constructed_state = cirq.Simulator().simulate(circuit, qubit_order = qubits).final_state
inp = np.sum(np.conj(state) * constructed_state)
corrected_state = constructed_state / np.exp(1.j * np.angle(inp))

print("Construction circuit:")
print(circuit)
print()

print("Constructed state (corrected for global phase) and correct state:")
print(corrected_state)
print(state)
print()

print("Amplitudes of constructed and correct state:")
print(np.abs(corrected_state))
print(np.abs(state))
print()

print("Phases of constructed and correct state:")
print(np.angle(corrected_state))
print(np.angle(state))
print()

print("Absolute value of the inner product:")
print(abs(inp))
