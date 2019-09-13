import cirq
import numpy as np
np.set_printoptions(linewidth=200)

def remove_measurements(orig_circuit):
    circuit = cirq.Circuit()
    for moment in orig_circuit._moments:
        new_moment = []
        for j,op in enumerate(moment.operations):
            if not hasattr(op, '_gate') or not isinstance(op._gate, cirq.ops.MeasurementGate):
                new_moment.append(op)
        if len(new_moment) > 0: circuit.append(new_moment)
    return circuit

def invert_perm(perm):
    inv = [None]*len(perm)
    for i,x in enumerate(perm):
        inv[x] = i
    return inv

def generate_permutation_circuit(qubits, perm):
    assert len(perm) == 2**len(qubits)

    # Build up a crude version of the circuit
    n = len(qubits)
    remaining = set(range(2**n))
    total_circuit = cirq.Circuit()
    while remaining:
        x = remaining.pop()
        y = x
        while perm[y] != x:
            orig = list('{{:0{}b}}'.format(n).format(y))
            new = list('{{:0{}b}}'.format(n).format(perm[y]))
            current = orig.copy()
            prepare_circuit = cirq.Circuit()
            for i,(c,d) in enumerate(zip(orig,new)):
                if c != d:
                    x_circuit = cirq.Circuit()
                    gate = cirq.X(qubits[i])
                    for j in range(n):
                        if i != j:
                            gate = gate.controlled_by(qubits[j])
                            if current[j] == '0':
                                x_circuit.append(cirq.X(qubits[j]))

                    swap_circuit = cirq.Circuit()
                    swap_circuit.append(x_circuit)
                    swap_circuit.append(gate)
                    swap_circuit.append(x_circuit)
                    current[i] = d
                    if not all(a == b for a,b in zip(current,new)):
                        prepare_circuit.append(swap_circuit)
                    else:
                        total_circuit.append(prepare_circuit)
                        total_circuit.append(swap_circuit)
                        total_circuit.append(cirq.inverse(prepare_circuit))
                        break
            y = perm[y]
            remaining.remove(y)
    return cirq.inverse(total_circuit)

def perform_tomography(obj, qubits, debug = False, initial_state = None, batch_size = 1024):
    n = len(qubits)

    nres = np.zeros(2**n, dtype = np.int_)
    total_npms = np.zeros((2**n,2**n), dtype = np.int_)
    npms = np.zeros((2**n, 2**n), dtype = np.int_)
    total_ipms = np.zeros((2**n,2**n), dtype = np.int_)
    ipms = np.zeros((2**n, 2**n), dtype = np.int_)

    return tomography_update_measurement_results(obj, qubits, nres, total_npms, npms, total_ipms, ipms,
            debug = debug,
            initial_state = initial_state,
            batch_size = batch_size,
        )

def perform_iterative_tomography(obj, qubits, repetitions = float("Inf"), debug = False, initial_state = None, batch_size = 1024):
    n = len(qubits)

    nres = np.zeros(2**n, dtype = np.int_)
    total_npms = np.zeros((2**n,2**n), dtype = np.int_)
    npms = np.zeros((2**n, 2**n), dtype = np.int_)
    total_ipms = np.zeros((2**n,2**n), dtype = np.int_)
    ipms = np.zeros((2**n, 2**n), dtype = np.int_)

    k = 0
    while k < repetitions:
        k += 1
        yield tomography_update_measurement_results(obj, qubits, nres, total_npms, npms, total_ipms, ipms,
                debug = debug,
                initial_state = initial_state,
                batch_size = batch_size,
            )

def tomography_update_measurement_results(obj, qubits, nres, total_npms, npms, total_ipms, ipms,
            debug = False,
            initial_state = None,
            batch_size = 1024,
        ):
    n = len(qubits)

    nmc = cirq.Circuit()
    nmc.append(obj)
    nmc.append(cirq.measure(*qubits, key = 'amps'))

    if initial_state is None:
        res = cirq.Simulator().run(nmc, repetitions = batch_size).measurements
    else:
        new_nmc = remove_measurements(nmc)
        fs = cirq.Simulator().simulate(new_nmc, qubit_order = qubits, initial_state = np.array(initial_state)).final_state
        ms = np.random.choice(np.arange(2**n), batch_size, p = np.square(np.abs(fs)))
        res = {'amps' : np.array([[True if c == '1' else False for c in '{{:0{}b}}'.format(n).format(m)] for m in ms])}

    for m in res['amps']:
        i = int(''.join('1' if x else '0' for x in m),2)
        nres[i] += 1

    amps = np.sqrt(nres / np.sum(nres))
    if debug:
        print("Amplitude measurements counts:", list(nres))
        print()

    sorted_amps = sorted(enumerate(amps), key = lambda x : x[1], reverse = True)
    perm,perm_amps = zip(*sorted_amps)
    inv_perm = invert_perm(perm)
    pc = generate_permutation_circuit(qubits, invert_perm(perm))

    if debug:
        print("Permutation:", perm)
        print("Inverse permutation circuit:")
        print(pc)
        print("Unitary matrix of the permutation:")
        print(pc.to_unitary_matrix())
        print()

    for i in range(n):
        npmc = cirq.Circuit()
        npmc.append(obj)
        npmc.append(pc)
        npmc.append(cirq.H(qubits[i]))
        npmc.append(cirq.measure(qubits[i], key = 'phase'))
        npmc.append(cirq.measure(*(qubits[:i] + qubits[(i+1):]), key = 'binary'))

        if debug:
            print("Normal phase measurement circuit {}:".format(i))
            print(npmc)

        if initial_state is None:
            res = cirq.Simulator().run(npmc, repetitions = batch_size).measurements
        else:
            new_npmc = remove_measurements(npmc)
            fs = cirq.Simulator().simulate(new_npmc, qubit_order = qubits, initial_state = np.array(initial_state)).final_state
            ms = np.random.choice(np.arange(2**n), batch_size, p = np.square(np.abs(fs)))
            str_ms = ['{{:0{}b}}'.format(n).format(m) for m in ms]
            sep_ms = {'phase' : [m[i] for m in str_ms], 'binary' : [m[:i] + m[(i+1):] for m in str_ms]}
            res = {k : np.array([[True if c == '1' else False for c in s] for s in v]) for k,v in sep_ms.items()}

        for mp,mb in zip(res['phase'], res['binary']):
            x = int(mp)
            b = int(''.join('1' if x else '0' for x in (list(mb[:i]) + [False] + list(mb[i:]))),2)
            j1 = perm[b]
            j2 = perm[b + 2**(n-1-i)]
            j1,j2 = sorted([j1,j2])
            total_npms[j1,j2] += 1
            total_npms[j2,j1] += 1
            npms[j1,j2] += x        # This estimates .5 - r1 * r2 * cos(phi1 - phi2)
            npms[j2,j1] += x

        ipmc = cirq.Circuit()
        ipmc.append(obj)
        ipmc.append(pc)
        ipmc.append(cirq.S(qubits[i]))
        ipmc.append(cirq.H(qubits[i]))
        ipmc.append(cirq.measure(qubits[i], key = 'phase'))
        ipmc.append(cirq.measure(*(qubits[:i] + qubits[(i+1):]), key = 'binary'))

        if debug:
            print("Imaginary phase measurement circuit {}:".format(i))
            print(ipmc)

        if initial_state is None:
            res = cirq.Simulator().run(ipmc, repetitions = batch_size).measurements
        else:
            new_ipmc = remove_measurements(ipmc)
            fs = cirq.Simulator().simulate(new_ipmc, qubit_order = qubits, initial_state = np.array(initial_state)).final_state
            ms = np.random.choice(np.arange(2**n), batch_size, p = np.square(np.abs(fs)))
            str_ms = ['{{:0{}b}}'.format(n).format(m) for m in ms]
            sep_ms = {'phase' : [m[i] for m in str_ms], 'binary' : [m[:i] + m[(i+1):] for m in str_ms]}
            res = {k : np.array([[True if c == '1' else False for c in s] for s in v]) for k,v in sep_ms.items()}

        for mp,mb in zip(res['phase'], res['binary']):
            x = int(mp)
            b = int(''.join('1' if x else '0' for x in (list(mb[:i]) + [False] + list(mb[i:]))),2)
            j1 = perm[b]
            j2 = perm[b + 2**(n-1-i)]
            total_ipms[j1,j2] += 1
            total_ipms[j2,j1] += 1
            ipms[j1,j2] += x        # This estimates .5 - r1 * r2 * sin(phi1 - phi2)
            ipms[j2,j1] += 1-x

        if debug:
            print("Measurement normal phase measurement circuit totals and counts:")
            print(total_npms)
            print(npms)

            print("Measurement imaginary phase measuremnet circuit totals and counts:")
            print(total_ipms)
            print(ipms)

    phases = np.zeros(2**n)
    solved = set([perm[0]])
    total_meas = np.minimum(total_npms, total_ipms)
    for i in perm[1:]:
        if amps[i] < 1e-3: break
        j,ms = max(((j,ms) for j,ms in enumerate(total_meas[i,:]) if j in solved), key = lambda x : x[1])
        if ms == 0: break
        x = (.5 - npms[i,j] / total_npms[i,j]) / (amps[i] * amps[j]) * (amps[i]**2 + amps[j]**2)
        y = (.5 - ipms[i,j] / total_ipms[i,j]) / (amps[i] * amps[j]) * (amps[i]**2 + amps[j]**2)
        phi = np.arctan2(y,x)
        phases[i] = phases[j] + phi
        solved.add(i)
        if debug:
            print("Difference between {} and {}:".format(j,i))
            print("x = {}, y = {}".format(x, y))

    if debug:
        print("Phases:", phases)
        print()

    return amps * np.exp(1.j * phases)

if __name__ == "__main__":
    n = 3
    qubits = [cirq.GridQubit(0,i) for i in range(n)]
    obj = cirq.Circuit()
    obj.append(cirq.H(qubits[0]))
    obj.append(cirq.CNOT(qubits[0], qubits[1]))
    obj.append(cirq.Y(qubits[0])**(1/3))
    obj.append(cirq.CNOT(qubits[1], qubits[0]))
    obj.append(cirq.CNOT(qubits[1], qubits[2]))
    obj.append(cirq.H(qubits[1]))
    obj.append(cirq.S(qubits[1]))
    obj.append(cirq.T(qubits[2]))

    print("Objective circuit:")
    print(obj)
    print()

    coeffs = perform_tomography(obj, qubits, initial_state = np.array([1,0,0,0,0,0,0,0], dtype = np.complex64))
    correct_coeffs = cirq.Simulator().simulate(obj, qubit_order = qubits).final_state
    inp = np.sum(np.conj(correct_coeffs) * coeffs)

    print("Recovered coefficients:")
    print(coeffs)
    print("Correct coefficients:")
    print(correct_coeffs)
    print("Absolute value of the inner product:")
    print(abs(inp))
