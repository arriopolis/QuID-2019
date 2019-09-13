import numpy as np

def points_from_file(filename, spacing = 1):
    points = np.loadtxt(filename, dtype = np.complex_)
    num_qubits = int(np.floor(np.log2(len(points))))
    new_points = np.empty(2**num_qubits, dtype = np.complex64)
    for i,idx in enumerate(np.linspace(0., len(points)-1, 2**num_qubits)):
        new_points[i] = points[int(round(idx))]
    new_points = new_points[::spacing]
    new_points /= np.linalg.norm(new_points)
    return new_points

def generate_points(M):
    # Simple circle with quadratic speed
    # f = lambda t : np.exp(2.j*np.pi*t**2)

    # Weird function
    # f = lambda t : np.exp(2.j * np.pi * t) + .2 * np.exp(2.j * np.pi * 12 * t)

    # Lissajous
    f = lambda t : np.sin(4. * np.pi * t) + np.cos(6. * np.pi * t) * 1.j

    boundaries = np.linspace(0.,1.,M+1)
    midpoints = .5 * (boundaries[:-1] + boundaries[1:])
    return np.array([f(t) for t in midpoints])
