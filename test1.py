import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Basic gates as matrices
GATES = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]]),
    "H": (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]]),
    "S": np.array([[1, 0], [0, 1j]]),
    "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
}

# Converts qubit state to Bloch vector (x, y, z)
def state_to_bloch(psi):
    a, b = psi[0], psi[1]
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(a) * b)
    z = np.abs(a)**2 - np.abs(b)**2
    return np.real([x, y, z])

# Apply quantum gate
def apply_gate(psi, gate):
    return gate @ psi

# Plot Bloch sphere with swapped X and Y axes
def plot_bloch(state_vector, gate_name="I"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Reduced resolution for faster plotting
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(y, x, z, color='pink', alpha=0.3, edgecolor='none')  # Swap X <-> Y

    # Axes (swapped)
    axis_len = 1.2
    ax.plot([0, 0], [0, axis_len], [0, 0], 'k')         # +X (now on Y axis)
    ax.plot([0, 0], [0, -axis_len], [0, 0], 'k--')       # -X
    ax.plot([0, axis_len], [0, 0], [0, 0], 'k')          # +Y (now on X axis)
    ax.plot([0, -axis_len], [0, 0], [0, 0], 'k--')       # -Y
    ax.plot([0, 0], [0, 0], [0, axis_len], 'k')          # +Z
    ax.plot([0, 0], [0, 0], [0, -axis_len], 'k--')       # -Z

    # Axis labels (swapped)
    ax.text(0, axis_len, 0, '+X', color='red', fontsize=12)
    ax.text(0, -axis_len, 0, '-X', color='red', fontsize=12)
    ax.text(axis_len, 0, 0, '+Y', color='red', fontsize=12)
    ax.text(-axis_len, 0, 0, '-Y', color='red', fontsize=12)
    ax.text(0, 0, axis_len, '+Z', color='red', fontsize=12)
    ax.text(0, 0, -axis_len, '-Z', color='red', fontsize=12)

    # State vector (swapped X and Y)
    bx, by, bz = state_to_bloch(state_vector)
    ax.quiver(0, 0, 0, by, bx, bz, color='orange', linewidth=2)
    ax.text(by, bx, bz, f"|ψ⟩ after {gate_name}", color='blue')

    # View and layout
    ax.view_init(elev=0, azim=135)
    ax.set_xlim([-axis_len, axis_len])
    ax.set_ylim([-axis_len, axis_len])
    ax.set_zlim([-axis_len, axis_len])
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

# Initial state |0⟩
psi_0 = np.array([1, 0], dtype=complex)

# Select gate to apply
selected_gate = "H"  # Change to any of: "I", "X", "Y", "Z", "H", "S", "T"

# Apply gate and plot
psi_final = apply_gate(psi_0, GATES[selected_gate])
print(f"Applying gate: {selected_gate}")
plot_bloch(psi_final, gate_name=selected_gate)
