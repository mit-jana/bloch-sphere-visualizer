import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define gates
GATES = {
    'I': np.eye(2),
    'H': (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]]),
    'X': np.array([[0, 1], [1, 0]]),
    'Y': np.array([[0, -1j], [1j, 0]]),
    'Z': np.array([[1, 0], [0, -1]]),
    'S': np.array([[1, 0], [0, 1j]]),
    'T': (1/np.sqrt(2)) * np.array([[1, 0], [0, (1 + 1j)/np.sqrt(2)]]),
    'A': (1/np.sqrt(2)) * np.array([[0, 1 - 1j], [1 + 1j, 0]]),
    'J': (1/np.sqrt(2)) * np.array([[1, -1j], [1j, -1]])
}

def spherical_to_bloch(theta, phi):
    return np.array([
        np.cos(theta / 2),
        np.exp(1j * phi) * np.sin(theta / 2)
    ])

def bloch_coordinates(qubit):
    a, b = qubit
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(a) * b)
    z = np.abs(a)**2 - np.abs(b)**2
    return x, y, z

def plot_multi_bloch(coords, labels):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='cyan', alpha=0.3, edgecolor='none')

    # Plot all states
    for i, ((x, y, z), label) in enumerate(zip(coords, labels)):
        ax.quiver(0, 0, 0, x, y, z, arrow_length_ratio=0.08, linewidth=2, label=label)
        ax.text(x, y, z, f"{label}", fontsize=9, color='black')

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Bloch Sphere: Gate-by-Gate Evolution")
    ax.legend()

    st.pyplot(fig)

# --- Streamlit UI ---
st.title("Quantum Bloch Sphere Visualizer with Gate Sequence")

theta_deg = st.slider("Theta (θ) in degrees", 0, 180, 45)
phi_deg = st.slider("Phi (φ) in degrees", 0, 360, 90)
gate_sequence = st.text_input("Enter gate sequence (e.g., HXYHZ)", "HXHY")

theta = np.radians(theta_deg)
phi = np.radians(phi_deg)

# Initial state
psi = spherical_to_bloch(theta, phi)
states = [psi]
labels = ["Start"]

# Apply each gate
for i, gate_symbol in enumerate(gate_sequence):
    if gate_symbol not in GATES:
        st.error(f"Invalid gate '{gate_symbol}' in sequence.")
        st.stop()
    gate = GATES[gate_symbol]
    psi = gate @ psi
    states.append(psi)
    labels.append(f"{gate_symbol}_{i+1}")

# Convert all states to Bloch coordinates
coords = [bloch_coordinates(q) for q in states]

# Show Bloch sphere
plot_multi_bloch(coords, labels)
