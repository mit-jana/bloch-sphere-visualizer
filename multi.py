import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define gates
GATES = {
    'I (Identity)': np.eye(2),
    'H (Hadamard)': (1/np.sqrt(2)) * np.array([[1, 1],
                                               [1, -1]]),
    'X (Pauli-X)': np.array([[0, 1],
                             [1, 0]]),
    'Y (Pauli-Y)': np.array([[0, -1j],
                             [1j, 0]]),
    'Z (Pauli-Z)': np.array([[1, 0],
                             [0, -1]]),
    'S (Pauli-S)': np.array([[1, 0],
                             [0, 1j]]),
    'T (Pauli-T)': (1/np.sqrt(2)) * np.array([[1, 0],
                                              [0, ((1 + 1j)/np.sqrt(2))]]),
    'A (Custom)': (1/np.sqrt(2)) * np.array([[0, 1 - 1j],
                                             [1 + 1j, 0]]),
    'J (Custom)': (1/np.sqrt(2)) * np.array([[1, -1j],
                                             [1j, -1]])
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

def plot_bloch_sphere(x1, y1, z1, x2, y2, z2, gate_names):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='cyan', alpha=0.3, edgecolor='none')

    ax.quiver(0, 0, 0, x1, y1, z1, color='blue', linewidth=2, arrow_length_ratio=0.08)
    ax.text(x1, y1, z1, "Original", color='blue')

    ax.quiver(0, 0, 0, x2, y2, z2, color='orange', linewidth=2, arrow_length_ratio=0.08)
    ax.text(x2, y2, z2, f"Final", color='orange')

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Bloch Sphere: {' → '.join(gate_names)}")

    st.pyplot(fig)

# --- Streamlit UI ---
st.title("Quantum Bloch Sphere Visualizer")

theta_deg = st.slider("Theta (θ) in degrees", 0, 180, 45)
phi_deg = st.slider("Phi (φ) in degrees", 0, 360, 90)
gate_names = st.multiselect("Choose quantum gates (in order)", list(GATES.keys()), default=['H (Hadamard)'])

theta = np.radians(theta_deg)
phi = np.radians(phi_deg)

# Initial state
psi = spherical_to_bloch(theta, phi)
x1, y1, z1 = bloch_coordinates(psi)

# Apply gates sequentially
resultant_matrix = np.eye(2)
for gate in gate_names:
    resultant_matrix = GATES[gate] @ resultant_matrix

# Apply final transformation
psi_new = resultant_matrix @ psi
x2, y2, z2 = bloch_coordinates(psi_new)

st.write(f"### Original Coordinates")
st.write(f"X: {x1:.4f}, Y: {y1:.4f}, Z: {z1:.4f}")

st.write(f"### Final Coordinates after applying `{' → '.join(gate_names)}`")
st.write(f"X: {x2:.4f}, Y: {y2:.4f}, Z: {z2:.4f}")

plot_bloch_sphere(x1, y1, z1, x2, y2, z2, gate_names)
