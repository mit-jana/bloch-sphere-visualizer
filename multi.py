import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

# Define gates
GATES = {
    'I': np.eye(2),
    'H': (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]]),
    'X': np.array([[0, 1], [1, 0]]),
    'Y': np.array([[0, -1j], [1j, 0]]),
    'Z': np.array([[1, 0], [0, -1]]),
    'S': np.array([[1, 0], [0, 1j]]),
    'T': (1 / np.sqrt(2)) * np.array([[1, 0], [0, (1 + 1j) / np.sqrt(2)]]),
    'A': (1 / np.sqrt(2)) * np.array([[0, 1 - 1j], [1 + 1j, 0]]),
    'J': (1 / np.sqrt(2)) * np.array([[1, -1j], [1j, -1]])
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

def parse_gate_sequence(seq):
    tokens = re.findall(r'([A-Za-z])(\d*)', seq)
    expanded = []
    for gate, count in tokens:
        gate = gate.upper()
        if gate not in GATES:
            raise ValueError(f"Invalid gate: {gate}")
        count = int(count) if count else 1
        expanded.extend([gate] * count)
    return expanded

def plot_bloch_sphere(coords_list, gate_seq):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the Bloch sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='cyan', alpha=0.3, edgecolor='none')

    # Plot qubit states
    colors = plt.cm.plasma(np.linspace(0, 1, len(coords_list)))
    for i, (x, y, z) in enumerate(coords_list):
        label = "Original" if i == 0 else gate_seq[i-1]
        ax.quiver(0, 0, 0, x, y, z, color=colors[i], linewidth=2, arrow_length_ratio=0.08)
        ax.text(x, y, z, label, color=colors[i])

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Bloch Sphere: Intermediate States")

    st.pyplot(fig)

# --- Streamlit UI ---
st.title("Quantum Bloch Sphere Visualizer with Multiple Gates")

theta_deg = st.slider("Theta (θ) in degrees", 0, 180, 45)
phi_deg = st.slider("Phi (φ) in degrees", 0, 360, 90)

gate_input = st.text_input("Enter gate sequence (e.g., H3X2Y or XZHHT)", "H3X2Y")

try:
    gate_sequence = parse_gate_sequence(gate_input)

    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    psi = spherical_to_bloch(theta, phi)

    # Store intermediate states
    coords = [bloch_coordinates(psi.copy())]

    for g in gate_sequence:
        psi = GATES[g] @ psi
        coords.append(bloch_coordinates(psi.copy()))

    # Show values
    st.write("### Coordinates After Each Step")
    for i, (x, y, z) in enumerate(coords):
        label = "Initial" if i == 0 else f"After {gate_sequence[i-1]}"
        st.write(f"{label}: X={x:.4f}, Y={y:.4f}, Z={z:.4f}")

    # Plot sphere
    plot_bloch_sphere(coords, gate_sequence)

except ValueError as e:
    st.error(str(e))
