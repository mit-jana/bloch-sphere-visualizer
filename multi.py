import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re

# Quantum gates
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

# Quantum helpers
def spherical_to_bloch(theta, phi):
    return np.array([
        np.cos(theta / 2),
        np.exp(1j * phi) * np.sin(theta / 2)
    ])

def bloch_coordinates(qubit):
    a, b = qubit
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(a) * b)
    z = np.abs(a) ** 2 - np.abs(b) ** 2
    return x, y, z

# Parser for gate sequences like H2X1Y3
def parse_gate_sequence(input_str):
    tokens = re.findall(r'([A-Za-z])(\d*)', input_str.upper())
    gate_list = []
    for g, n in tokens:
        if g in GATES:
            gate_list.extend([g] * (int(n) if n else 1))
    return gate_list

# Plot single vector
def plot_single_bloch(x, y, z, label):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='cyan', alpha=0.3, edgecolor='none')

    ax.quiver(0, 0, 0, x, y, z, color='orange', linewidth=2, arrow_length_ratio=0.08)
    ax.text(x, y, z, label, color='orange')

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Bloch Sphere: {label}")
    st.pyplot(fig)

# Main App
st.title("Quantum Gate Visualizer on Bloch Sphere")

theta_deg = st.slider("Theta (θ) in degrees", 0, 180, 45)
phi_deg = st.slider("Phi (φ) in degrees", 0, 360, 90)
gate_input = st.text_input("Enter gate sequence (e.g. H3X2Y1)", "H3X2Y1")

theta = np.radians(theta_deg)
phi = np.radians(phi_deg)
psi = spherical_to_bloch(theta, phi)

gate_sequence = parse_gate_sequence(gate_input)

states = [psi]
labels = ["Initial"]
amplitudes = [psi]
coords = [bloch_coordinates(psi)]
gate_counts = {}

for gate in gate_sequence:
    gate_counts[gate] = gate_counts.get(gate, 0) + 1
    label = f"{gate}_{gate_counts[gate]}"
    psi = GATES[gate] @ psi
    states.append(psi)
    amplitudes.append(psi)
    coords.append(bloch_coordinates(psi))
    labels.append(label)

# Slider
step = st.slider("Step", 0, len(labels)-1, 0, format="Step %d")

st.write(f"### Gate Step: `{labels[step]}`")
x, y, z = coords[step]
plot_single_bloch(x, y, z, labels[step])

# State Table
st.write("### Quantum State Table")
import pandas as pd
data = []
for i, (label, amp, (x, y, z)) in enumerate(zip(labels, amplitudes, coords)):
    alpha, beta = amp
    data.append({
        "Step": label,
        "α (real)": round(np.real(alpha), 4),
        "α (imag)": round(np.imag(alpha), 4),
        "β (real)": round(np.real(beta), 4),
        "β (imag)": round(np.imag(beta), 4),
        "X": round(x, 4),
        "Y": round(y, 4),
        "Z": round(z, 4)
    })
df = pd.DataFrame(data)
st.dataframe(df, use_container_width=True)
