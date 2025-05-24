import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define gates
GATES = {
    'I': np.eye(2),
    'H': (1/np.sqrt(2)) * np.array([[1, 1],
                                    [1, -1]]),
    'X': np.array([[0, 1],
                   [1, 0]]),
    'Y': np.array([[0, -1j],
                   [1j, 0]]),
    'Z': np.array([[1, 0],
                   [0, -1]]),
    'A': (1/np.sqrt(2)) * np.array([[0, 1 - 1j],
                                    [1 + 1j, 0]])
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
    return x, y, z  # Standard Bloch coordinates

def visualize_gate_effect(theta_deg, phi_deg, gate_name):
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    
    psi = spherical_to_bloch(theta, phi)
    x1, y1, z1 = bloch_coordinates(psi)

    gate = GATES.get(gate_name.upper())
    if gate is None:
        print(f"Unknown gate '{gate_name}'")
        return

    psi_new = gate @ psi
    x2, y2, z2 = bloch_coordinates(psi_new)

    print(f"\nGate applied: {gate_name.upper()}")
    print(f"Original Bloch vector:   X = {x1:.4f}, Y = {y1:.4f}, Z = {z1:.4f}")
    print(f"After Gate {gate_name}:  X = {x2:.4f}, Y = {y2:.4f}, Z = {z2:.4f}")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Sphere surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='lightblue', alpha=0.3, edgecolor='none')

    # Original state vector
    ax.quiver(0, 0, 0, x1, y1, z1, color='blue', linewidth=2, arrow_length_ratio=0.08)
    ax.text(x1, y1, z1, "Original", color='blue', fontsize=12)

    # After gate applied
    ax.quiver(0, 0, 0, x2, y2, z2, color='orange', linewidth=2, arrow_length_ratio=0.08)
    ax.text(x2, y2, z2, f"After {gate_name.upper()}", color='orange', fontsize=12)

    # Axes labels
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Y", fontsize=14)
    ax.set_zlabel("Z", fontsize=14)
    ax.set_title(f"Effect of Gate {gate_name.upper()} on Bloch Sphere",
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.show()

# === User Input ===
theta_input = float(input("Enter theta (θ) in degrees (0 to 180): "))
phi_input = float(input("Enter phi (φ) in degrees (0 to 360): "))

print("\nAvailable gates: I, H, X, Y, Z, A")
gate_input = input("Enter the gate to apply: ")

visualize_gate_effect(theta_input, phi_input, gate_input)
