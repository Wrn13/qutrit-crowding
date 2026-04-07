import matplotlib.pyplot as plt
import numpy as np
from qutip import average_gate_fidelity, destroy, qeye, tensor
from scipy.optimize import curve_fit


# %%
def simulate_infidelity(
    detuning_list, intended_term, ideal_gate, prefactor, spectator_term
):
    """Runs QuTiP simulations to compute the infidelity vs. detuning."""
    infidelity_list = []

    for detuning in detuning_list:
        spectator_amplitude = (2 * prefactor) / (2 * np.pi * detuning * 1e6)
        H = (np.pi / 2) * intended_term + spectator_amplitude * spectator_term
        U_t_f = (-1.0j * H).expm()
        fidelity = 1 - average_gate_fidelity(U_t_f, ideal_gate)
        infidelity_list.append(fidelity)

    return np.array(infidelity_list)


# def decay_fit(detuning, a, b, c, d):
#     """Power law function for fitting infidelity curves."""
#     return a * ((detuning + d) ** -b) + c


# def fit_infidelity(detuning_list, infidelity_list):
#     """Fits the simulated infidelity data to the power-law model.

#     Returns:
#         - Best-fit parameters (a, b, c)
#     """
#     p0 = [1, 2, 0, 0]  # Initial guess
#     params, _ = curve_fit(decay_fit, detuning_list, infidelity_list, p0=p0)
#     return params

import numpy as np
from scipy.optimize import curve_fit


def decay_fit(detuning, x0, x1):
    """Modified power law function for fitting infidelity curves."""
    return x0 * ((2 / (detuning + x1)) ** 2)


def fit_infidelity(detuning_list, infidelity_list):
    """Fits the infidelity data to the modified power-law model with better convergence."""
    p0 = [1, 1]  # Improved initial guess
    params, _ = curve_fit(decay_fit, detuning_list, infidelity_list, p0=p0)
    # print(params)
    return params


# %%
def compute_infidelity_parameters(detuning_list, lambdaq, eta, alpha, g3):
    """Generates (a, b, c) infidelity parameters dynamically from QuTiP simulations."""
    # Compute prefactors
    intra_prefactors = {
        "snail-qubit": 6 * eta * lambdaq * g3,
        "qubit-sub": 3 * eta**2 * lambdaq * g3,
        "qubit-qubit": 6 * eta * lambdaq**2 * g3,
    }

    inter_prefactors = {
        "snail-qubit (inter)": 6 * eta * lambdaq**3 * g3,
        "qubit-sub (inter)": 3 * eta**2 * lambdaq**3 * g3,
    }

    # Combine all prefactors
    prefactors = {**intra_prefactors, **inter_prefactors}

    # === Hilbert Space 1: Three Qutrit System === #
    q = destroy(3)
    q1 = tensor(q, qeye(3), qeye(3))  # First qutrit (Main interaction)
    q2 = tensor(qeye(3), q, qeye(3))  # Second qutrit (Main interaction)
    q3 = tensor(qeye(3), qeye(3), q)  # Spectator qutrit
    q1dag = q1.dag()
    q2dag = q2.dag()
    q3dag = q3.dag()

    # Intended gate (always between q1 and q2)
    intended_term_qubits = q1dag * q2 + q1 * q2dag
    ideal_gate_qubits = (-1.0j * (np.pi / 2) * intended_term_qubits).expm()

    # Spectator terms for qubit system
    spectator_ops_qubits = {
        "qubit-qubit": (q1dag * q3 + q1 * q3dag, ideal_gate_qubits),
        "qubit-sub": (q3dag + q3, ideal_gate_qubits),
        "qubit-sub (inter)": (q1dag + q1, ideal_gate_qubits),
    }

    # === Hilbert Space 2: Two Qutrits + SNAIL System === #
    n_dim_snail = 8  # 8-Level SNAIL Mode
    qs = destroy(3)  # Qubit part of SNAIL system
    s = destroy(n_dim_snail)  # SNAIL oscillator
    qs1 = tensor(qs, qeye(3), qeye(n_dim_snail))  # First qutrit
    qs2 = tensor(qeye(3), qs, qeye(n_dim_snail))  # Second qubit
    s1 = tensor(qeye(3), qeye(3), s)  # SNAIL mode
    qs1dag = qs1.dag()
    qs2dag = qs2.dag()
    s1dag = s1.dag()

    # Intended gate (always between qs1 and qs2)
    intended_term_snail = qs1dag * qs2 + qs1 * qs2dag
    ideal_gate_snail = (-1.0j * (np.pi / 2) * intended_term_snail).expm()

    # Spectator terms for SNAIL system
    spectator_ops_snail = {
        "snail-qubit": (qs1dag * s1 + qs1 * s1dag, ideal_gate_snail),
        "snail-qubit (inter)": (qs1dag * s1 + qs1 * s1dag, ideal_gate_snail),
    }

    # Compute infidelity curves and fit (a, b, c)
    infidelity_params = {}
    fidelity_results = {}

    # Compute for qubit-based spectators
    for key in spectator_ops_qubits:
        spectator_term, gate_target = spectator_ops_qubits[key]
        fidelity_results[key] = simulate_infidelity(
            detuning_list,
            intended_term_qubits,
            gate_target,
            prefactors[key],
            spectator_term,
        )
        infidelity_params[key] = fit_infidelity(detuning_list, fidelity_results[key])

    # Compute for SNAIL-based spectators
    for key in spectator_ops_snail:
        spectator_term, gate_target = spectator_ops_snail[key]
        fidelity_results[key] = simulate_infidelity(
            detuning_list,
            intended_term_snail,
            gate_target,
            prefactors[key],
            spectator_term,
        )
        infidelity_params[key] = fit_infidelity(detuning_list, fidelity_results[key])

    return infidelity_params, fidelity_results
