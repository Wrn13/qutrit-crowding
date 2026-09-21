import warnings

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
        #print(U_t_f)
        fidelity = 1 - average_gate_fidelity(U_t_f, ideal_gate)
        infidelity_list.append(fidelity)

    return np.array(infidelity_list)

def simulate_drag_infidelity(
    detuning_list,
    intended_term,
    ideal_gate,
    prefactor,
    diagonal_op,
    g_tgt,
    induced_op=None,
    induced_resonant=False,
    stark_factor=1.0
):
    """Infidelity vs. detuning AFTER DRAG removes the first-order spectator.

    Args:
        detuning_list (array-like):
            Spectator detunings in MHz (matches Evan's convention).
        intended_term (Qobj):
            Activated exchange generator (q_a^dag q_b + h.c.). The gate angle is pi/2.
        ideal_gate (Qobj):
            Target unitary exp(-i (pi/2) intended_term).
        prefactor (float):
            Table 1 coupling g_s in rad/s (eta already included).
        diagonal_op (Qobj):
            Second-order diagonal residual operator, e.g. (n_a - n_b) or n_a.
        g_tgt (float):
            Activated target rate in rad/s; sets T_gate = (pi/2)/g_tgt.
        induced_op (Qobj | None):
            Induced-exchange operator (q_b^dag q_c + h.c.) from [i G_s, H_tgt].
        induced_resonant (bool):
            Include the induced term only if its b<->c transition is resonant.
        stark_factor (float):
            Coefficient on the diagonal residual: 1 for exchange spectators
            ([O,O^dag] = n_a - n_b), 2 for direct single-mode drives
            ([q,q^dag] = 1 - 2n -> 2 g_s^2/Delta on n).
 
    Returns:
        np.ndarray: infidelities of the target gate.
    """
    T_gate = (np.pi / 2) / g_tgt
    infidelity_list = []
 
    for detuning in detuning_list:
        Delta = 2 * np.pi * detuning * 1e6  # rad/s
 
        # Diagonal Stark/cross-Kerr residual: angle = (g_s^2 / Delta) * T_gate
        diag_angle = stark_factor * (prefactor**2 / Delta) * T_gate
 
        H = (np.pi / 2) * intended_term + diag_angle * diagonal_op
 
        # Induced exchange (only when the b<->c transition is resonant):
        # angle = (g_s / Delta) * g_tgt * T_gate = (g_s / Delta) * (pi/2)
        if induced_op is not None and induced_resonant:
            induced_angle = (prefactor / Delta) * (np.pi / 2)
            H = H + induced_angle * induced_op
 
        U_t_f = (-1.0j * H).expm()
        infidelity_list.append(1 - average_gate_fidelity(U_t_f, ideal_gate))
 
    return np.array(infidelity_list)

import numpy as np
from scipy.optimize import curve_fit


def decay_fit(detuning, x0, x1):
    """Modified power law function for fitting infidelity curves."""
    return x0 * ((2 / (detuning + x1)) ** 2)


def fit_infidelity(detuning_list, infidelity_list, label=None):
    """Fits the infidelity data to the modified power-law model with better convergence.

    Warns if `decay_fit` cannot actually describe the data. The monotonic power law only
    fits a perturbative spectator (g_s/Delta << 1); a strongly driven one saturates and
    turns over in detuning, and curve_fit will still return *something*, so the residual
    has to be checked rather than trusted.
    """
    p0 = [1, 1]  # Improved initial guess
    # maxfev raised from the default 600: the stiffer spectators (e.g. a direct drive
    # on the multi-level SNAIL mode) need more iterations to converge.
    params, _ = curve_fit(
        decay_fit, detuning_list, infidelity_list, p0=p0, maxfev=20000
    )
    fitted = decay_fit(np.asarray(detuning_list), *params)
    residual = np.max(np.abs(fitted - infidelity_list))
    scale = np.max(np.abs(infidelity_list))
    if scale > 0 and residual / scale > 0.05:
        warnings.warn(
            f"decay_fit is a poor model for {label or 'this spectator'}: max residual "
            f"{residual:.3e} is {residual / scale:.0%} of the peak infidelity "
            f"{scale:.3e}. The spectator is likely non-perturbative (simulated "
            f"infidelity saturating or non-monotonic in detuning), so these fit "
            f"parameters should not be trusted.",
            RuntimeWarning,
            stacklevel=2,
        )
    # print(params)
    return params


# %%
def compute_infidelity_parameters(
    detuning_list, lambdaq, eta, alpha, g3, snail_sub_participation=None
):
    """Generates (a, b, c) infidelity parameters dynamically from QuTiP simulations.

    Parameters
    ----------
    snail_sub_participation : float, optional
        Mode participation for the "snail-sub" spectator -- a second-order
        (two-pump-photon) direct drive on the SNAIL mode at f_SNAIL/2. Defaults to
        `lambdaq`, which keeps the term perturbative and fittable by `decay_fit`.

        Counting participations the way the other prefactors do (one factor of
        `lambdaq` per *qubit* mode in the operator, 1 for the SNAIL mode) argues for
        1.0 here. That is deliberately not the default: at 1.0 the prefactor is
        3*eta**2*g3 ~ 583 MHz, so g_s/Delta ranges 0.6-3.7 over a 50-1000 MHz
        detuning sweep. The dynamics are then far outside the perturbative regime
        `simulate_infidelity` assumes -- the simulated infidelity runs 0.21-0.97 and is
        *non-monotonic* in detuning, which the monotonic `decay_fit` power law cannot
        represent (best-case residual ~0.26). Physically that says the SNAIL
        subharmonic is not a perturbative spectator at all but a hard exclusion zone,
        which is why `speedlimit_fit.py` handles it as a pump-power ceiling instead.
        Pass 1.0 only if you also replace the fitting ansatz.
    """
    if snail_sub_participation is None:
        snail_sub_participation = lambdaq

    # Compute prefactors
    intra_prefactors = {
        "snail-qubit": 6 * eta * lambdaq * g3,
        "qubit-sub": 3 * eta**2 * lambdaq * g3,
        "qubit-qubit": 6 * eta * lambdaq**2 * g3,
        # SNAIL subharmonic: same second-order direct-drive structure as "qubit-sub",
        # but the driven mode is the multi-level SNAIL rather than a qubit.
        "snail-sub": 3 * eta**2 * snail_sub_participation * g3,
    }

    inter_prefactors = {
        "snail-qubit (inter)": 6 * eta * lambdaq**3 * g3,
        "qubit-sub (inter)": 3 * eta**2 * lambdaq**3 * g3,
    }

    # Combine all prefactors
    prefactors = {**intra_prefactors, **inter_prefactors}

    # === Hilbert Space 1: Three Qubit System === #
    q = destroy(2)
    q1 = tensor(q, qeye(2), qeye(2))  # First qubit (Main interaction)
    q2 = tensor(qeye(2), q, qeye(2))  # Second qubit (Main interaction)
    q3 = tensor(qeye(2), qeye(2), q)  # Spectator qubit
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

    # === Hilbert Space 2: Two Qubits + SNAIL System === #
    n_dim_snail = 8  # 8-Level SNAIL Mode
    qs = destroy(2)  # Qubit part of SNAIL system
    s = destroy(n_dim_snail)  # SNAIL oscillator
    qs1 = tensor(qs, qeye(2), qeye(n_dim_snail))  # First qubit
    qs2 = tensor(qeye(2), qs, qeye(n_dim_snail))  # Second qubit
    s1 = tensor(qeye(2), qeye(2), s)  # SNAIL mode
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
        # Direct drive on the 8-level SNAIL mode: pumping at f_SNAIL/2 populates the
        # SNAIL's higher levels, which is the leakage this term is meant to capture.
        "snail-sub": (s1dag + s1, ideal_gate_snail),
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
        infidelity_params[key] = fit_infidelity(
            detuning_list, fidelity_results[key], label=key
        )

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
        infidelity_params[key] = fit_infidelity(
            detuning_list, fidelity_results[key], label=key
        )

    return infidelity_params, fidelity_results

# %%
def compute_dragged_infidelity_parameters(detuning_list, lambdaq, eta, alpha, g3):
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

    g_tgt = intra_prefactors["qubit-qubit"]
    print(g_tgt)

    # Combine all prefactors
    prefactors = {**intra_prefactors, **inter_prefactors}

    # === Hilbert Space 1: Three Qubit System === #
    q = destroy(2)
    q1 = tensor(q, qeye(2), qeye(2))  # First qubit (Main interaction)
    q2 = tensor(qeye(2), q, qeye(2))  # Second qubit (Main interaction)
    q3 = tensor(qeye(2), qeye(2), q)  # Spectator qubit
    q1dag = q1.dag()
    q2dag = q2.dag()
    q3dag = q3.dag()
    n1, n2, n3 = q1dag * q1, q2dag * q2, q3dag * q3

    # Intended gate (always between q1 and q2)
    intended_term_qubits = q1dag * q2 + q1 * q2dag
    ideal_gate_qubits = (-1.0j * (np.pi / 2) * intended_term_qubits).expm()

    drag_ops_qubits = {
        # q1<->q3 exchange: shares q1 -> induces q2<->q3
        "qubit-qubit": (n1 - n3, q2dag * q3 + q2 * q3dag, ideal_gate_qubits),
        # direct subharmonic drive on q3 (spectator qubit): Stark shift only
        "qubit-sub": (n3, None, ideal_gate_qubits),
        # direct subharmonic drive on q1 (target qubit): Stark shift on q1
        "qubit-sub (inter)": (n1, None, ideal_gate_qubits),
    }


    # === Hilbert Space 2: Two Qubits + SNAIL System === #
    n_dim_snail = 8  # 8-Level SNAIL Mode
    qs = destroy(2)  # Qubit part of SNAIL system
    s = destroy(n_dim_snail)  # SNAIL oscillator
    qs1 = tensor(qs, qeye(2), qeye(n_dim_snail))  # First qubit
    qs2 = tensor(qeye(2), qs, qeye(n_dim_snail))  # Second qubit
    s1 = tensor(qeye(2), qeye(2), s)  # SNAIL mode
    qs1dag = qs1.dag()
    qs2dag = qs2.dag()
    s1dag = s1.dag()
    nqs1, ns1 = qs1dag * qs1, s1dag * s1

    # Intended gate (always between qs1 and qs2)
    intended_term_snail = qs1dag * qs2 + qs1 * qs2dag
    ideal_gate_snail = (-1.0j * (np.pi / 2) * intended_term_snail).expm()

    drag_ops_snail = {
        "snail-qubit": (nqs1 - ns1, qs2dag * s1 + qs2 * s1dag, ideal_gate_snail),
        "snail-qubit (inter)": (nqs1 - ns1, qs2dag * s1 + qs2 * s1dag, ideal_gate_snail),
    }

    # Set which induced exchanges are resonant for your frequency allocation.
    # Default False: assume the induced b<->c transition is off-resonant and
    # rotates away. Flip to True for any frequency collision you want to test.
    induced_resonant_flags = {
        "qubit-qubit": False,
        "snail-qubit": False,
        "snail-qubit (inter)": False,
    }

    # Diagonal residual coefficient: exchange rows -> 1 (n_a - n_b);
    # direct-drive 'sub' rows -> 2 ([q,q^dag] = 1 - 2n gives 2 g_s^2/Delta on n).
    stark_factors = {
        "qubit-qubit": 1.0,
        "qubit-sub": 2.0,
        "qubit-sub (inter)": 2.0,
        "snail-qubit": 1.0,
        "snail-qubit (inter)": 1.0,
    }
 

    # Compute infidelity curves and fit (a, b, c)
    infidelity_params = {}
    fidelity_results = {}

    # Compute for qubit-based spectators
    for key, (diag_op, induced_op, gate) in drag_ops_qubits.items():
        fidelity_results[key] = simulate_drag_infidelity(
            detuning_list, intended_term_qubits, gate, prefactors[key],
            diag_op, g_tgt, induced_op, induced_resonant_flags.get(key, False),
            stark_factors[key],
        )
        #infidelity_params[key] = fit_infidelity(detuning_list, fidelity_results[key])

    # Compute for SNAIL-based spectators
    for key, (diag_op, induced_op, gate) in drag_ops_snail.items():
        fidelity_results[key] = simulate_drag_infidelity(
            detuning_list, intended_term_snail, gate, prefactors[key],
            diag_op, g_tgt, induced_op, induced_resonant_flags.get(key, False),
            stark_factors[key],
        )
        #infidelity_params[key] = fit_infidelity(detuning_list, fidelity_results[key])

    return infidelity_params, fidelity_results