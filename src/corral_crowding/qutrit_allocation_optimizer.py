# import networkx
import lovelyplots
import networkx as nx
import numpy as np
import rustworkx as rx
import scienceplots
from rustworkx.visualization import graphviz_draw, mpl_draw
from scipy.optimize import minimize
from scipy.stats import gmean
from tqdm import tqdm

from corral_crowding.detuning_fit import compute_infidelity_parameters, decay_fit
from corral_crowding.module_graph import QuantumModuleGraph
from corral_crowding.speedlimit_fit import (
    lifetime_decay_fit,
    speedlimit_infidelity_params,
)


class GateFidelityOptimizer:
    def __init__(
        self,
        module,
        lambdaq,
        eta,
        g3,
        alpha=0.12,  # 120Mhz
        min_bare_space_ghz=0.2,  # 200mhz
        T_1=120e-6,
        qubit_bounds=(3.3, 5.7),
        snail_bounds=(4.2, 4.7),
        drop_k=0,  # 0 for best, 1 to drop worst, 2 to drop 2 worst, etc
        use_lifetime=False,
    ):
        self.lambdaq = lambdaq
        self.eta = eta
        self.g3 = g3
        self.alpha = alpha
        self.min_bare_space_ghz = min_bare_space_ghz
        self.qubit_bounds = qubit_bounds
        self.snail_bounds = snail_bounds
        self.module_graph = module
        self.best_frequencies = None
        self.best_cost = np.inf
        self.drop_k = drop_k

        detuning_list = np.linspace(50, 1000, 64)
        self.infidelity_params, _ = compute_infidelity_parameters(
            detuning_list, lambdaq=lambdaq, eta=eta, alpha=120e6, g3=g3
        )
        ###
        self.use_lifetime = use_lifetime
        avg_snail = (snail_bounds[0] + snail_bounds[1]) / 2
        self.speedlimit_params, _ = speedlimit_infidelity_params(
            f_SNAIL=avg_snail, t_f_calib=250e-9, T1=T_1, g3=g3, lambdaq=lambdaq
        )

    def _unit_crosstalk(self, intended_freq, spectator_key, spectator_freq):
        distance = np.abs(intended_freq - spectator_freq)
        units_distance = distance * 1e3  # Convert GHz → MHz

        # Strong crosstalk penalty
        if distance < 0.05:
            return 0.5

        # anharmonicity higher level transitions penalty
        if spectator_key == "qubit-qubit" and distance < self.alpha:
            return 0.2

        # if detuning is > 800MHz, then infidelity is 0
        # this acts to normalize against configs with more terms
        # otherwise they get artificially higher infidelity
        if distance > 0.8:
            return 0

        params = self.infidelity_params.get(spectator_key)
        if params is None:
            raise KeyError(f"Unknown interaction type: {spectator_key}")

        return decay_fit(units_distance, *params)

    def _unit_decay(self, intended_freq, snail_sub):
        if self.use_lifetime:
            distance = np.abs(intended_freq - snail_sub)
            units_distance = distance * 1e3  # Convert GHz → MHz
            return lifetime_decay_fit(units_distance, *self.speedlimit_params)
        return 0

    def _compute_gate_infidelity(self, edge, interaction_data):
        driven_freq = interaction_data["qubit-qubit"][edge]
        gate_infidelity = sum(
            self._unit_crosstalk(driven_freq, interaction_type, spectator_freq)
            for interaction_type in ["qubit-qubit", "snail-qubit", "qubit-sub"]
            for spectator_edge, spectator_freq in interaction_data[
                interaction_type
            ].items()
            if spectator_edge != edge
        )
        # to combine coherent and incoherent fidelities, multiply (ESP)
        # however our variables are infidelities, so take 1-term
        # multiply (1-infidelity)(1-lifetime loss)
        spectator_freq = interaction_data["snail-sub"]["SNAIL"]
        gate_infidelity_with_lifetime = 1 - (1 - gate_infidelity) * (
            1 - self._unit_decay(driven_freq, spectator_freq)
        )
        return gate_infidelity, gate_infidelity_with_lifetime

    def _compute_bare_infidelity(self, edge, interaction_data):
        # for each qubit-resonance, add penality of 1 if closer than min_bare_space_ghz to
        # any of the other bare resonances (qubit-resonance or snail-resonance)
        driven_freq = interaction_data["qubit-resonance"][edge]
        cost = lambda freq: 1.0 - freq / self.min_bare_space_ghz
        gate_infidelity = sum(
            (
                cost(np.abs(driven_freq - spec_freq))
                if np.abs(driven_freq - spec_freq) < self.min_bare_space_ghz
                else 0
            )
            for spectator_edge, spec_freq in interaction_data["qubit-resonance"].items()
            if spectator_edge != edge
        )
        gate_infidelity += sum(
            (
                cost(np.abs(driven_freq - spec_freq))
                if np.abs(driven_freq - spec_freq) < self.min_bare_space_ghz
                else 0
            )
            for spec_freq in interaction_data["snail-resonance"].values()
        )
        return gate_infidelity

    def compute_total_infidelity(self, frequencies):
        qubit_frequencies, snail_frequency = frequencies[:-1], frequencies[-1]
        interaction_data = self.module_graph.get_interaction_frequencies(
            qubit_frequencies, snail_frequency
        )
        two_qubit_crowding = [
            self._compute_gate_infidelity(edge, interaction_data)[1]
            for edge in interaction_data["qubit-qubit"]
        ]
        # two_qubit_crowding = sum(two_qubit_crowding[self.drop_k :])
        two_qubit_crowding = sum(
            sorted(two_qubit_crowding, reverse=True)[self.drop_k :]
        )
        one_qubit_crowding = sum(
            self._compute_bare_infidelity(edge, interaction_data)
            for edge in interaction_data["qubit-resonance"]
        )
        return two_qubit_crowding + one_qubit_crowding

    def optimize_frequencies(self, attempts=128):
        qubit_count = self.module_graph.num_qubits
        self.best_cost = np.inf
        for _ in tqdm(range(attempts)):
            initial_guess = np.append(
                np.random.uniform(
                    self.qubit_bounds[0], self.qubit_bounds[1], qubit_count
                ),
                np.random.uniform(self.snail_bounds[0], self.snail_bounds[1]),
            )
            result = minimize(
                self.compute_total_infidelity,
                initial_guess,
                bounds=[self.qubit_bounds] * qubit_count + [self.snail_bounds],
                method="Nelder-Mead",
            )
            temp_result = np.mean(self.get_final_infidelities(result.x))
            if temp_result < self.best_cost:
                self.best_cost = temp_result
                self.best_frequencies = result.x
                best_result = result
        print(best_result.message)
        return self.best_frequencies, self.best_cost

    def get_final_infidelities(self, freqs=None):
        if freqs is None:
            if self.best_frequencies is None:
                print("No optimized frequencies available.")
                return
            else:
                temp_freqs = self.best_frequencies
        else:
            temp_freqs = freqs

        qubit_frequencies, snail_frequency = (
            temp_freqs[:-1],
            temp_freqs[-1],
        )
        interaction_data = self.module_graph.get_interaction_frequencies(
            qubit_frequencies, snail_frequency
        )
        gate_infidelities = {
            edge: self._compute_gate_infidelity(edge, interaction_data)[1]
            for edge in list(interaction_data["qubit-qubit"])
        }

        return sorted(gate_infidelities.values(), reverse=True)[self.drop_k :]

        # avg_gate_infidelity = gmean(list(gate_infidelities.values()))
        # return avg_gate_infidelity

    def report_results(self):
        if self.best_frequencies is None:
            print("No optimized frequencies available.")
            return

        qubit_frequencies, snail_frequency = (
            self.best_frequencies[:-1],
            self.best_frequencies[-1],
        )
        interaction_data = self.module_graph.get_interaction_frequencies(
            qubit_frequencies, snail_frequency
        )

        if not self.use_lifetime:
            print("Lifetime loss not considered.")

        print("Qubit Frequencies:", qubit_frequencies, "GHz")
        print(f"SNAIL Frequency: {snail_frequency} GHz")
        print("Gate Infidelities:")

        gate_infidelities = {
            edge: self._compute_gate_infidelity(edge, interaction_data)
            for edge in list(interaction_data["qubit-qubit"])
        }

        for edge, (
            infidelity_no_lifetime,
            infidelity_with_lifetime,
        ) in gate_infidelities.items():
            freq = interaction_data["qubit-qubit"][edge]
            print(
                f"  Gate {edge}: {freq:.6f} GHz → "
                f"fidelity (no lifetime loss): {1 - infidelity_no_lifetime:.6e}, "
                f"fidelity (with lifetime loss): {1 - infidelity_with_lifetime:.6e}"
            )

        gate_infidelities = sorted(list(gate_infidelities.values()), reverse=True)[
            self.drop_k :
        ]
        avg_infidelity_no_lifetime = gmean([inf[0] for inf in gate_infidelities])
        avg_infidelity_with_lifetime = gmean([inf[1] for inf in gate_infidelities])

        print(
            f"\nAverage Gate fidelity (no lifetime loss): {1 - avg_infidelity_no_lifetime}"
        )
        print(
            f"Average Gate fidelity (with lifetime loss): {1 - avg_infidelity_with_lifetime}"
        )

        self.module_graph.plot_graph(qubit_frequencies, snail_frequency)
        self.module_graph.plot_interaction_frequencies(
            qubit_frequencies, snail_frequency
        )
