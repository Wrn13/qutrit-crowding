import matplotlib.pyplot as plt
import networkx as nx


class QuantumModuleGraph:
    def __init__(self, num_qubits, edges=None):
        self.G = nx.Graph()
        self.num_qubits = num_qubits  # Default for topology setup
        if edges is None:
            self._add_edges()
        else:
            for u, v, interaction, color in edges:
                self.G.add_edge(u, v, interaction=interaction, color=color)

    def _add_edges(self):
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                self.G.add_edge(
                    f"Q{i}", f"Q{j}", interaction="qubit-qubit", color="blue"
                )
            self.G.add_edge(f"Q{i}", "SNAIL", interaction="snail-qubit", color="orange")

    def get_interaction_frequencies(self, qubit_frequencies, snail_frequency):
        interaction_freqs = {
            "qubit-qubit": {},
            "snail-qubit": {},
            "qubit-resonance": {},
            "snail-resonance": {},
            "qubit-sub": {},
            "snail-sub": {},
        }
        for i, freq in enumerate(qubit_frequencies):
            interaction_freqs["qubit-resonance"][f"Q{i}"] = freq
            interaction_freqs["qubit-sub"][f"Q{i}"] = freq / 2
        interaction_freqs["snail-resonance"]["SNAIL"] = snail_frequency
        interaction_freqs["snail-sub"]["SNAIL"] = snail_frequency / 2
        for u, v in self.G.edges:
            if u.startswith("Q") and v.startswith("Q"):
                interaction_freqs["qubit-qubit"][(u, v)] = abs(
                    qubit_frequencies[int(u[1:])] - qubit_frequencies[int(v[1:])]
                )
            elif v == "SNAIL":
                interaction_freqs["snail-qubit"][(u, v)] = abs(
                    qubit_frequencies[int(u[1:])] - snail_frequency
                )
        return interaction_freqs

    def plot_graph(self, qubit_frequencies, snail_frequency):
        pos = nx.spring_layout(self.G, seed=42)
        labels = {
            node: (
                f"{node}\n{qubit_frequencies[int(node[1:])]:.2f} GHz"
                if node.startswith("Q")
                else f"SNAIL\n{snail_frequency:.2f} GHz"
            )
            for node in self.G.nodes
        }
        node_colors = [
            "green" if node.startswith("Q") else "red" for node in self.G.nodes
        ]
        plt.figure(figsize=(2, 2))
        nx.draw(
            self.G,
            pos,
            with_labels=True,
            node_color=node_colors,
            edgecolors="black",
            width=2,
        )
        nx.draw_networkx_edges(
            self.G,
            pos,
            edge_color=[self.G.edges[e]["color"] for e in self.G.edges],
            width=2,
        )
        plt.show()

    def plot_interaction_frequencies(self, qubit_frequencies, snail_frequency):
        all_freqs = list(qubit_frequencies) + [snail_frequency]
        interaction_freqs = self.get_interaction_frequencies(
            qubit_frequencies, snail_frequency
        )
        with plt.style.context(["ieee", "use_mathtext", "science"]):
            fig, ax = plt.subplots(figsize=(3.5, 1))
            max_freq = max(all_freqs) * 1.05
            ax.set_xlim(0, max_freq)
            ax.get_yaxis().set_visible(False)
            added_labels = set()
            color_map = {
                "qubit-qubit": "blue",
                "qubit-resonance": "green",
                "snail-qubit": "orange",
                "qubit-sub": "gray",
                "snail-sub": "magenta",
                "snail-resonance": "red",
            }
            legend_labels = {
                "qubit-qubit": "Two-Qubit Gates",
                "qubit-resonance": "Qubit Modes",
                "snail-qubit": "SNAIL Qubit Difference",
                "qubit-sub": "Qubit Subharmonic",
                "snail-sub": "SNAIL Subharmonic",
                "snail-resonance": "SNAIL Mode",
            }
            for interaction_type, freqs in interaction_freqs.items():
                if not freqs:
                    continue
                color = color_map.get(interaction_type, "black")
                if interaction_type in {"snail-resonance", "qubit-resonance"}:
                    linestyle = "-"
                else:
                    linestyle = (0, (2.1, 1.4))  # fine dashed line
                label = (
                    legend_labels[interaction_type]
                    if interaction_type not in added_labels
                    else ""
                )
                for freq in freqs.values():
                    ax.axvline(
                        freq,
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.5,
                        alpha=0.8,
                        label=label,
                    )
                    added_labels.add(interaction_type)
            ax.set_xlabel("Frequency (GHz)")

            # Ordered legend (manually controlled order)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            legend_order = [
                "Qubit Modes",
                "SNAIL Mode",
                "Two-Qubit Gates",
                "Qubit Subharmonic",
                "SNAIL Qubit Difference",
                "SNAIL Subharmonic",
            ]
            ordered_handles = [
                by_label[label] for label in legend_order if label in by_label
            ]
            ordered_labels = [label for label in legend_order if label in by_label]

            ax.legend(
                ordered_handles,
                ordered_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.32),
                ncol=2,
                fontsize=8,
                columnspacing=0.8,
                handlelength=1.2,
            )

        plt.savefig("corral_crowding_4q_optimal_frequencies.pdf", bbox_inches="tight")
        plt.show()

    def get_graph(self):
        """Returns the NetworkX graph object."""
        return self.G
