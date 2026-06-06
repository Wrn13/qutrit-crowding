import matplotlib.pyplot as plt
import networkx as nx

from corral_crowding.module_graph import QuantumModuleGraph


class QutritModuleGraph(QuantumModuleGraph):
    """Extension of 2 Qubit Module Graph for Qutrits in various topologies."""

    def __init__(self, num_qutrits, edge_type, edges: list[tuple[str, str, str, str]] = None):
        self.G = nx.Graph()
        self.num_qutrits = num_qutrits  # Default for topology setup
        if edge_type == "all-to-all":
            self._add_edges()
        else:
            for u, v, interaction, color in edges:
                self.G.add_edge(u, v, interaction=interaction, color=color)

    def _add_edges(self):
        for i in range(self.num_qutrits):
            for j in range(i + 1, self.num_qutrits):
                self.G.add_edge(
                    f"Q{i}", f"Q{j}", interaction="qutrit-qutrit", color="blue"
                )
            self.G.add_edge(f"Q{i}", "SNAIL", interaction="snail-qutrit", color="orange")

    def get_interaction_frequencies(self, qutrit_frequencies, alpha, snail_frequency):
        interaction_freqs = {
            "qutrit-qutrit": {},
            "snail-qutrit": {},
            "qutrit-ge": {},
            "qutrit-ef": {},
            #"qutrit-gf": {},
            "snail-resonance": {},
            "qutrit-sub": {},
            "snail-sub": {},
        }
        for i, freq in enumerate(qutrit_frequencies):
            interaction_freqs["qutrit-ge"][f"Q{i}"] = freq
            interaction_freqs["qutrit-ef"][f"Q{i}"] = freq + alpha[i]
            #interaction_freqs["qutrit-gf"][f"Q{i}"] = 2*freq + alpha[i]
            interaction_freqs["qutrit-sub"][f"Q{i}"] = freq / 2
        interaction_freqs["snail-resonance"]["SNAIL"] = snail_frequency
        interaction_freqs["snail-sub"]["SNAIL"] = snail_frequency / 2
        for u, v in self.G.edges:
            if u.startswith("Q") and v.startswith("Q"):
                interaction_freqs["qutrit-qutrit"][(u, v)] = abs(
                    qutrit_frequencies[int(u[1:])] - qutrit_frequencies[int(v[1:])]
                )
            elif v == "SNAIL":
                interaction_freqs["snail-qutrit"][(u, v)] = abs(
                    qutrit_frequencies[int(u[1:])] - snail_frequency
                )
        return interaction_freqs

    def plot_graph(self, qutrit_frequencies, snail_frequency):
        pos = nx.spring_layout(self.G, seed=42)
        labels = {
            node: (
                f"{node}\n{qutrit_frequencies[int(node[1:])]:.2f} GHz"
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

    def plot_graph(self, qutrit_frequencies, snail_frequency, fidelities:dict[tuple[str,str],float]):
        newG = self.G.copy()
        self.G = newG
        self.G.remove_node("SNAIL")

        pos = nx.spring_layout(self.G, seed=42)
        
        weights = fidelities.values()
        plt.figure(figsize=(2, 2))
        nx.draw(
            self.G,
            pos,
            node_size = 500,
            node_color='white',
            edgecolors='#333333',
            linewidth=1.5,
        )
        nx.draw_networkx_edges(
            self.G,
            pos,
            edge_color=[self.G.edges[e]["color"] for e in self.G.edges],
            width=2,
        )
        nx.draw_networkx_labels(self.G,pos,fontsize=7, font_color='#333333')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels = fidelities, font_size=7)
        plt.show()


# for ax, name in zip(axes, topo_names):
#     G, pos = subgraphs[name]
#     edge_items  = list(G.edges(data=True))
#     weights     = [d['weight'] for _, _, d in edge_items]
#     norm        = lambda w: (w - vmin) / (vmax - vmin + 1e-9)
#     edge_colors = [cmap(norm(w)) for w in weights]
#     edge_widths = [1.5 + 4 * norm(w) for w in weights]

#     nx.draw_networkx_nodes(G, pos, node_size=500, node_color='white',
#                            edgecolors='#333333', linewidths=1.5, ax=ax)
#     nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, _ in edge_items],
#                            edge_color=edge_colors, width=edge_widths, ax=ax)
#     nx.draw_networkx_labels(G, pos, font_size=7, font_color='#333333', ax=ax)

#     edge_labels = {(u, v): f"{d['weight']:.3f}" for u, v, d in edge_items}
#     diag_labels  = {(u, v): lbl for (u, v), lbl in edge_labels.items()
#                     if pos[u][1] != pos[v][1] and pos[u][0] != pos[v][0]}
#     axial_labels = {(u, v): lbl for (u, v), lbl in edge_labels.items()
#                     if (u, v) not in diag_labels}
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=axial_labels, font_size=7, ax=ax)
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=diag_labels,
#                                  font_size=7, label_pos=0.3, ax=ax)


    def plot_interaction_frequencies(self, qutrit_frequencies, qutrit_anharmonicities, snail_frequency):
        all_freqs = list(qutrit_frequencies) + [snail_frequency]
        interaction_freqs = self.get_interaction_frequencies(
            qutrit_frequencies, qutrit_anharmonicities, snail_frequency
        )
        with plt.style.context(["ieee", "use_mathtext", "science"]):
            fig, ax = plt.subplots(figsize=(3.5, 1))
            max_freq = max(all_freqs) * 1.05
            ax.set_xlim(0, max_freq)
            ax.get_yaxis().set_visible(False)
            added_labels = set()
            cmap = plt.get_cmap('jet', 7)
            color_map = {
                "qutrit-qutrit": cmap(0),
                "qutrit-ge": cmap(1),
                "qutrit-ef": cmap(2),
                "snail-qutrit": cmap(3),
                "qutrit-sub": cmap(4),
                "snail-sub": cmap(5),
                "snail-resonance": cmap(6),
            }
            legend_labels = {
                "qutrit-qutrit": "Two-qutrit Gates",
                "qutrit-ge": "Qutrit g-e Transition",
                "qutrit-ef": "Qutrit e-f Transition",
                "snail-qutrit": "SNAIL Qutrit Difference",
                "qutrit-sub": "Qutrit Subharmonic",
                "snail-sub": "SNAIL Subharmonic",
                "snail-resonance": "SNAIL Mode",
            }

            linestyles = {
                "qutrit-qutrit": "-",
                "qutrit-ge": "-",
                "qutrit-ef": ":",
                "snail-qutrit": "--",
                "qutrit-sub": ":",
                "snail-sub": ":",
                "snail-resonance": "-",
            }
            for interaction_type, freqs in interaction_freqs.items():
                if not freqs:
                    continue
                color = color_map.get(interaction_type, "black")
                # if interaction_type in {"snail-resonance", "qutrit-resonance"}:
                linestyle = linestyles[interaction_type]
                # else:
                #     linestyle = (0, (2.1, 1.4))  # fine dashed line
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
                "Qutrit g-e Transition",
                "Qutrit e-f Transition",
                "SNAIL Mode",
                "Two-qutrit Gates",
                "Qutrit Subharmonic",
                "SNAIL Qutrit Difference",
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

        plt.savefig("qutrit_crowding_3q_all_to_all_optimal_freq.pdf", bbox_inches="tight")
        plt.show()

    def get_graph(self):
        """Returns the NetworkX graph object."""
        return self.G
