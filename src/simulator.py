import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from topology import NetworkTopologyGenerator
from ospf_standard import StandardOSPF
from ospf_extended import ExtendedOSPF
from metrics import MetricsCalculator

class OSPFSimulator:
    def __init__(self):
        self.topology_gen = NetworkTopologyGenerator()
        self.networks = []
        self._create_networks()

    def __init__(self):
        self.topology_gen = NetworkTopologyGenerator()
        self.networks = []
        self._create_networks()
    def _create_networks(self):
        self.networks = [
            ("Ring", self.topology_gen.generate_ring(20), 20),
            ("Star", self.topology_gen.generate_star(25), 25),
            ("Grid 5x5", self.topology_gen.generate_grid(5, 5), 25),
            ("Random Regular", self.topology_gen.generate_random_regular(30, 4), 30),
            ("Scale-Free", self.topology_gen.generate_scale_free(35), 35),
            ("Small World", self.topology_gen.generate_small_world(40), 40),
            ("Hierarchical", self.topology_gen.generate_hierarchical(45), 45),
            ("Core-Distribution", self.topology_gen.generate_core_distribution(48), 48),
            ("Grid 7x7", self.topology_gen.generate_grid(7, 7), 49),
            ("Large Scale-Free", self.topology_gen.generate_scale_free(50, 4), 50)
        ]
        for _, graph, _ in self.networks:
            self.topology_gen.add_weights(graph)
    def simulate_single_network(self, name: str, graph: nx.Graph) -> dict:
        print(f"\nSimulating {name} (Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()})")
        std_ospf = StandardOSPF(graph)
        std_conv_time = std_ospf.converge()
        std_delay = MetricsCalculator.calculate_end_to_end_delay(graph, std_ospf.routing_tables)
        std_stretch = MetricsCalculator.calculate_path_stretch(graph, std_ospf.routing_tables)
        ext_ospf = ExtendedOSPF(graph)
        ext_conv_time = ext_ospf.converge()
        ext_delay = MetricsCalculator.calculate_end_to_end_delay(graph, ext_ospf.routing_tables)
        ext_stretch = MetricsCalculator.calculate_path_stretch(graph, ext_ospf.routing_tables)
        if graph.number_of_edges() > 0:
            edge_betweenness = nx.edge_betweenness_centrality(graph, weight='weight')
            critical_edge = max(edge_betweenness.items(), key=lambda x: x[1])[0]
            graph_copy = graph.copy()
            graph_copy.remove_edge(*critical_edge)
            if nx.is_connected(graph_copy):
                std_ospf_fail = StandardOSPF(graph_copy)
                std_reconv = std_ospf_fail.converge()
                ext_ospf_fail = ExtendedOSPF(graph_copy)
                ext_reconv = ext_ospf_fail.converge()
            else:
                std_reconv = std_conv_time * 2
                ext_reconv = ext_conv_time * 2
        else:
            std_reconv = ext_reconv = 0
        results = {
            'network': name,
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'std_conv_time': std_conv_time * 1000,
            'ext_conv_time': ext_conv_time * 1000,
            'std_delay': std_delay,
            'ext_delay': ext_delay,
            'std_stretch': std_stretch,
            'ext_stretch': ext_stretch,
            'std_reconv': std_reconv * 1000,
            'ext_reconv': ext_reconv * 1000,
            'conv_improvement': ((std_conv_time - ext_conv_time) / std_conv_time * 100),
            'delay_improvement': ((std_delay - ext_delay) / std_delay * 100) if std_delay > 0 else 0
        }
        print(f"  Standard OSPF: Conv={results['std_conv_time']:.3f}ms, Delay={results['std_delay']:.2f}")
        print(f"  Extended OSPF: Conv={results['ext_conv_time']:.3f}ms, Delay={results['ext_delay']:.2f}")
        print(f"  Improvements: Conv={results['conv_improvement']:.1f}%, Delay={results['delay_improvement']:.1f}%")
        return results
    def run_full_simulation(self):
        print("=" * 80)
        print("OSPF vs Extended OSPF Performance Comparison")
        print("=" * 80)
        results = []
        for name, graph, _ in self.networks:
            result = self.simulate_single_network(name, graph)
            results.append(result)
        self._print_summary(results)
        # Removed call to self._plot_results(results) since it no longer exists
        return results
    def _print_summary(self, results):
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        avg_conv_imp = np.mean([r['conv_improvement'] for r in results])
        avg_delay_imp = np.mean([r['delay_improvement'] for r in results])
        avg_std_conv = np.mean([r['std_conv_time'] for r in results])
        avg_ext_conv = np.mean([r['ext_conv_time'] for r in results])
        avg_std_delay = np.mean([r['std_delay'] for r in results])
        avg_ext_delay = np.mean([r['ext_delay'] for r in results])
        print(f"\nConvergence Time:")
        print(f"  Standard OSPF Average: {avg_std_conv:.3f} ms")
        print(f"  Extended OSPF Average: {avg_ext_conv:.3f} ms")
        print(f"  Average Improvement: {avg_conv_imp:.1f}%")
        print(f"\nEnd-to-End Delay:")
        print(f"  Standard OSPF Average: {avg_std_delay:.2f}")
        print(f"  Extended OSPF Average: {avg_ext_delay:.2f}")
        print(f"  Average Improvement: {avg_delay_imp:.1f}%")
        print(f"\nReconvergence Time (after link failure):")
        print(f"  Standard OSPF Average: {np.mean([r['std_reconv'] for r in results]):.3f} ms")
        print(f"  Extended OSPF Average: {np.mean([r['ext_reconv'] for r in results]):.3f} ms")
    def plot_convergence_time(self, results):
        networks = [r['network'] for r in results]
        x = np.arange(len(networks))
        width = 0.35
        std_conv = [r['std_conv_time'] for r in results]
        ext_conv = [r['ext_conv_time'] for r in results]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, std_conv, width, label='Standard OSPF', color='#1f77b4', alpha=0.8)
        bars2 = ax.bar(x + width/2, ext_conv, width, label='Extended OSPF', color='#ff7f0e', alpha=0.8)
        ax.set_ylabel('Convergence Time (ms)', fontsize=12)
        ax.set_title('OSPF Convergence Time Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(networks, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.show()

    def plot_end_to_end_delay(self, results):
        networks = [r['network'] for r in results]
        x = np.arange(len(networks))
        width = 0.35
        std_delay = [r['std_delay'] for r in results]
        ext_delay = [r['ext_delay'] for r in results]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, std_delay, width, label='Standard OSPF', color='#1f77b4', alpha=0.8)
        bars2 = ax.bar(x + width/2, ext_delay, width, label='Extended OSPF', color='#ff7f0e', alpha=0.8)
        ax.set_ylabel('Average End-to-End Delay', fontsize=12)
        ax.set_title('OSPF End-to-End Delay Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(networks, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.show()

    def plot_improvements(self, results):
        networks = [r['network'] for r in results]
        x = np.arange(len(networks))
        width = 0.35
        conv_imp = [r['conv_improvement'] for r in results]
        delay_imp = [r['delay_improvement'] for r in results]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, conv_imp, width, label='Convergence', color='#2ca02c', alpha=0.8)
        ax.bar(x + width/2, delay_imp, width, label='Delay', color='#d62728', alpha=0.8)
        ax.set_ylabel('Improvement (%)', fontsize=12)
        ax.set_title('Extended OSPF Improvements over Standard OSPF', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(networks, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.show()

    def plot_performance_vs_size(self, results):
        sizes = [r['nodes'] for r in results]
        conv_imp = [r['conv_improvement'] for r in results]
        delay_imp = [r['delay_improvement'] for r in results]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(sizes, conv_imp, label='Convergence Improvement', s=100, alpha=0.7, color='#2ca02c')
        ax.scatter(sizes, delay_imp, label='Delay Improvement', s=100, alpha=0.7, color='#d62728')
        if len(sizes) > 1:
            z_conv = np.polyfit(sizes, conv_imp, 1)
            p_conv = np.poly1d(z_conv)
            z_delay = np.polyfit(sizes, delay_imp, 1)
            p_delay = np.poly1d(z_delay)
            ax.plot(sizes, p_conv(sizes), "--", color='#2ca02c', alpha=0.8)
            ax.plot(sizes, p_delay(sizes), "--", color='#d62728', alpha=0.8)
        ax.set_xlabel('Network Size (nodes)', fontsize=12)
        ax.set_ylabel('Improvement (%)', fontsize=12)
        ax.set_title('Performance Improvement vs Network Size', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def show_network_graph(self, graph, title="Network Topology"):
        """Display a graphical view of the given network graph."""
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(graph, seed=42)
        nx.draw_networkx_nodes(graph, pos, node_color='#1f78b4', node_size=300, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, width=1.5, alpha=0.5)
        nx.draw_networkx_labels(graph, pos, font_size=10, font_color='black')
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        if edge_labels:
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def show_all_networks_graph(self):
        """Display all generated networks in a single window with subplots."""
        num_networks = len(self.networks)
        if num_networks == 0:
            print("No networks to display.")
            return
        # Determine grid size (try to make it as square as possible)
        cols = int(np.ceil(np.sqrt(num_networks)))
        rows = int(np.ceil(num_networks / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = np.array(axes).reshape(-1)  # Flatten in case of 1D
        for idx, (name, graph, _) in enumerate(self.networks):
            ax = axes[idx]
            pos = nx.spring_layout(graph, seed=42)
            nx.draw_networkx_nodes(graph, pos, node_color='#1f78b4', node_size=200, alpha=0.8, ax=ax)
            nx.draw_networkx_edges(graph, pos, width=1.2, alpha=0.5, ax=ax)
            nx.draw_networkx_labels(graph, pos, font_size=8, font_color='black', ax=ax)
            edge_labels = nx.get_edge_attributes(graph, 'weight')
            if edge_labels:
                nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6, ax=ax)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.axis('off')
        # Hide any unused subplots
        for j in range(idx+1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    simulator = OSPFSimulator()
    results = simulator.run_full_simulation()
    print("\n" + "=" * 80)
    print("ADDITIONAL ANALYSIS")
    print("=" * 80)
    best_conv = max(results, key=lambda x: x['conv_improvement'])
    best_delay = max(results, key=lambda x: x['delay_improvement'])
    print(f"\nBest convergence improvement: {best_conv['network']} ({best_conv['conv_improvement']:.1f}%)")

    # Show all generated networks in a single window
    simulator.show_all_networks_graph()

    # Show performance graphs separately
    simulator.plot_convergence_time(results)
    simulator.plot_end_to_end_delay(results)
    simulator.plot_improvements(results)
    simulator.plot_performance_vs_size(results)
