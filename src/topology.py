import networkx as nx
import random
from typing import Tuple

class NetworkTopologyGenerator:
    """Module for generating various network topologies"""
    @staticmethod
    def generate_ring(n: int) -> nx.Graph:
        return nx.cycle_graph(n)
    @staticmethod
    def generate_star(n: int) -> nx.Graph:
        return nx.star_graph(n - 1)
    @staticmethod
    def generate_grid(rows: int, cols: int) -> nx.Graph:
        G = nx.grid_2d_graph(rows, cols)
        return nx.convert_node_labels_to_integers(G)
    @staticmethod
    def generate_random_regular(n: int, degree: int) -> nx.Graph:
        return nx.random_regular_graph(degree, n)
    @staticmethod
    def generate_scale_free(n: int, m: int = 3) -> nx.Graph:
        return nx.barabasi_albert_graph(n, m)
    @staticmethod
    def generate_small_world(n: int, k: int = 6, p: float = 0.3) -> nx.Graph:
        return nx.watts_strogatz_graph(n, k, p)
    @staticmethod
    def generate_hierarchical(n: int) -> nx.Graph:
        G = nx.Graph()
        levels = []
        remaining = n
        level = 0
        while remaining > 0:
            level_size = min(2 ** level, remaining)
            level_nodes = list(range(n - remaining, n - remaining + level_size))
            levels.append(level_nodes)
            G.add_nodes_from(level_nodes)
            remaining -= level_size
            level += 1
        for i in range(len(levels) - 1):
            for node in levels[i]:
                connections = min(3, len(levels[i + 1]))
                targets = random.sample(levels[i + 1], connections)
                for target in targets:
                    G.add_edge(node, target)
        for level_nodes in levels:
            if len(level_nodes) > 3:
                for _ in range(len(level_nodes) // 2):
                    u, v = random.sample(level_nodes, 2)
                    if not G.has_edge(u, v):
                        G.add_edge(u, v)
        return G
    @staticmethod
    def generate_core_distribution(n: int) -> nx.Graph:
        G = nx.Graph()
        core_size = max(3, n // 10)
        core_nodes = list(range(core_size))
        for i in core_nodes:
            for j in core_nodes:
                if i < j:
                    G.add_edge(i, j)
        remaining = n - core_size
        distribution_size = remaining // 3
        access_size = remaining - distribution_size
        node_id = core_size
        for _ in range(distribution_size):
            G.add_node(node_id)
            core_connections = random.sample(core_nodes, min(2, core_size))
            for core in core_connections:
                G.add_edge(node_id, core)
            node_id += 1
        distribution_nodes = list(range(core_size, node_id))
        for _ in range(access_size):
            if node_id < n:
                G.add_node(node_id)
                if distribution_nodes:
                    dist_connections = random.sample(distribution_nodes, min(2, len(distribution_nodes)))
                    for dist in dist_connections:
                        G.add_edge(node_id, dist)
                node_id += 1
        return G
    @staticmethod
    def add_weights(G: nx.Graph, weight_range: Tuple[int, int] = (1, 10)):
        for (u, v) in G.edges():
            G[u][v]['weight'] = random.randint(*weight_range)
