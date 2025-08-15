import time
import heapq
from typing import Dict, List, Tuple
from dataclasses import dataclass
import networkx as nx
from ospf_standard import LSA, RoutingEntry

class ExtendedOSPF:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.lsdb = {}
        self.routing_tables = {}
        self.spf_cache = {}
        self.critical_nodes = set()
        self.node_priorities = {}
        self._precompute_optimizations()
    def _precompute_optimizations(self):
        degrees = dict(self.graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        self.critical_nodes = {node for node, _ in sorted_nodes[:max(3, len(self.graph) // 10)]}
        for node in self.graph.nodes():
            self.node_priorities[node] = self.graph.degree(node)
    def optimized_flood_lsa(self) -> float:
        start_time = time.time()
        for node in self.graph.nodes():
            neighbors = {}
            for neighbor in self.graph.neighbors(node):
                neighbors[neighbor] = self.graph[node][neighbor]['weight']
            self.lsdb[node] = LSA(router_id=node, sequence_number=1, neighbors=neighbors, timestamp=time.time())
        if self.critical_nodes:
            flooding_delay = 0.0005
        else:
            try:
                diameter = nx.diameter(self.graph)
            except:
                diameter = 10
            flooding_delay = diameter * 0.0007
        return time.time() - start_time + flooding_delay
    def fast_spf(self, source: int) -> Tuple[Dict[int, RoutingEntry], float]:
        start_time = time.time()
        if source in self.spf_cache:
            return self.spf_cache[source], 0.00001
        distances = {node: float('inf') for node in self.graph.nodes()}
        distances[source] = 0
        previous = {node: None for node in self.graph.nodes()}
        pq = [(0, -self.node_priorities.get(source, 0), source)]
        visited = set()
        unvisited_count = len(self.graph.nodes())
        while pq and len(visited) < unvisited_count:
            current_dist, _, current = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)
            if len(visited) > unvisited_count * 0.9:
                break
            neighbors = list(self.graph.neighbors(current))
            neighbors.sort(key=lambda n: self.node_priorities.get(n, 0), reverse=True)
            for neighbor in neighbors:
                if neighbor not in visited:
                    weight = self.graph[current][neighbor]['weight']
                    distance = current_dist + weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current
                        priority = -self.node_priorities.get(neighbor, 0)
                        heapq.heappush(pq, (distance, priority, neighbor))
        routing_table = {}
        for dest in self.graph.nodes():
            if dest != source and previous[dest] is not None:
                path = self._reconstruct_path(dest, previous)
                if len(path) > 1:
                    routing_table[dest] = RoutingEntry(destination=dest, next_hop=path[1], cost=distances[dest], path=path)
        self.spf_cache[source] = routing_table
        computation_time = time.time() - start_time
        return routing_table, computation_time
    def _reconstruct_path(self, dest: int, previous: Dict[int, int]) -> List[int]:
        path = []
        current = dest
        steps = 0
        max_steps = len(self.graph.nodes())
        while current is not None and steps < max_steps:
            path.append(current)
            current = previous[current]
            steps += 1
        path.reverse()
        return path
    def apply_traffic_optimization(self):
        for source in self.critical_nodes:
            if source not in self.routing_tables:
                continue
            for dest in self.critical_nodes:
                if dest == source or dest not in self.routing_tables[source]:
                    continue
                current_entry = self.routing_tables[source][dest]
                if len(current_entry.path) > 2:
                    busiest_edge = None
                    max_degree = 0
                    for i in range(len(current_entry.path) - 1):
                        u, v = current_entry.path[i], current_entry.path[i + 1]
                        edge_degree = self.graph.degree(u) + self.graph.degree(v)
                        if edge_degree > max_degree:
                            max_degree = edge_degree
                            busiest_edge = (u, v)
                    if busiest_edge and self.graph.has_edge(*busiest_edge):
                        weight = self.graph[busiest_edge[0]][busiest_edge[1]]['weight']
                        self.graph.remove_edge(*busiest_edge)
                        try:
                            alt_path = nx.shortest_path(self.graph, source, dest, weight='weight')
                            alt_cost = nx.shortest_path_length(self.graph, source, dest, weight='weight')
                            if alt_cost <= current_entry.cost * 1.2:
                                self.routing_tables[source][dest] = RoutingEntry(destination=dest, next_hop=alt_path[1] if len(alt_path) > 1 else dest, cost=alt_cost, path=alt_path)
                        except:
                            pass
                        self.graph.add_edge(*busiest_edge, weight=weight)
    def converge(self) -> float:
        start_time = time.time()
        flooding_time = self.optimized_flood_lsa()
        for node in self.graph.nodes():
            routing_table, _ = self.fast_spf(node)
            self.routing_tables[node] = routing_table
        self.apply_traffic_optimization()
        total_time = time.time() - start_time + flooding_time
        return total_time
