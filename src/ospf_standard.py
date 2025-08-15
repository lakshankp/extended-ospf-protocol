import time
import heapq
from typing import Dict, List, Tuple
from dataclasses import dataclass
import networkx as nx

@dataclass
class LSA:
    router_id: int
    sequence_number: int
    neighbors: Dict[int, float]
    timestamp: float

@dataclass
class RoutingEntry:
    destination: int
    next_hop: int
    cost: float
    path: List[int]

class StandardOSPF:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.lsdb = {}
        self.routing_tables = {}
    def flood_lsa(self, source: int) -> float:
        start_time = time.time()
        for node in self.graph.nodes():
            neighbors = {}
            for neighbor in self.graph.neighbors(node):
                neighbors[neighbor] = self.graph[node][neighbor]['weight']
            self.lsdb[node] = LSA(router_id=node, sequence_number=1, neighbors=neighbors, timestamp=time.time())
        try:
            diameter = nx.diameter(self.graph)
        except:
            diameter = 10
        flooding_delay = diameter * 0.001
        return time.time() - start_time + flooding_delay
    def compute_routing_table(self, source: int) -> Tuple[Dict[int, RoutingEntry], float]:
        start_time = time.time()
        distances = {node: float('inf') for node in self.graph.nodes()}
        distances[source] = 0
        previous = {node: None for node in self.graph.nodes()}
        pq = [(0, source)]
        visited = set()
        while pq:
            current_dist, current = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.graph.neighbors(current):
                weight = self.graph[current][neighbor]['weight']
                distance = current_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        routing_table = {}
        for dest in self.graph.nodes():
            if dest != source and previous[dest] is not None:
                path = self._reconstruct_path(dest, previous)
                if len(path) > 1:
                    routing_table[dest] = RoutingEntry(destination=dest, next_hop=path[1], cost=distances[dest], path=path)
        computation_time = time.time() - start_time
        return routing_table, computation_time
    def _reconstruct_path(self, dest: int, previous: Dict[int, int]) -> List[int]:
        path = []
        current = dest
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        return path
    def converge(self) -> float:
        start_time = time.time()
        flooding_time = self.flood_lsa(0)
        for node in self.graph.nodes():
            routing_table, _ = self.compute_routing_table(node)
            self.routing_tables[node] = routing_table
        total_time = time.time() - start_time + flooding_time
        return total_time
