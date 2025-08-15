import networkx as nx
from typing import Dict
from ospf_standard import RoutingEntry

class MetricsCalculator:
    @staticmethod
    def calculate_end_to_end_delay(graph: nx.Graph, routing_tables: Dict[int, Dict[int, RoutingEntry]]) -> float:
        total_delay = 0
        path_count = 0
        for source in routing_tables:
            for dest, entry in routing_tables[source].items():
                total_delay += entry.cost
                path_count += 1
        return total_delay / path_count if path_count > 0 else 0
    @staticmethod
    def calculate_path_stretch(graph: nx.Graph, routing_tables: Dict[int, Dict[int, RoutingEntry]]) -> float:
        total_stretch = 0
        count = 0
        for source in routing_tables:
            for dest, entry in routing_tables[source].items():
                try:
                    optimal = nx.shortest_path_length(graph, source, dest, weight='weight')
                    if optimal > 0:
                        stretch = entry.cost / optimal
                        total_stretch += stretch
                        count += 1
                except:
                    pass
        return total_stretch / count if count > 0 else 1.0
