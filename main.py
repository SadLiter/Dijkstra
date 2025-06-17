#!/usr/bin/env python3
"""
Dijkstra's Algorithm Implementation and Visualization

Usage:
  python dijkstra.py
"""
import heapq
import networkx as nx
import matplotlib.pyplot as plt


def dijkstra(adj, s):
    """
    Compute the shortest paths from source s to all other vertices in graph.

    Parameters:
    - adj: dict, adjacency list where adj[u] is list of (v, weight)
    - s: source vertex

    Returns:
    - dist: dict of shortest distances to each vertex
    - prev: dict of predecessors on shortest paths
    """
    dist = {v: float('inf') for v in adj}
    prev = {v: None for v in adj}
    dist[s] = 0
    heap = [(0, s)]

    while heap:
        d_u, u = heapq.heappop(heap)
        if d_u > dist[u]:
            continue
        for v, w in adj[u]:
            alt = d_u + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))
    return dist, prev


def visualize_dijkstra(adj, s):
    """
    Visualize the step-by-step execution of Dijkstra's algorithm.

    Parameters:
    - adj: dict, adjacency list
    - s: source vertex
    """
    G = nx.DiGraph()
    for u, neighbors in adj.items():
        for v, w in neighbors:
            G.add_edge(u, v, weight=w)
    pos = nx.spring_layout(G)

    dist = {v: float('inf') for v in G}
    dist[s] = 0
    visited = set()
    heap = [(0, s)]
    step = 0

    while heap:
        d_u, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        plt.figure()
        node_colors = ['lightgreen' if n in visited else 'lightblue' for n in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_color=node_colors)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title(f"Step {step}: processed vertex {u}")
        plt.show()
        step += 1

        for v, w in adj[u]:
            alt = dist[u] + w
            if alt < dist[v]:
                dist[v] = alt
                heapq.heappush(heap, (alt, v))


if __name__ == "__main__":
    # Example usage
    adjacency = {
        'A': [('B', 3), ('C', 1)],
        'B': [('D', 2)],
        'C': [('B', 1), ('D', 5)],
        'D': []
    }
    dist, prev = dijkstra(adjacency, 'A')
    print("Shortest distances:", dist)
    print("Predecessors:", prev)

    # Uncomment to visualize
    visualize_dijkstra(adjacency, 'A')