#!/usr/bin/env python3
"""
Macro clustering utility for ERPlacer netlists.

Strategy:
  For each macro, BFS up to depth N in the instance-net bipartite graph.
  Collect all reachable instances. Build macro-macro graph weighted by
  neighborhood overlap. Run community detection on that graph.

Usage:
    python3 tools/macro_clusters.py netlist.csv --num-movable 539595 [--bfs-depth 2]

Requires: networkx, python-louvain (pip install networkx python-louvain)
"""

import argparse
import csv
import sys
from collections import defaultdict, deque

def load_netlist(path):
    """Returns:
      instance_to_nets: {instance_id: [net_id, ...]}
      net_to_instances: {net_id: [instance_id, ...]}
    """
    instance_to_nets = defaultdict(list)
    net_to_instances = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            net_id = int(row['net_id'])
            instance_id = int(row['instance_id'])
            instance_to_nets[instance_id].append(net_id)
            net_to_instances[net_id].append(instance_id)
    return instance_to_nets, net_to_instances

def bfs_neighborhood(start_instance, instance_to_nets, net_to_instances, depth):
    """BFS on instance-net bipartite graph up to `depth` hops.
    One hop = instance → net → instances.
    Returns set of reachable instance_ids (excluding start)."""
    visited_instances = {start_instance}
    visited_nets = set()
    # frontier is instances
    frontier = {start_instance}
    for _ in range(depth):
        next_frontier = set()
        for inst in frontier:
            for net_id in instance_to_nets.get(inst, []):
                if net_id in visited_nets:
                    continue
                visited_nets.add(net_id)
                for neighbor in net_to_instances.get(net_id, []):
                    if neighbor not in visited_instances:
                        visited_instances.add(neighbor)
                        next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    visited_instances.discard(start_instance)
    return visited_instances

def macro_distances(macro_ids_raw, instance_to_nets, net_to_instances):
    """BFS distance between every pair of macros (hops through instance-net graph).
    One hop = instance → net → instance."""
    macro_set = set(macro_ids_raw)
    distances = {}  # (a, b) -> min hops

    for start in macro_ids_raw:
        # BFS from start, track distance in hops (each hop crosses one net)
        dist = {start: 0}
        # frontier: (instance_id, hop_count)
        queue = deque([(start, 0)])
        visited_nets = set()
        while queue:
            inst, d = queue.popleft()
            for net_id in instance_to_nets.get(inst, []):
                if net_id in visited_nets:
                    continue
                visited_nets.add(net_id)
                for neighbor in net_to_instances.get(net_id, []):
                    if neighbor not in dist:
                        dist[neighbor] = d + 1
                        queue.append((neighbor, d + 1))
        for other in macro_ids_raw:
            if other != start and other in dist:
                distances[(start, other)] = dist[other]
    return distances
    """Build weighted macro-macro graph from neighborhood overlap."""
    try:
        import networkx as nx
    except ImportError:
        print("pip install networkx", file=sys.stderr)
        sys.exit(1)

    macro_set = set(macro_ids)
    G = nx.Graph()
    G.add_nodes_from(macro_ids)

    macro_list = list(macro_ids)
    for i in range(len(macro_list)):
        for j in range(i + 1, len(macro_list)):
            a, b = macro_list[i], macro_list[j]
            # Weight = size of intersection of neighborhoods
            overlap = len(neighborhoods[a] & neighborhoods[b])
            if overlap > 0:
                G.add_edge(a, b, weight=overlap)
    return G

def build_macro_graph(macro_ids, neighborhoods):
    """Build weighted macro-macro graph from neighborhood overlap (intersection size)."""
    try:
        import networkx as nx
    except ImportError:
        print("pip install networkx", file=sys.stderr)
        sys.exit(1)
    G = nx.Graph()
    G.add_nodes_from(macro_ids)
    for i in range(len(macro_ids)):
        for j in range(i + 1, len(macro_ids)):
            a, b = macro_ids[i], macro_ids[j]
            overlap = len(neighborhoods[a] & neighborhoods[b])
            if overlap > 0:
                G.add_edge(a, b, weight=overlap)
    return G

def run_community_detection(G, method="louvain"):
    import networkx as nx
    if method == "louvain":
        try:
            import community as community_louvain
            return community_louvain.best_partition(G, weight='weight')
        except ImportError:
            pass
    if method == "spectral":
        try:
            import numpy as np
            from sklearn.cluster import SpectralClustering
            nodes = list(G.nodes())
            n = len(nodes)
            A = np.zeros((n, n))
            for i, u in enumerate(nodes):
                for j, v in enumerate(nodes):
                    if G.has_edge(u, v):
                        A[i][j] = G[u][v]['weight']
            k = max(2, n // 4)
            labels = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42).fit_predict(A)
            return {nodes[i]: int(labels[i]) for i in range(n)}
        except ImportError:
            print("pip install scikit-learn numpy for spectral clustering", file=sys.stderr)
    # Fallback: greedy modularity
    partition = {}
    for cid, component in enumerate(nx.algorithms.community.greedy_modularity_communities(G, weight='weight')):
        for node in component:
            partition[node] = cid
    return partition

def main():
    parser = argparse.ArgumentParser(description="Macro cluster analysis via BFS neighborhood")
    parser.add_argument("netlist", help="Path to netlist CSV (net_id,pin_id,instance_id)")
    parser.add_argument("--num-movable", type=int, required=True,
                        help="Number of movable instances (fixed instances start at this index)")
    parser.add_argument("--bfs-depth", type=int, default=2,
                        help="BFS depth from each macro (default: 2)")
    parser.add_argument("--exclude-macros", type=int, nargs="+", default=[],
                        help="Macro indices to exclude (e.g. --exclude-macros 16)")
    parser.add_argument("--show-graph", action="store_true",
                        help="Plot the macro-macro graph (requires matplotlib)")
    args = parser.parse_args()

    print(f"Loading netlist from {args.netlist}...")
    instance_to_nets, net_to_instances = load_netlist(args.netlist)
    all_instances = set(instance_to_nets.keys())
    macro_ids_raw = sorted(i for i in all_instances if i >= args.num_movable)
    macro_index = {iid: idx for idx, iid in enumerate(macro_ids_raw)}
    excluded_iids = {macro_ids_raw[i] for i in args.exclude_macros if i < len(macro_ids_raw)}
    if excluded_iids:
        print(f"  Excluding macros: {sorted(args.exclude_macros)} (instance ids: {sorted(excluded_iids)})")
        # Remove excluded instances from net_to_instances so BFS won't traverse through them
        for net_id in list(net_to_instances.keys()):
            net_to_instances[net_id] = [i for i in net_to_instances[net_id] if i not in excluded_iids]
        macro_ids_raw = [iid for iid in macro_ids_raw if iid not in excluded_iids]
        macro_index = {iid: idx for idx, iid in enumerate(macro_ids_raw)}
    print(f"  {len(macro_ids_raw)} macros, {len(all_instances)} total instances, "
          f"{len(net_to_instances)} nets")

    print(f"BFS depth={args.bfs_depth} from each macro...")
    neighborhoods = {}  # macro_idx -> set of reachable instance_ids
    for iid in macro_ids_raw:
        idx = macro_index[iid]
        nbr = bfs_neighborhood(iid, instance_to_nets, net_to_instances, args.bfs_depth)
        neighborhoods[idx] = nbr
        print(f"  macro {idx} (id={iid}): {len(nbr)} reachable instances")

    # Remap macro ids to indices
    macro_ids_idx = list(range(len(macro_ids_raw)))

    print(f"Computing macro-to-macro BFS distances...")
    distances = macro_distances(macro_ids_raw, instance_to_nets, net_to_instances)

    # Print distance matrix
    n = len(macro_ids_raw)
    print(f"\nMacro-to-macro hop distances (rows=from, cols=to):")
    header = "     " + "".join(f"{i:5}" for i in range(n))
    print(header)
    for i, a in enumerate(macro_ids_raw):
        row = f"{i:3}  " + "".join(
            f"{distances.get((a, b), -1):5}" for b in macro_ids_raw
        )
        print(row)

    # Build distance-based graph: edge weight = 1/distance (closer = stronger)
    print("\nBuilding macro-macro distance graph...")
    import networkx as nx
    G_dist = nx.Graph()
    G_dist.add_nodes_from(range(n))
    for i, a in enumerate(macro_ids_raw):
        for j, b in enumerate(macro_ids_raw):
            if i >= j:
                continue
            d = distances.get((a, b))
            if d is not None and d > 0:
                G_dist.add_edge(i, j, weight=1.0 / d)

    # Also build overlap graph
    G_overlap = build_macro_graph(macro_ids_idx, neighborhoods)

    print(f"  distance graph: {G_dist.number_of_edges()} edges")
    print(f"  overlap graph:  {G_overlap.number_of_edges()} edges")

    spectral_partition = None
    for method, G_use in [("louvain (overlap)", G_overlap), ("greedy_modularity (overlap)", G_overlap),
                           ("spectral (overlap)", G_overlap)]:
        algo = method.split()[0]
        print(f"\nCommunity detection — {method}:")
        partition = run_community_detection(G_use, method=algo)
        if algo == "spectral":
            spectral_partition = partition
        clusters = defaultdict(list)
        for node, cid in partition.items():
            clusters[cid].append(node)
        print(f"  {len(clusters)} communities:")
        for cid, members in sorted(clusters.items()):
            print(f"  cluster {cid}: macros {sorted(members)}")

    G = G_overlap  # use for visualization
    partition = spectral_partition or partition

    # Write cluster colors CSV for Rust visualizer (keyed by instance_id)
    import matplotlib.pyplot as plt
    cmap = plt.cm.tab10
    colors_path = "macro_colors.csv"
    best_partition = spectral_partition or partition
    with open(colors_path, "w") as f:
        f.write("instance_id,r,g,b\n")
        for node, cid in best_partition.items():
            instance_id = macro_ids_raw[node]  # map index back to instance_id
            rgba = cmap(cid / max(1, max(best_partition.values())))
            r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
            f.write(f"{instance_id},{r},{g},{b}\n")
    print(f"\nWrote cluster colors to {colors_path}")
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if args.show_graph:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import networkx as nx
            import numpy as np

            fig, ax = plt.subplots(figsize=(10, 10))

            # Group nodes by cluster, arrange clusters in a circle
            clusters_map = defaultdict(list)
            for node, cid in partition.items():
                clusters_map[cid].append(node)
            num_clusters = len(clusters_map)
            cmap = plt.cm.tab10

            pos = {}
            cluster_radius = 2.5
            node_radius = 0.6
            for ci, (cid, members) in enumerate(sorted(clusters_map.items())):
                # Cluster center on a big circle
                angle = 2 * np.pi * ci / num_clusters
                cx, cy = cluster_radius * np.cos(angle), cluster_radius * np.sin(angle)
                # Nodes arranged in a small circle within cluster
                for ni, node in enumerate(members):
                    a = 2 * np.pi * ni / len(members)
                    pos[node] = (cx + node_radius * np.cos(a), cy + node_radius * np.sin(a))

            colors = [cmap(partition[n] / max(max(partition.values()), 1)) for n in G.nodes()]
            weights = [G[u][v]['weight'] for u, v in G.edges()]
            min_w, max_w = min(weights), max(weights)
            # Normalize edge widths — emphasize differences
            norm_w = [(w - min_w) / (max_w - min_w + 1e-9) * 3 + 0.2 for w in weights]

            nx.draw_networkx_edges(G, pos, ax=ax, width=norm_w, alpha=0.4, edge_color='gray')
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=900)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')

            # Legend
            patches = [mpatches.Patch(color=cmap(cid / max(max(clusters_map.keys()), 1)),
                       label=f"cluster {cid}: {sorted(members)}")
                       for cid, members in sorted(clusters_map.items())]
            ax.legend(handles=patches, loc='upper right', fontsize=9)
            ax.set_title(f"Macro Spectral Clusters (BFS depth={args.bfs_depth})")
            ax.axis('off')
            plt.tight_layout()
            plt.savefig("macro_graph.png", dpi=150)
            print("\nSaved macro_graph.png")
        except ImportError as e:
            print(f"pip install matplotlib: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
