import numpy as np


def _get_node_idx(id_to_node, node_type, node_id, ptr):
    if node_type in id_to_node:
        if node_id in id_to_node[node_type]:
            node_idx = id_to_node[node_type][node_id]
        else:
            id_to_node[node_type][node_id] = ptr
            node_idx = ptr
            ptr += 1
    else:
        id_to_node[node_type] = {}
        id_to_node[node_type][node_id] = ptr
        node_idx = ptr
        ptr += 1

    return node_idx, id_to_node, ptr


def parse_edgelist(edges, id_to_node, header=False, source_type='user', sink_type='user'):
    """
    Parse an edgelist path file and return the edges as a list of tuple
    :param edges: path to comma separated file containing bipartite edges with header for edgetype
    :param id_to_node: dictionary containing mapping for node names(id) to dgl node indices
    :param header: boolean whether or not the file has a header row
    :param source_type: type of the source node in the edge. defaults to 'user' if no header
    :param sink_type: type of the sink node in the edge. defaults to 'user' if no header.
    :return: (list, dict) a list containing edges of a single relationship type as tuples and updated id_to_node dict.
    """
    edge_list = []
    source_pointer, sink_pointer = 0, 0
    with open(edges, "r") as fh:
        for i, line in enumerate(fh):
            source, sink = line.strip().split(",")
            if i == 0:
                if header:
                    source_type, sink_type = source, sink
                if source_type in id_to_node:
                    source_pointer = max(id_to_node[source_type].values()) + 1
                if sink_type in id_to_node:
                    sink_pointer = max(id_to_node[sink_type].values()) + 1
                continue

            source_node, id_to_node, source_pointer = _get_node_idx(id_to_node, source_type, source, source_pointer)
            if source_type == sink_type:
                sink_node, id_to_node, source_pointer = _get_node_idx(id_to_node, sink_type, sink, source_pointer)
            else:
                sink_node, id_to_node, sink_pointer = _get_node_idx(id_to_node, sink_type, sink, sink_pointer)

            edge_list.append((source_node, sink_node))

    return edge_list, id_to_node, source_type, sink_type


def from_csv(g, edges, nodes=None):
    """
    Add nodes and edges to an empty DGL Graph from paths to node features and edge list

    :param g: empty DGL graph
    :param edges: path to comma separated file containing all edges
    :param nodes: path to comma separated file containing all nodes + features
    :return: (G, dict) passed in DGL graph with nodes and edges added, id_to_node dictionary containing mappings
    from node names(id) to dgl node indices
    """
    node_pointer = 0
    id_to_node = {}
    features = []
    sources, sinks = [], []
    if nodes is not None:
        with open(nodes, "r") as fh:
            for line in fh:
                node_feats = line.strip().split(",")
                node_id = node_feats[0]
                if node_id not in id_to_node:
                    id_to_node[node_id] = node_pointer
                    node_pointer += 1
                    if len(node_feats) > 1:
                        feats = np.array(list(map(float, node_feats[1:])))
                        features.append(feats)
        with open(edges, "r") as fh:
            for line in fh:
                source, sink = line.strip().split(",")
                sources.append(id_to_node[source])
                sinks.append(id_to_node[sink])
    else:
        with open(edges, "r") as fh:
            for line in fh:
                source, sink = line.strip().split(",")
                if source not in id_to_node:
                    id_to_node[source] = node_pointer
                    node_pointer += 1
                if sink not in id_to_node:
                    id_to_node[sink] = node_pointer
                    node_pointer += 1
                sources.append(id_to_node[source])
                sinks.append(id_to_node[sink])

    g.add_nodes(node_pointer)
    g.add_edges(sources, sinks)
    if features:
        g.ndata['features'] = np.array(features).astype('float32')
    return g, id_to_node
