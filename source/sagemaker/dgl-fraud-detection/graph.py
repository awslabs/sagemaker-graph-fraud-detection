import os
import logging
import dgl

from data import *


def construct_graph(training_dir, edges, nodes, heterogeneous=True):
    if heterogeneous:
        logging.info("Getting relation graphs from the following edge lists : {} ".format(edges))
        edgelists, id_to_node = {}, {}
        for i, edge in enumerate(edges):
            edgelist, id_to_node, src, dst = parse_edgelist(os.path.join(training_dir, edge), id_to_node, header=False)
            edgelists[(src, 'relation{}'.format(i), dst)] = edgelist
            logging.info("Read edges for relation{} from edgelist: {}".format(i, os.path.join(training_dir, edge)))

            # reverse edge list so that relation is undirected
            # edgelists[(dst, 'reverse_relation{}'.format(i), src)] = [(b, a) for a, b in edgelist]


        # get features for nodes
        features, new_nodes = get_features(id_to_node['user'], os.path.join(training_dir, nodes))
        logging.info("Read in user features for user nodes")
        # handle user nodes that have features but don't have any connections
        if new_nodes:
            edgelists[('user', 'relation'.format(i+1), 'none')] = [(node, 0) for node in new_nodes]
            edgelists[('none', 'reverse_relation{}'.format(i + 1), 'user')] = [(0, node) for node in new_nodes]

        # add self relation
        edgelists[('user', 'self_relation', 'user')] = [(node, node) for node in id_to_node['user'].values()]

        g = dgl.heterograph(edgelists)
        logging.info(
            "Constructed heterograph with the following metagraph structure: Node types {}, Edge types{}".format(
                g.ntypes, g.canonical_etypes))
        logging.info("Number of nodes of type user : {}".format(g.number_of_nodes('user')))

        g.nodes['user'].data['features'] = features

        id_to_node = id_to_node['user']

    else:
        sources, sinks, features, id_to_node = read_edges(os.path.join(training_dir, edges[0]),
                                                          os.path.join(training_dir, nodes))

        # add self relation
        all_nodes = sorted(id_to_node.values())
        sources.extend(all_nodes)
        sinks.extend(all_nodes)

        g = dgl.graph((sources, sinks))

        if features:
            g.ndata['features'] = np.array(features).astype('float32')

        logging.info('read graph from node list and edge list')

        features = g.ndata['features']

    return g, features, id_to_node
