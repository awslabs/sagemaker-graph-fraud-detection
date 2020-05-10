import os
os.environ['DGLBACKEND'] = 'mxnet'
import mxnet as mx
from mxnet import nd, gluon, autograd
import dgl

import numpy as np
import pandas as pd

import argparse
import time
import logging
import pickle

from data import *
from utils import *
from model import *
from sampler import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--nodes', type=str, default='user_features.csv')
    parser.add_argument('--edges', type=str, default='homogeneous_user_edgelist.csv')
    parser.add_argument('--heterogeneous', type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        default=True, help='use hetero graph')
    parser.add_argument('--no-features', type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        default=False, help='do not use node features')
    parser.add_argument('--mini-batch', type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        default=True, help='use mini-batch training and sample graph')
    parser.add_argument('--labels', type=str, default='tags.csv')
    parser.add_argument('--new-accounts', type=str, default='test_users.csv')
    parser.add_argument('--predictions', type=str, default='preds.csv', help='file to save predictions on new-accounts')
    parser.add_argument('--compute-metrics', type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        default=True, help='compute evaluation metrics after training')
    parser.add_argument('--threshold', type=float, default=0, help='threshold for making predictions, default : argmax')
    parser.add_argument('--model', type=str, default='rgcn', help='gnn to use. options: gcn, graphsage, gat, gem')
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n-epochs', type=int, default=20)
    parser.add_argument('--n-neighbors', type=int, default=10, help='number of neighbors to sample')
    parser.add_argument('--n-hidden', type=int, default=16, help='number of hidden units')
    parser.add_argument('--n-layers', type=int, default=3, help='number of hidden layers')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight for L2 loss')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout probability, for gat only features')
    parser.add_argument('--attn-drop', type=float, default=0.6, help='attention dropout for gat/gem')
    parser.add_argument('--num-heads', type=int, default=4, help='number of hidden attention heads for gat/gem')
    parser.add_argument('--num-out-heads', type=int, default=1, help='number of output attention heads for gat/gem')
    parser.add_argument('--residual', action="store_true", default=False, help='use residual connection for gat')
    parser.add_argument('--alpha', type=float, default=0.2, help='the negative slop of leaky relu')
    parser.add_argument('--aggregator-type', type=str, default="gcn", help="graphsage aggregator: mean/gcn/pool/lstm")
    parser.add_argument('--embedding-size', type=int, default=360, help="embedding size for node embedding")

    return parser.parse_args()


def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    return logger


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

        g = dgl.heterograph(edgelists)
        logging.info(
            "Constructed heterograph with the following metagraph structure: Node types {}, Edge types{}".format(
                g.ntypes, g.canonical_etypes))
        logging.info("Number of nodes of type user : {}".format(g.number_of_nodes('user')))

        features = nd.array(features)
        g.nodes['user'].data['features'] = features

        id_to_node = id_to_node['user']

    else:
        g = dgl.DGLGraph()
        g, id_to_node = from_csv(g, os.path.join(training_dir, edges[0]), os.path.join(training_dir, nodes))

        logging.info('read graph from node list and edge list')

        features = normalize(g.ndata['features'])
        g.ndata['features'] = features

    return g, features, id_to_node


def normalize(feature_matrix):
    mean = nd.mean(feature_matrix, axis=0)
    stdev = nd.sqrt(nd.sum((feature_matrix - mean)**2, axis=0)/feature_matrix.shape[0])
    return (feature_matrix - mean) / stdev


def get_dataloader(data_size, batch_size, mini_batch=True):
    batch_size = batch_size if mini_batch else data_size
    train_dataloader = gluon.data.BatchSampler(gluon.data.RandomSampler(data_size), batch_size, 'keep')
    test_dataloader = gluon.data.BatchSampler(gluon.data.SequentialSampler(data_size), batch_size, 'keep')

    return train_dataloader, test_dataloader


def train(model, trainer, loss, features, labels, train_loader, test_loader, train_g, test_g, train_mask, test_mask,
          ctx, n_epochs, batch_size, output_dir, thresh, scale_pos_weight, compute_metrics=True, mini_batch=True):
    duration = []
    for epoch in range(n_epochs):
        tic = time.time()
        loss_val = 0.

        for n, batch in enumerate(train_loader):
            # logging.info("Iteration: {:05d}".format(n))
            node_flow, batch_nids = train_g.sample_block(batch)
            batch_indices = nd.array(batch, ctx=ctx)
            with autograd.record():
                pred = model(node_flow, features[batch_nids.as_in_context(ctx)])
                l = loss(pred, labels[batch_indices], mx.nd.expand_dims(scale_pos_weight*train_mask, 1)[batch_indices])
                l = l.sum()/len(batch)

            l.backward()
            trainer.step(batch_size=1, ignore_stale_grad=True)

            loss_val += l.asscalar()
            # logging.info("Current loss {:04f}".format(loss_val/(n+1)))

        duration.append(time.time() - tic)
        metric = evaluate(model, train_g, features, labels, train_mask, ctx, batch_size, mini_batch)
        logging.info("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | F1 {:.4f} | ETputs(KTEPS) {:.2f}".format(
                epoch, np.mean(duration), loss_val/(n+1), metric, n_edges / np.mean(duration) / 1000))

    class_preds, pred_proba = get_model_class_predictions(model, test_g, test_loader, features, ctx, threshold=thresh)

    if compute_metrics:
        acc, f1, p, r, roc, cm = get_metrics(class_preds, pred_proba, labels, test_mask, output_dir)
        logging.info("Metrics")
        logging.info("""Confusion Matrix: 
                        {}
                        f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, acc: {:.4f}, roc: {:.4f}
                     """.format(cm, f1, p, r, acc, roc))

    return model, class_preds, pred_proba


def evaluate(model, g, features, labels, mask, ctx, batch_size, mini_batch=True):
    f1 = mx.metric.F1()
    preds = []
    batch_size = batch_size if mini_batch else features.shape[0]
    dataloader = gluon.data.BatchSampler(gluon.data.SequentialSampler(features.shape[0]),  batch_size, 'keep')
    for batch in dataloader:
        node_flow, batch_nids = g.sample_block(batch)
        preds.append(model(node_flow, features[batch_nids.as_in_context(ctx)]))
        nd.waitall()

    # preds = nd.concat(*preds, dim=0).argmax(axis=1)
    preds = nd.concat(*preds, dim=0)
    mask = nd.array(np.where(mask.asnumpy()), ctx=ctx)
    f1.update(preds=nd.softmax(preds[mask], axis=1).reshape(-3, 0), labels=labels[mask].reshape(-1,))
    return f1.get()[1]


def get_model_predictions(model, g, dataloader, features, ctx):
    pred = []
    for batch in dataloader:
        node_flow, batch_nids = g.sample_block(batch)
        pred.append(model(node_flow, features[batch_nids.as_in_context(ctx)]))
        nd.waitall()
    return nd.concat(*pred, dim=0)


def get_model_class_predictions(model, g, datalaoder, features, ctx, threshold=None):
    unnormalized_preds = get_model_predictions(model, g, datalaoder, features, ctx)
    pred_proba = nd.softmax(unnormalized_preds)[:, 1].asnumpy().flatten()
    if not threshold:
        return unnormalized_preds.argmax(axis=1).asnumpy().flatten().astype(int), pred_proba
    return np.where(pred_proba > threshold, 1, 0), pred_proba


def save_prediction(pred, pred_proba, id_to_node, training_dir, new_accounts, output_dir, predictions_file):
    prediction_query = read_masked_nodes(os.path.join(training_dir, new_accounts))
    pred_indices = np.array([id_to_node[query] for query in prediction_query])

    pd.DataFrame.from_dict({'user': prediction_query,
                            'pred_proba': pred_proba[pred_indices],
                            'pred': pred[pred_indices]}).to_csv(os.path.join(output_dir, predictions_file),
                                                                index=False)


def save_model(g, model, model_dir, hyperparams):
    model.save_parameters(os.path.join(model_dir, 'model.params'))
    with open(os.path.join(model_dir, 'model_hyperparams.pkl'), 'wb') as f:
        pickle.dump(hyperparams, f)
    with open(os.path.join(model_dir, 'graph.pkl'), 'wb') as f:
        pickle.dump(g, f)


def get_model(g, hyperparams, in_feats, n_classes, ctx, model_dir=None):

    if model_dir:  # load using saved model state
        with open(os.path.join(model_dir, 'model_hyperparams.pkl'), 'rb') as f:
            hyperparams = pickle.load(f)
        with open(os.path.join(model_dir, 'graph.pkl'), 'rb') as f:
            g = pickle.load(f)

    if hyperparams['heterogeneous']:
        model = HeteroRGCN(g,
                           in_feats,
                           hyperparams['n_hidden'],
                           n_classes,
                           hyperparams['n_layers'],
                           hyperparams['embedding_size'],
                           ctx)
    else:
        if hyperparams['model'] == 'gcn':
            model = GCN(g,
                        in_feats,
                        hyperparams['n_hidden'],
                        n_classes,
                        hyperparams['n_layers'],
                        nd.relu,
                        hyperparams['dropout'])
        elif hyperparams['model'] == 'graphsage':
            model = GraphSAGE(g,
                              in_feats,
                              hyperparams['n_hidden'],
                              n_classes,
                              hyperparams['n_hidden'],
                              nd.relu,
                              hyperparams['dropout'],
                              hyperparams['aggregator_type'])
        else:
            heads = ([hyperparams['num_heads']] * hyperparams['n_layers']) + [hyperparams['num_out_heads']]
            model = GAT(g,
                        in_feats,
                        hyperparams['n_hidden'],
                        n_classes,
                        hyperparams['n_layers'],
                        heads,
                        gluon.nn.Lambda(lambda data: nd.LeakyReLU(data, act_type='elu')),
                        hyperparams['dropout'],
                        hyperparams['attn_drop'],
                        hyperparams['alpha'],
                        hyperparams['residual'])

    if hyperparams['no_features']:
        model = NodeEmbeddingGNN(model, in_feats, hyperparams['embedding_size'])

    if model_dir:
        model.load_parameters(os.path.join(model_dir, 'model.params'))
    else:
        model.initialize(ctx=ctx)

    return model


if __name__ == '__main__':
    logging = get_logger(__name__)
    logging.info('numpy version:{} MXNet version:{} DGL version:{}'.format(np.__version__,
                                                                           mx.__version__,
                                                                           dgl.__version__))

    args = parse_args()

    args.edges = args.edges.split(",")

    g, features, id_to_node = construct_graph(args.training_dir, args.edges, args.nodes, args.heterogeneous)

    logging.info("Getting labels")
    labels, train_mask, test_mask = get_labels(id_to_node,
                                               g.number_of_nodes('user'),
                                               os.path.join(args.training_dir, args.labels),
                                               os.path.join(args.training_dir, args.new_accounts))
    logging.info("Got labels")

    labels = nd.array(labels).astype('float32')
    train_mask = nd.array(train_mask).astype('float32')
    test_mask = nd.array(test_mask).astype('float32')

    n_nodes = sum([g.number_of_nodes(n_type) for n_type in g.ntypes]) if args.heterogeneous else g.number_of_nodes()
    n_edges = sum([g.number_of_edges(e_type) for e_type in g.etypes]) if args.heterogeneous else g.number_of_edges()

    logging.info("""----Data statistics------'
                      #Nodes: {}
                      #Edges: {}
                      #Features Shape: {}
                      #Labeled Train samples: {}
                      #Unlabeled Test samples: {}""".format(n_nodes,
                                                            n_edges,
                                                            features.shape,
                                                            train_mask.sum().asscalar(),
                                                            test_mask.sum().asscalar()))

    if args.num_gpus:
        cuda = True
        ctx = mx.gpu(0)
    else:
        cuda = False
        ctx = mx.cpu(0)

    logging.info("Initializing Model")
    in_feats = args.embedding_size if args.no_features else features.shape[1]
    n_classes = 2
    model = get_model(g, vars(args), in_feats, n_classes, ctx)
    logging.info("Initialized Model")

    if args.no_features:
        features = nd.array(g.nodes('user'), ctx) if args.heterogeneous else nd.array(g.nodes(), ctx)
    else:
        features = features.as_in_context(ctx)

    labels = labels.as_in_context(ctx)
    train_mask = train_mask.as_in_context(ctx)
    test_mask = test_mask.as_in_context(ctx)

    if not args.heterogeneous:
        # normalization
        degs = g.in_degrees().astype('float32')
        norm = mx.nd.power(degs, -0.5)
        if cuda:
            norm = norm.as_in_context(ctx)
        g.ndata['norm'] = mx.nd.expand_dims(norm, 1)

    if args.mini_batch:
        train_g = HeteroGraphNeighborSampler(g, 'user', args.n_layers, args.n_neighbors) if args.heterogeneous\
            else NeighborSampler(g, args.n_layers, args.n_neighbors)

        test_g = HeteroGraphNeighborSampler(g, 'user', args.n_layers) if args.heterogeneous\
            else NeighborSampler(g, args.n_layers)
    else:
        train_g, test_g = FullGraphSampler(g, args.n_layers), FullGraphSampler(g, args.n_layers)

    train_data, test_data = get_dataloader(features.shape[0], args.batch_size, args.mini_batch)

    loss = gluon.loss.SoftmaxCELoss()
    scale_pos_weight = ((train_mask.shape[0] - train_mask.sum()) / train_mask.sum())

    logging.info(model)
    logging.info(model.collect_params())
    trainer = gluon.Trainer(model.collect_params(), args.optimizer, {'learning_rate': args.lr, 'wd': args.weight_decay})

    logging.info("Starting Model training")
    model, pred, pred_proba = train(model, trainer, loss, features, labels, train_data, test_data, train_g, test_g,
                                    train_mask, test_mask, ctx, args.n_epochs, args.batch_size, args.output_dir,
                                    args.threshold, scale_pos_weight, args.compute_metrics, args.mini_batch)
    logging.info("Finished Model training")

    logging.info("Saving model")
    save_model(g, model, args.model_dir, vars(args))

    logging.info("Saving model predictions for new accounts")
    save_prediction(pred, pred_proba, id_to_node, args.training_dir, args.new_accounts, args.output_dir, args.predictions)
