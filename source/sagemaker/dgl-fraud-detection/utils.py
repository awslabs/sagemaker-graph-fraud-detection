import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from mxnet import nd, gluon
import dgl
import networkx as nx
import matplotlib.pyplot as plt


def _get_label_dict(labels_path):
    """
    Returns a dictionary that maps nodes to labels

    :param labels_path: Path to where the labels are store (accepts name of a dynamo table or a local path)
    :return: dict
    """
    tag_df = pd.read_csv(labels_path)
    return {row[tag_df.columns[0]]: row[tag_df.columns[1]] for _, row in tag_df.iterrows()}


def _get_label_for_node(node, node_to_label):
    """
    For a particular node returns the label of that node given a dictionary

    :param node: a node to get the label for
    :param node_to_label: a dictionary that maps nodes to their labels
    :return: np.float the label of the node or nan
    """
    return node_to_label[int(node)] if int(node) in node_to_label else np.nan


def read_masked_nodes(masked_nodes_path):
    """
    Returns a list of nodes extracted from the path passed in

    :param masked_nodes_path: filepath containing list of nodes to be masked
    :return: list
    """
    with open(masked_nodes_path, "r") as fh:
        masked_nodes = [line.strip() for line in fh]
    return masked_nodes


def _get_mask(id_to_node, node_to_id, num_nodes, masked_nodes, additional_mask_rate):
    """
    Returns the train and test mask arrays

    :param id_to_node: dictionary mapping node names(id) to dgl node idx
    :param node_to_id: dictionary mapping dgl node idx to node names(id)
    :param num_nodes: number of user/account nodes in the graph
    :param masked_nodes: list of nodes to be masked during training, nodes without labels
    :param additional_mask_rate: float for additional masking of nodes with labels during training
    :return: (list, list) train and test mask array
    """
    train_mask = np.ones(num_nodes)
    test_mask = np.zeros(num_nodes)
    for node_id in masked_nodes:
        train_mask[id_to_node[node_id]] = 0
        test_mask[id_to_node[node_id]] = 1
    if additional_mask_rate and additional_mask_rate < 1:
        unmasked = np.array([idx for idx in range(num_nodes) if node_to_id[idx] not in masked_nodes])
        yet_unmasked = np.random.permutation(unmasked)[:int(additional_mask_rate*num_nodes)]
        train_mask[yet_unmasked] = 0
    return train_mask, test_mask


def get_labels(id_to_node, labels_path, masked_nodes_path, additional_mask_rate=0):
    """

    :param id_to_node: dictionary mapping node names(id) to dgl node idx
    :param labels_path: filepath containing labelled nodes
    :param masked_nodes_path: filepath containing list of nodes to be masked
    :param additional_mask_rate: additional_mask_rate: float for additional masking of nodes with labels during training
    :return: (list, list) train and test mask array
    """
    node_to_id = {v: k for k, v in id_to_node.items()}
    num_nodes = max(id_to_node.values()) + 1
    user_to_tag = _get_label_dict(labels_path)
    labels = np.array([_get_label_for_node(node_to_id[idx], user_to_tag) for idx in range(num_nodes)])
    masked_nodes = read_masked_nodes(masked_nodes_path)
    train_mask, test_mask = _get_mask(id_to_node, node_to_id,  num_nodes, masked_nodes,
                                      additional_mask_rate=additional_mask_rate)
    return labels, train_mask, test_mask


def get_features(id_to_node, node_features):
    """

    :param id_to_node: dictionary mapping node names(id) to dgl node idx
    :param node_features: path to file containing node features
    :return: (np.ndarray, list) node feature matrix in order and new nodes not yet in the graph
    """
    indices, features, new_nodes = [], [], []
    max_node = max(id_to_node.values())
    with open(node_features, "r") as fh:
        for line in fh:
            node_feats = line.strip().split(",")
            node_id = node_feats[0]
            feats = np.array(list(map(float, node_feats[1:])))
            features.append(feats)
            if node_id not in id_to_node:
                max_node += 1
                id_to_node[node_id] = max_node
                new_nodes.append(max_node)

            indices.append(id_to_node[node_id])

    features = np.array(features).astype('float32')
    features = features[np.argsort(indices), :]
    return features, new_nodes


def get_metrics(model, features, labels, dataloader, g, mask, ctx, out_dir):
    raw_preds = get_model_predictions(model, g, dataloader, features, ctx)
    pred = raw_preds.argmax(axis=1).asnumpy().flatten().astype(int)
    labels, mask = labels.asnumpy().flatten().astype(int), mask.asnumpy().flatten().astype(int)
    labels, pred = labels[np.where(mask)], pred[np.where(mask)]

    acc = ((pred == labels)).sum() / mask.sum()

    true_pos = (np.where(pred == 1, 1, 0) + np.where(labels == 1, 1, 0) > 1).sum()
    false_pos = (np.where(pred == 1, 1, 0) + np.where(labels == 0, 1, 0) > 1).sum()
    false_neg = (np.where(pred == 0, 1, 0) + np.where(labels == 1, 1, 0) > 1).sum()
    true_neg = (np.where(pred == 0, 1, 0) + np.where(labels == 0, 1, 0) > 1).sum()

    precision = true_pos/(true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos/(true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

    f1 = 2*(precision*recall)/(precision + recall) if (precision + recall) > 0 else 0
    confusion_matrix = pd.DataFrame(np.array([[true_pos, false_pos], [false_neg, true_neg]]),
                                    columns=["labels positive", "labels negative"],
                                    index=["predicted positive", "predicted negative"])

    pred_proba = normalize_predictions(raw_preds)[np.where(mask)]
    fpr, tpr, _ = roc_curve(labels, pred_proba)
    roc_auc = auc(fpr, tpr)

    save_roc_curve(fpr, tpr, roc_auc, os.path.join(out_dir, "roc_curve.png"))

    return acc, f1, precision, recall, confusion_matrix


def get_model_predictions(model, g, dataloader, features, ctx):
    pred = []
    for batch in dataloader:
        node_flow, batch_nids = g.sample_block(batch)
        pred.append(model(node_flow, features[batch_nids.as_in_context(ctx)]))
        nd.waitall()
    return nd.concat(*pred, dim=0)


def normalize_predictions(preds):
    return nd.softmax(preds)[:, 1].asnumpy().flatten()


def get_model_class_predictions(model, g, datalaoder, features, ctx, threshold=None):
    unnormalized_preds = get_model_predictions(model, g, datalaoder, features, ctx)
    if not threshold:
        return unnormalized_preds.argmax(axis=1).asnumpy().flatten().astype(int)
    return np.where(normalize_predictions(unnormalized_preds) > threshold, 1, 0)


def save_roc_curve(fpr, tpr, roc_auc, location):
    f = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model ROC curve')
    plt.legend(loc="lower right")
    f.savefig(location)


def save_graph_drawing(g, location):
    plt.figure(figsize=(12, 8))
    node_colors = {node: 0.0 if 'user' in node else 0.5 for node in g.nodes()}
    nx.draw(g, node_size=10000, pos=nx.spring_layout(g), with_labels=True, font_size=14,
            node_color=list(node_colors.values()), font_color='white')
    plt.savefig(location, bbox_inches='tight')

