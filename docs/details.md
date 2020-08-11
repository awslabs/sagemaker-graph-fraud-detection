# Fraud Detection with Graph Neural Networks

## Overview

Graph Neural Networks (GNNs) have shown promising results in solving problems in various domains from recommendations to fraud detection. For fraud detection, GNN techniques are especially powerful because they can learn representations of users, transactions and other entities in an inductive fashion. This means GNNs are able to model potentially fraudulent users and transactions based on both their features and connectivity patterns in an interaction graph. This allows GNN based fraud detections methods to be resilient towards evasive or camouflaging techniques that are employed by malicious users to fool rules based systems or simple feature based models. However, real world applications of GNNs for fraud detection have been limited due to the complex infrastructure required to train large graphs. This project addresses this issue by using Deep Graph Library (DGL) and Amazon SageMaker to manage the complexity of training a GNN on large graphs for fraud detection.

DGL is an easy-to-use, high performance and scalable Python package for deep learning on graphs. It supports the major deep learning frameworks (Pytorch, MXNet and Tensorflow) as a backend. This project uses DGL to define the graph and implement the GNN models and performs all of the modeling and training using Amazon SageMaker managed resources.

## Problem Description and GNN Formulation

Many businesses lose billions annually to fraud but machine learning based fraud detection models can help businesses predict based on training data what interactions or users are likely fraudulent or malicious and save them from incurring those costs. In this project, we formulate the problem of fraud detection as a classification task, where the machine learning model is a Graph Neural Network that learns good latent representations that can be easily separated into fraud and legitimate. The model is trained using historical transactions or interactions data that contains ground-truth labels for some of the transactions/users.

The interaction data is assumed to be in the form of a relational table or a set of relational tables. The tables record interactions between a user and other users or other relevant entities. From this table, we extract all the different kind of relations and create edge lists per relation type. In order to make the node representation learning inductive, we also assume that the data contains some attributes or information about the user. We use the attributes if they are present, to create initial feature vectors. We can also encode temporal attributes extracted from the interaction table into the user features to capture the temporal behavior of the user in the case where we our interaction data is timestamped.

Using the edge lists, we construct a heterogeneous graph which contains the user nodes and all the other node types corresponding to relevant entities in the edge lists.  A heterogeneous graph is one where user/account nodes and possibly other entities have several kinds of distinct relationships. Examples of use cases that would fall under this include 


* a financial network where users transact with other users as well as specific financial institutions or applications
* a gaming network where users interact with other users but also with distinct games or devices
* a social network where users can have different types of links to other users


Once the graph is constructed, we define an R-GCN model to learn representations for the graph nodes. The R-GCN module is connected to a fully connected neural network layer to perform classification based on the R-GCN learned node representations. The full model is trained end-to-end in a semi-supervised fashion, where the training loss is computed only using information from nodes that have labels.

Overall, the project is divided into two modules: 


* The [first module](../source/sagemaker/data-preprocessing) uses [Amazon SageMaker Processing](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html) to construct a heterogeneous graph with node features, from a relational table of transactions or interactions. 



* The [second module](../source/sagemaker/dgl-fraud-detection) shows how to use DGL to define a Graph Neural Network model and train the model using [Amazon SageMaker training infrastructure](https://docs.aws.amazon.com/sagemaker/latest/dg/deep-graph-library.html). 


To run the full project end to end, use the [jupyter notebook](../source/sagemaker/dgl-fraud-detection.ipynb) 

## Data Processing and Feature Engineering

The data processing and feature engineering steps convert the data from relational form in a table, to a set of edge lists and features for the user nodes. 

Amazon SageMaker Processing is used perform the data processing and feature engineering. The Amazon SageMaker Processing ScriptProcessor requires a docker container with the processing environment and dependencies, and a processing script that defines the actual data processing implementation. All the artifacts necessary for building the processing environment docker container are in the [_container_ folder](../source/sagemaker/data-preprocessing/container). The [_Dockerfile_](source/sagemaker/data-preoprocessing/container/Dockerfile) specifies the content of the container. The only requirements for the data processing script is pandas so that’s the only package that’s installed in the container. 

The actual data processing script is [_graph_data_preprocessor.py_](source/sagemaker/data-preoprocessing/graph_data_preprocessor.py). The script accepts the transaction data and the identity attributes as input arguments and performs a train/validation split by choosing for the validation data new users that joined on after the train_days argument. The script then extracts all the various relations from the relation table and performs deduplication before writing the relations to an output folder. The script also performs feature engineering to encode the categorical features and the temporal features. Finally, if the construct_homogenous argument is passed in, the script also writes a homogeneous edge list that consists only of edges between user nodes to the output folder.

Once the SageMaker Processing instance finishes running the script the files in the output folder are then uploaded to S3 for graph modeling and training.


## Graph Modeling and Training

The graph modeling and training code is implemented using DGL with MXNet as the backend framework and is designed to be run on the managed SageMaker training instances. The [_dgl-fraud-detection_ folder](../source/sagemaker/dgl-fraud-detection) contains the code that is run on those training instances. The supported graph neural network models are defined in [_model.py_](source/sagemaker/dgl-fraud-detection/model.py), and helper functions for graph construction are implemented in [_data.py_](../source/sagemaker/dgl-fraud-detection/data.py). The graph sampling functions for mini-batch graph training are implemented in [sampler.py](../source/sagemaker/dgl-fraud-detection/sampler.py) and [_utils.py_](../source/sagemaker/dgl-fraud-detection/utils.py) contains utility functions. The entry point for the graph modeling and training however is [_train_dgl_entry_point.py_](source/sagemaker/dgl-fraud-detection/train_dgl_entry_point.py).

The entry point script orchestrates the entire graph training process by going through the following steps:


* Reading in the edge lists and user features to construct the graph using the DGLGraph or DGLHeteroGraph API
* Reading in the labels for the target nodes and masking labels for target nodes that won't have labels during training
* Creating the Graph Neural Network model
* Initializing the DataLoader and the Graph Sampler if performing mini-batch graph training
* Initializing the model parameters and training the model


At the end of model training, the script saves the models and metrics or predictions to the the output folder which gets uploaded to S3 before the SageMaker training instance is terminated.



## FAQ

### What is fraud detection?

Fraud is when a malicious actor illicitly or deceitfully tries to obtain goods or services that a business provides means. Fraud detection is a set of techniques for identifying fraudulent cases and distinguishing them from normal or legitimate cases. In this project, we model fraud detection as a semi-supervised learning process, where we have some amount of users that have already been labelled as fraudulent or legitimate, and other users which have no labels during training. The task is to use the trained model to infer the labels for the unlabelled users.

### What are Graphs?

Graphs are a data structure that can be used to represent relationships between entities. They are convenient and flexible way of representing information about interacting entities, and can be easily used to model a lot of real world processes. Graphs consists of a set of entities called the nodes, where pairs of the nodes are connected by links called edges. Lots of systems that exist in the world are networks that are naturally expressed as graphs. Graphs can be directed if the edges have an orientation or are asymmetric, while undirected graphs have symmetric edges. A homogeneous graph consists of nodes and edges of one type while a heterogeneous allows multiple node types and/or edge types.


### What are Graph Neural Networks?

Graph Neural Networks are a family of neural networks that using the graph structure directly to learn useful representations for nodes and edges in a graph and solve graph based tasks like node classification, link prediction or graph classification. Effectively, graph neural networks are message passing networks that learn node representations by using deep learning techniques to aggregate information from neighboring nodes and edges. Popular examples of Graph Neural Networks are [Graph Convolutional Networks (GCN)](https://arxiv.org/abs/1609.02907), [GraphSage](https://arxiv.org/abs/1706.02216), [Graph Attention Networks (GAT)](https://arxiv.org/abs/1710.10903) and [Relational-Graph Convolutional Networks (R-GCN)](https://arxiv.org/abs/1703.06103).


### What makes Graph Neural Networks useful?

One reason Graph neural networks are useful is that they can learn both inductive and transductive representations compared to classical graph learning techniques, like random walks and graph factorizations, that can only learn transductive representations. A transductive representation is one that applies specifically to a particular node instance, while an inductive representation is one that can levarage features of the node, and change change as the node features change, allowing for better generalization. Additionally, varying the depth of a Graph Neural Network allows the network to integrate topologically distant information into the representation of a particular node. Graph Neural Networks are also end-to-end differentiable so they can be trained jointly with a downstream, task-specific model, which allows the downstream model to supervise and tailor the representations learned by the GNN.


### How do Graph Neural Networks work?

Graph Neural Networks are trained, like other Deep Neural Networks, by using gradient based optimizers like sgd or adam to learn network parameter values that optimize a particular loss function. As with other neural networks, this is performed by running a forward step - to compute the feature representations and the loss function, a backward step - to compute the gradients of loss with respect to the network parameters, and an optimize step - to update the network parameter values with the computed gradient. Graph Neural Networks are unique in the forward step. They compute the intermediate representations by a process known as ‘message passing’. For a particular node, this involves using the graph structure to collect all or a subset of the neighboring nodes and edges. At each layer, the intermediate representation of the neighboring nodes and edges are then aggregated into a single message which is combined with the previous intermediate representation of the node to form the new node representation. At the earliest layers, a node representation is informed by it’s eager network - it’s immediate neighbors, but at later layers, the nodes representation is informed by the current representation of the nodes neighbors which had been earlier informed by those neighbor’s neighbors, thus extending the sphere of influence to nodes that are multiple hops away.


### What is an R-GCN Model?

The Relational Graph Convolutional Network (R-GCN) (https://arxiv.org/abs/1703.06103) model  is a GNN that specifically models different edge types and node types differently during message passing and aggregation. Thus, it is especially effective for learning on heterogenous graphs and R-GCN is the default model used in this project for node representation learning. It is based on the simpler GCN architecture but adapted for multi-relational data. 

