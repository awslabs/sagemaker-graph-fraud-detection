# Amazon SageMaker and Deep Graph Library for Fraud Detection in Heterogeneous Graphs

This project shows how to use [Amazon SageMaker](https://aws.amazon.com/sagemaker/) and [Deep Graph Library (DGL)](https://www.dgl.ai/) to train a graph neural network (GNN) model to detect malicious or fraudulent accounts or users.

## Project Organization
The project is divided into two main modules.

The [first module](source/sagemaker/data-preprocessing) uses [Amazon SageMaker Processing](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html) to construct a heterogeneous graph with node features, from a relational table of transactions or interactions. A heterogeneous graph is one where user/account nodes and possibly other entities have several kinds of distinct relationships. Examples include 

* a financial network where users transact with other users as well as specific financial institutions or applications
* a gaming network where users interact with other users but also with distinct games or devices
* a social network where users can have different types of links to other users


The [second module](source/sagemaker/dgl-fraud-detection) shows how to use DGL to define a Graph Neural Network model and train the model using [Amazon SageMaker training infrastructure](https://docs.aws.amazon.com/sagemaker/latest/dg/deep-graph-library.html). Graph Neural Networks (GNNs) are a family of models that can learn low-dimension representation of nodes in a graph using both the neighbourhood structure of the node and the node/edge features.


The project contains a [jupyter notebook](source/sagemaker/dgl-fraud-detection.ipynb) that shows how to run the full project on an [example dataset](https://linqs-data.soe.ucsc.edu/public/social_spammer/).


The project also contains a [cloud formation template](deployment/sagemaker-graph-fraud-detection.yaml) that deploys the code in this repo and all AWS resources needed to run the project in an end-to-end manner in the AWS account it's launched in.

## Contents

* `deployment/`
  * `sagemaker-graph-fraud-detection.yaml`: Creates AWS CloudFormation Stack for solution
* `source/`
  * `lambda/`
    * `data-preprocessing/`
      * `index.py`: Lambda function script for invoking SageMaker processing
    * `graph-modelling/` 
      * `index.py`: Lambda function script for invoking SageMaker training
  * `sagemaker/`
    * `baselines/`
      * `graph-fraud-baseline.ipynb`:  Jupyter notebook for a baseline method using just the graph structure
      * `mlp-fraud-baseline.ipynb`:  Jupyter notebook for feature based MLP baseline method using SageMaker and MXNet
      * `mlp_fraud_entry_point.py`: python entry point used by the MLP baseline notebook for MXNet training/deployment
      * `xgboost-fraud-entry-point.ipynb`: Jupyter notebook for feature based XGBoost baseline method using SageMaker
    * `data-preprocessing/`
      * `container/`
        * `Dockerfile`: Describes custom Docker image hosted on Amazon ECR for SageMaker Processing
        * `build_and_push.sh`: Script to build Docker image and push to Amazon ECR
      * `graph_data_preprocessor.py`: Custom script used by SageMaker Processing for data processing/feature engineering
    * `dgl-fraud-detection/`
      * `data.py`: Contains functions for reading node features and edgelists into DGL Graphs
      * `model.py`: Implements the various graph neural network models used in the project
      * `requirements.txt`: Describes Python package requirements of the Amazon SageMaker training instance
      * `sampler.py`: Contains functions for graph sampling for mini-batch training
      * `train_dgl_entry_point.py`: python entry point used by the SageMaker DGL notebook for GNN training
    * `dgl-fraud-detection.ipynb`: Orchestrates the solution. Triggers preprocessing and model training

## Architecture

The project architecture deployed by the cloud formation template is shown here.

![](deployment/arch.png)

## License

This project is licensed under the Apache-2.0 License.

