import os
import pandas as pd

def get_data():
    data_prefix = "processed-data/"

    if not os.path.exists(data_prefix):
        print("""Expected the following folder {} to contain the preprocessed data. 
                 Run data processing first in main notebook before running baselines comparisons""".format(data_prefix))
        return

    features = pd.read_csv(data_prefix + "features.csv", header=None)
    labels = pd.read_csv(data_prefix + "tags.csv").set_index('TransactionID')
    test_users = pd.read_csv(data_prefix + "test.csv", header=None)

    test_X = features.merge(test_users, on=[0], how='inner')
    train_X = features[~features[0].isin(test_users[0].values)]
    test_y = labels.loc[test_X[0]].isFraud
    train_y = labels.loc[train_X[0]].isFraud

    return train_X, train_y, test_X, test_y