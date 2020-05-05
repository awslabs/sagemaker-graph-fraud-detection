import argparse
import logging
import os
import multiprocessing

import pandas as pd
import numpy as np
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-days', type=int, default=0,
                        help='how many days before which new users are not included in test data.')
    parser.add_argument('--data-dir', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output')
    parser.add_argument('--user-data', type=str, default='usersdata.csv.gz', help='name of operations file')
    parser.add_argument('--relations', type=str, default='relations.csv.gz', help='name of transactions file')
    parser.add_argument('--tags', type=str, default='tags.csv', help='name of tags file')
    parser.add_argument('--construct-homogeneous', action="store_true", default=False,
                        help='use bipartite graphs edgelists to construct homogenous graph edgelist')
    return parser.parse_args()


def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    return logger


def load_data():
    user_cols = ['userId', 'sex', 'timePassedValidation', 'ageGroup', 'label']
    users_df = pd.read_csv(os.path.join(args.data_dir, args.user_data),
                           sep='\t', compression='gzip', names=user_cols)
    logging.info("Shape of users data is {}".format(users_df.shape))
    logging.info("Number of users who have been tagged {}".format(len(users_df) - users_df.label.isnull().sum()))

    relations_cols = ["day", "ms", "src", "dst", "relation"]
    relations_df = pd.read_csv(os.path.join(args.data_dir, args.relations),
                               sep='\t', compression='gzip', names=relations_cols)
    logging.info("Shape of relations data is {}".format(relations_df.shape))

    # Take only first two days of relations for demo
    relations_df = relations_df[relations_df.day < 2]

    users_df = users_df.merge(pd.DataFrame({'userId': relations_df[["src", "dst"]].stack().drop_duplicates().values}),
                              on='userId',
                              how='inner')

    return users_df, relations_df


def get_test_users(users_df, relations_df, train_days):
    train_relations_df = relations_df.iloc[(relations_df.day < train_days).values]
    test_relations_df = relations_df.iloc[(relations_df.day >= train_days).values]

    all_users_in_train = train_relations_df.src.unique()
    logging.info("Number of total users in train {}".format(len(all_users_in_train)))

    all_users_in_test = test_relations_df.src.unique()
    logging.info("Number of total users in test {}".format(len(all_users_in_test)))

    new_users_test = set(all_users_in_test) - set(all_users_in_train)
    logging.info("Number of new users in test {}".format(len(new_users_test)))

    new_users_tags = users_df[users_df.userId.isin(new_users_test)]['label'].values
    known_users_tags = users_df[users_df.userId.isin(all_users_in_train)]['label'].values

    get_freq = lambda counts: (counts, {key: value/sum(counts.values()) for key, value in counts.items()})

    logging.info("Label frequency {} and distribution {} for new users".format(*get_freq(Counter(new_users_tags))))
    logging.info("Label frequency {} and distribution {} for known users".format(*get_freq(Counter(known_users_tags))))
    logging.info("Label frequency {} and distribution {} for all users".format(*get_freq(Counter(users_df.label.values))))

    with open(os.path.join(args.output_dir, 'test_users.csv'), 'w') as f:
        f.writelines(map(lambda x: str(x) + "\n", new_users_test))

    logging.info("Wrote test users to file: {}".format(os.path.join(args.output_dir, 'test_users.csv')))

    return new_users_test


def extract_activity_features(rel_df):
    activity_features_size = 2 * 24
    time_fts = np.zeros((rel_df.shape[0], activity_features_size))
    rel_df['hr'] = rel_df['ms'].apply(lambda x: int(x//3.6e6))
    time_fts[np.arange(time_fts.shape[0]), (rel_df['day'].values*24 + rel_df['hr'].values)] = 1
    time_df = pd.DataFrame(time_fts)
    time_df['userId'] = rel_df['src']
    return time_df.groupby('userId').sum().reset_index()


def parallelize_feature_extraction(df, func, n_cores=multiprocessing.cpu_count()):
    with multiprocessing.Pool(n_cores) as pool:
        df = pd.concat(pool.map(func, np.array_split(df, n_cores*1000))).groupby('userId').sum().reset_index()
    return df


def get_features_and_edgelist(users_df, relations_df):
    # Feature size = {hourly(24) * daily(2) usage}  + 10 demographic = 58
    user_demo_features = pd.get_dummies(users_df.iloc[:, :-1], columns=['sex', 'ageGroup'])
    user_activity_features = parallelize_feature_extraction(relations_df, extract_activity_features)
    user_features = user_demo_features.merge(user_activity_features, how='left', on='userId').fillna(0)
    logging.info("Shape of user features: {}".format(user_features.values[:, 1:].shape))
    # save extracted user features to a csv file
    user_features.to_csv(os.path.join(args.output_dir, 'user_features.csv'), index=False, header=False)
    logging.info("Wrote user features to file: {}".format(os.path.join(args.output_dir, 'user_features.csv')))

    # extract edges
    user_edges = {}
    for relation, frame in relations_df[['src', 'dst', 'relation']].groupby('relation'):
        edgelist = frame[['src', 'dst']].drop_duplicates()
        edgelist.to_csv(os.path.join(args.output_dir, 'relation{}_edgelist.csv').format(relation), index=False,
                        header=False)
        logging.info("Wrote relation edgelist to: {}".format(os.path.join(args.output_dir,
                                                                          'relation{}_edgelist.csv').format(relation)))
        user_edges[relation] = edgelist

    return user_features, user_edges


def create_homogeneous_edgelist(user_edges):
    homogeneous_edgelist = pd.concat(user_edges.values()).drop_duplicates()
    homogeneous_edgelist.to_csv(os.path.join(args.output_dir, 'homogeneous_edgelist.csv'), index=False, header=False)
    logging.info("Wrote homogeneous edgelist to file: {}".format(os.path.join(args.output_dir,
                                                                              'homogeneous_edgelist.csv')))
    return homogeneous_edgelist


if __name__ == '__main__':
    logging = get_logger(__name__)

    args = parse_args()

    users_df, relations_df = load_data()

    # train_days > 0 implies we need to create test users since this is for an experiment
    # for live predictions we should already have a list of new users to predict for
    if args.train_days:
        test_users = get_test_users(users_df, relations_df, args.train_days)

    edge_types = relations_df.relation.unique()
    logging.info("Found the following distinct relation types: {}".format(edge_types))

    user_features, user_edges = get_features_and_edgelist(users_df, relations_df)

    if args.construct_homogeneous:
        user_user_edges = create_homogeneous_edgelist(user_edges)

    tag_df = users_df[['userId', 'label']]
    tag_df.to_csv(os.path.join(args.output_dir, 'tags.csv'), index=False)
    logging.info("Wrote labels to file: {}".format(os.path.join(args.output_dir, 'tags.csv')))

