import os
import time
import pickle
import argparse
import json
import logging
import numpy as np
import mxnet as mx
from mxnet import nd, gluon, autograd
from scipy.sparse import load_npz


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=200000)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n-epochs', type=int, default=20)
    parser.add_argument('--n-hidden', type=int, default=64, help='number of hidden units')
    parser.add_argument('--n-layers', type=int, default=5, help='number of hidden layers')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight for L2 loss')

    return parser.parse_args()


def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    return logger


def get_data():
    filename = 'mlp-fraud-dataset.npz'
    matrix = load_npz(os.path.join(args.training_dir, filename)).toarray().astype('float32')
    scale_pos_weight = np.sqrt((matrix.shape[0] - matrix[:, 0].sum()) / matrix[:, 0].sum())
    weight_mask = np.ones(matrix.shape[0]).astype('float32')
    weight_mask[np.where(matrix[:, 0])] = scale_pos_weight
    dataloader = gluon.data.DataLoader(gluon.data.ArrayDataset(matrix[:, 1:], matrix[:, 0], weight_mask),
                                       args.batch_size,
                                       shuffle=True,
                                       last_batch='keep')
    return dataloader, matrix.shape[0]


def evaluate(model, dataloader, ctx):
    f1 = mx.metric.F1()
    for data, label, mask in dataloader:
        pred = model(data.as_in_context(ctx))
        f1.update(label.as_in_context(ctx), nd.softmax(pred, axis=1))
    return f1.get()[1]


def train(model, trainer, loss, train_data, ctx):
    duration = []
    for epoch in range(args.n_epochs):
        tic = time.time()
        loss_val = 0.

        for features, labels, weight_mask in train_data:
            with autograd.record():
                pred = model(features.as_in_context(ctx))
                l = loss(pred, labels.as_in_context(ctx), mx.nd.expand_dims(weight_mask, 1).as_in_context(ctx)).sum()
            l.backward()
            trainer.step(args.batch_size)

        loss_val += l.asscalar()
        duration.append(time.time() - tic)
        f1 = evaluate(model, train_data, ctx)
        logging.info("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | F1 Score {:.4f} ".format(
            epoch, np.mean(duration), loss_val / n, f1))
    save_model(model)
    return model


def save_model(model):
    model.save_parameters(os.path.join(args.model_dir, 'model.params'))
    with open(os.path.join(args.model_dir, 'model_hyperparams.pkl'), 'wb') as f:
        pickle.dump(args, f)


def get_model(model_dir, ctx, n_classes=2, load_stored=False):
    if load_stored:  # load using saved model state
        with open(os.path.join(model_dir, 'model_hyperparams.pkl'), 'rb') as f:
            hyperparams = pickle.load(f)
    else:
        hyperparams = args

    model = gluon.nn.Sequential()
    for _ in range(hyperparams.n_layers):
        model.add(gluon.nn.Dense(hyperparams.n_hidden, activation='relu'))
    model.add(gluon.nn.Dense(n_classes))

    if load_stored:
        model.load_parameters(os.path.join(model_dir, 'model.params'), ctx=ctx)
    else:
        model.initialize(ctx=ctx)
    return model


if __name__ == '__main__':
    logging = get_logger(__name__)
    logging.info('numpy version:{} MXNet version:{}'.format(np.__version__, mx.__version__))

    args = parse_args()

    train_data, n = get_data()

    ctx = mx.gpu(0) if args.num_gpus else mx.cpu(0)

    model = get_model(args.model_dir, ctx, n_classes=2)

    logging.info(model)
    logging.info(model.collect_params())

    loss = gluon.loss.SoftmaxCELoss()
    trainer = gluon.Trainer(model.collect_params(), args.optimizer, {'learning_rate': args.lr, 'wd': args.weight_decay})

    logging.info("Starting Model training")
    model = train(model, trainer, loss, train_data, ctx)


# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    ctx = mx.gpu(0) if list(mx.test_utils.list_gpus()) else mx.cpu(0)
    net = get_model(model_dir, ctx, n_classes=2, load_stored=True)
    return net


def transform_fn(net, data, input_content_type, output_content_type):
    ctx = mx.gpu(0) if list(mx.test_utils.list_gpus()) else mx.cpu(0)
    nda = mx.nd.array(json.loads(data))
    prediction = nd.softmax(net(nda.as_in_context(ctx)), axis=1)[:, 1]
    response_body = json.dumps(prediction.asnumpy().tolist())
    return response_body, output_content_type
