# -*- coding: utf-8 -*-
import chainer
import argparse
import os
import time
import json
from chainer.training import extensions

ROOT_PATH = os.path.dirname(__file__)
import compressed_image_recoginition


def save_json(filename, obj):
    json.dump(obj, open(filename, 'w'), sort_keys=True, indent=4)


parser = argparse.ArgumentParser()
parser.add_argument("--format", type=str, default='JPEG')
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--units", type=int, default=100)
parser.add_argument("--lstm_layers", type=int, default=2)
args = parser.parse_args()

output_directory = os.path.join(
    ROOT_PATH,
    "output",
    "{format}_unit{unit}_layer{layer}_{time}".format(
        format=args.format, unit=args.units, time=int(time.time()), layer=args.lstm_layers))
os.makedirs(output_directory)

train_mnist_dataset, test_mnist_dataset = chainer.datasets.get_mnist(withlabel=True, ndim=2)
dataset = compressed_image_recoginition.datasets.MnistCompressedBinaryDataset(base_dataset=train_mnist_dataset,
                                                                              image_format=args.format)
# iterator = chainer.iterators.SerialIterator(dataset=dataset, batch_size=1)
iterator = chainer.iterators.MultiprocessIterator(dataset=dataset, batch_size=1, shuffle=True, n_processes=16)

test_dataset = compressed_image_recoginition.datasets.HeadDataset(
    head=50,
    base_dataset=compressed_image_recoginition.datasets.MnistCompressedBinaryDataset(base_dataset=test_mnist_dataset)
)

test_iterator = chainer.iterators.SerialIterator(dataset=test_dataset, batch_size=1, repeat=False)


def loss(binary, label):
    calculated_loss = chainer.functions.softmax_cross_entropy(model(binary), label)
    chainer.reporter.report({'loss': calculated_loss}, model)
    return calculated_loss


def evaluation(binary, label):
    predicted = model(binary)
    calculated_loss = chainer.functions.softmax_cross_entropy(predicted, label)
    chainer.reporter.report({'loss': calculated_loss}, model)
    accuracy = chainer.functions.accuracy(predicted, label)
    chainer.reporter.report({'accuracy': accuracy}, model)


optimizer = chainer.optimizers.Adam()
model = compressed_image_recoginition.models.Model(vocab_size=256, midsize=args.units, output_dimention=10,
                                                   num_lstm_layer=args.lstm_layers)
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make the GPU current
    model.to_gpu()
optimizer.setup(model)

updater = chainer.training.StandardUpdater(iterator, optimizer, device=args.gpu, loss_func=loss)
trainer = chainer.training.Trainer(updater, (10, 'epoch'), out=output_directory)

save_json(os.path.join(trainer.out, 'argument.json'), args.__dict__)

save_interval = (10000, 'iteration')
evaluation_interval = (1000, 'iteration')
trainer.extend(extensions.snapshot_object(model, '{.updater.iteration}.model'), trigger=save_interval)

trainer.extend(extensions.LogReport(trigger=(10, 'iteration'), log_name="log.txt"))
trainer.extend(
    chainer.training.extensions.PrintReport(
        ['epoch', 'iteration', "validation/main/accuracy", 'main/loss', "validation/main/loss", "elapsed_time"]),
    trigger=evaluation_interval)

trainer.extend(extensions.ProgressBar(update_interval=1))

trainer.extend(extensions.Evaluator(test_iterator, target=model, eval_func=evaluation, device=args.gpu),
               trigger=(10, 'iteration'))

trainer.run()
