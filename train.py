# -*- coding: utf-8 -*-
import chainer
import argparse
import os
import time
import json
from chainer.training import extensions

ROOT_PATH = os.path.dirname(__file__)
import compressed_image_recoginition

parser = argparse.ArgumentParser()
parser.add_argument("--format", type=str, default='JPEG')
parser.add_argument("--gpu", type=int, default=-1)
args = parser.parse_args()

train_mnist_dataset, test_mnist_dataset = chainer.datasets.get_mnist(withlabel=True, ndim=2)
dataset = compressed_image_recoginition.datasets.MnistCompressedBinaryDataset(base_dataset=train_mnist_dataset,
                                                                              image_format=args.format)
# iterator = chainer.iterators.SerialIterator(dataset=dataset, batch_size=1)
iterator = chainer.iterators.MultiprocessIterator(dataset=dataset, batch_size=1, shuffle=True)

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
model = compressed_image_recoginition.models.Model(vocab_size=256, midsize=100, output_dimention=10)
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make the GPU current
    model.to_gpu()
optimizer.setup(model)

updater = chainer.training.StandardUpdater(iterator, optimizer, device=args.gpu, loss_func=loss)
trainer = chainer.training.Trainer(updater, (10, 'epoch'),
                                   out=os.path.join(
                                       ROOT_PATH, "output",
                                       "{format}_{time}".format(format=args.format, time=int(time.time()))))


def save_json(filename, obj):
    json.dump(obj, open(filename, 'w'), sort_keys=True, indent=4)


trainer.extend(extensions.snapshot_object(args.__dict__, 'argument.json', savefun=save_json),
               invoke_before_training=True)

save_interval = (1000, 'iteration')
trainer.extend(extensions.snapshot_object(model, '{.updater.iteration}.model'), trigger=save_interval)

trainer.extend(extensions.LogReport(trigger=(10, 'iteration'), log_name="log.txt"))

trainer.extend(extensions.ProgressBar(update_interval=1))

trainer.extend(extensions.Evaluator(test_iterator, target=model, eval_func=evaluation, device=args.gpu),
               trigger=(1000, 'iteration'))

trainer.run()
