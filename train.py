# -*- coding: utf-8 -*-
import chainer
import argparse
import os
import time
import json
import random
import numpy
from chainer.training import extensions

ROOT_PATH = os.path.dirname(__file__)
import compressed_image_recoginition


def save_json(filename, obj):
    json.dump(obj, open(filename, 'w'), sort_keys=True, indent=4)


parser = argparse.ArgumentParser()
parser.add_argument("--format", type=str, default='JPEG')
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--batch", type=int, default=1000)
parser.add_argument("--noise_ratio", type=float, default=0.0)
parser.add_argument("--augment", action="store_true")
compressed_image_recoginition.models.model_args_parser(parser)
args = parser.parse_args()

output_directory = os.path.join(
    ROOT_PATH,
    "output",
    "{format}_{model}_unit{unit}_layer{layer}{cnnlayer}_batch{batch}{bn}{augment}{noise}_{time}".format(
        model=args.model, format=args.format, unit=args.units, batch=args.batch, time=int(time.time()),
        cnnlayer=("_{}".format(args.cnn_layers) if args.model == "convlstm" else ""),
        noise=("_noise{}".format(args.noise_ratio) if args.noise_ratio >0 else ""),
        bn=("_bn" if args.bn else ""), augment=("_aug" if args.augment else ""), layer=args.lstm_layers))
os.makedirs(output_directory)


def augmenation(datum):
    image, label = datum
    random_shift = (
        random.choice(list(range(-3, 4, 1))),
        random.choice(list(range(-3, 4, 1)))
    )
    shifted = numpy.zeros(image.shape, dtype=image.dtype)
    top = max(0, random_shift[0])
    bottom = min(image.shape[0], image.shape[0] + random_shift[0])
    left = max(0, random_shift[1])
    right = min(image.shape[1], image.shape[1] + random_shift[1])

    target_top = max(0, -random_shift[0])
    target_bottom = min(image.shape[0], image.shape[0] - random_shift[0])
    target_left = max(0, -random_shift[1])
    target_right = min(image.shape[1], image.shape[1] - random_shift[1])

    shifted[target_top:target_bottom, target_left:target_right] = image[top:bottom, left:right]


    rand_source = numpy.random.random(shifted.shape)
    shifted[rand_source < args.noise_ratio] = shifted.max()

    return shifted, label

train_mnist_dataset, test_mnist_dataset = chainer.datasets.get_mnist(withlabel=True, ndim=2)
if args.augment:
    train_mnist_dataset = chainer.datasets.TransformDataset(train_mnist_dataset, augmenation)

dataset = compressed_image_recoginition.datasets.MnistCompressedBinaryDataset(base_dataset=train_mnist_dataset,
                                                                              image_format=args.format)
dataset = compressed_image_recoginition.datasets.PaddedDataset(dataset)
iterator = chainer.iterators.MultiprocessIterator(dataset=dataset, batch_size=args.batch, shuffle=True, n_processes=16,
                                                  shared_mem=100000, n_prefetch=10)
test_dataset = compressed_image_recoginition.datasets.HeadDataset(
    head=50,
    base_dataset=compressed_image_recoginition.datasets.MnistCompressedBinaryDataset(base_dataset=test_mnist_dataset)
)
test_dataset = compressed_image_recoginition.datasets.PaddedDataset(test_dataset, max_length=dataset.max_length)

test_iterator = chainer.iterators.SerialIterator(dataset=test_dataset, batch_size=args.batch, repeat=False)


def loss(binary, label):
    predicted = model(binary)
    calculated_loss = chainer.functions.softmax_cross_entropy(predicted, label)
    chainer.reporter.report({'loss': calculated_loss}, model)
    accuracy = chainer.functions.accuracy(predicted, label)
    chainer.reporter.report({'accuracy': accuracy}, model)
    return calculated_loss


def evaluation(binary, label):
    predicted = model(binary)
    calculated_loss = chainer.functions.softmax_cross_entropy(predicted, label)
    chainer.reporter.report({'loss': calculated_loss}, model)
    accuracy = chainer.functions.accuracy(predicted, label)
    chainer.reporter.report({'accuracy': accuracy}, model)


optimizer = chainer.optimizers.Adam()


model = compressed_image_recoginition.models.load_model(args)
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make the GPU current
    model.to_gpu()
optimizer.setup(model)

updater = chainer.training.StandardUpdater(iterator, optimizer, device=args.gpu, loss_func=loss)
trainer = chainer.training.Trainer(updater, (10000, 'epoch'), out=output_directory)

save_json(os.path.join(trainer.out, 'argument.json'), args.__dict__)

save_interval = (1000, 'iteration')
evaluation_interval = (100, 'iteration')
trainer.extend(extensions.snapshot_object(model, '{.updater.iteration}.model'), trigger=save_interval)

trainer.extend(extensions.LogReport(trigger=(10, 'iteration'), log_name="log.txt"))
trainer.extend(
    chainer.training.extensions.PrintReport(
        ['epoch', 'iteration', "main/accuracy", 'main/loss', "validation/main/accuracy", "validation/main/loss",
         "elapsed_time"]),
    trigger=evaluation_interval)

trainer.extend(extensions.ProgressBar(update_interval=1))

trainer.extend(extensions.Evaluator(test_iterator, target=model, eval_func=evaluation, device=args.gpu),
               trigger=evaluation_interval)

trainer.run()
