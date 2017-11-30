import argparse
import random

import chainer
import numpy

import compressed_image_recoginition


def main(args: argparse.Namespace):
    train_mnist_dataset, test_mnist_dataset = chainer.datasets.get_mnist(withlabel=True, ndim=2)

    model = compressed_image_recoginition.models.load_model(args)

    chainer.serializers.load_npz(args.model_weight, model)

    img, label = train_mnist_dataset[0]
    data = compressed_image_recoginition.datasets.MnistCompressedBinaryDataset.convert(
        img,
        format=args.format
    )
    print(data.shape)

    predicted = model(numpy.array([data]))
    print("true label: {}".format(label))
    print(predicted.data)
    compressed_image_recoginition.datasets.MnistCompressedBinaryDataset.save_formatted(data, "test.jpg")

    i = random.randint(0, data.shape[0])
    data[i] = 0
    predicted = model(numpy.array([data]))
    compressed_image_recoginition.datasets.MnistCompressedBinaryDataset.save_formatted(data, "test2.jpg")
    print(predicted.data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_weight")
    parser.add_argument("--format", type=str, default='JPEG')
    parser.add_argument("--gpu", type=int, default=-1)
    compressed_image_recoginition.models.model_args_parser(parser)
    args = parser.parse_args()
    main(args)
