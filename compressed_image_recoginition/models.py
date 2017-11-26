# -*- coding: utf-8 -*-
import chainer


class Model(chainer.Chain):
    def __init__(self, vocab_size, midsize, output_dimention, num_lstm_layer):
        self.lstm_layer_keys = ["rnn{}".format(i) for i in range(num_lstm_layer)]
        self.lstm_layers = {key: chainer.links.LSTM(midsize, midsize) for key in self.lstm_layer_keys}
        super().__init__(
            word_embed=chainer.links.EmbedID(vocab_size, midsize),
            out_layer=chainer.links.Linear(midsize, output_dimention),
            **self.lstm_layers
        )

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """
        :param x: [batch, bytes]
        """
        self.reset_state()
        outputs = []
        for each_data in chainer.functions.separate(x):
            for byte in chainer.functions.separate(each_data):
                byte = chainer.functions.expand_dims(byte, axis=0)
                lstm_out = self._forward_lstms(byte)
            last_output = self.out_layer(lstm_out)
            outputs.append(last_output)
        return chainer.functions.concat(outputs)

    def _forward_lstms(self, x):
        h = self.word_embed(x)
        for key in self.lstm_layer_keys:
            h = getattr(self, key)(h)
        return h

    def reset_state(self):
        for key in self.lstm_layer_keys:
            getattr(self, key).reset_state()
