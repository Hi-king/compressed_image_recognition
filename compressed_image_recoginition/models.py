# -*- coding: utf-8 -*-
import chainer


class Model(chainer.Chain):
    def __init__(self, vocab_size, midsize, output_dimention):
        super().__init__(
            word_embed=chainer.functions.EmbedID(vocab_size, midsize),
            lstm0=chainer.links.lstm.LSTM(midsize, midsize),
            lstm1=chainer.links.lstm.LSTM(midsize, midsize),
            out_layer=chainer.functions.Linear(midsize, output_dimention)
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
        if hasattr(self, "lstm0"):
            h = self.lstm0(h)
        if hasattr(self, "lstm1"):
            h = self.lstm1(h)
        if hasattr(self, "lstm2"):
            h = self.lstm2(h)
        return h

    def reset_state(self):
        if hasattr(self, "lstm0"):
            self.lstm0.reset_state()
        if hasattr(self, "lstm1"):
            self.lstm1.reset_state()
        if hasattr(self, "lstm2"):
            self.lstm2.reset_state()
