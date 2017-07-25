"""
Example of ctc in TensorFlow 1.2

Sorts a random list of integers

Author: Peter Plantinga
Date: Summer 2017
"""

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
from random import shuffle

class ctc_example:

    tokens = {"PAD": 0, "EOS": 1, "GO": 2, "UNK": 3} 

    minLength = 5
    maxLength = 10
    samples = 10000
    vocab_size = 50
    embedding_size = 15
    dropout = 0.3
    layers = 2
    layer_size = 100
    batch_size = 50

    def __init__(self):

        # Random integers up to vocab size (not including reserved values)
        self.data = np.random.randint(
                low  = len(self.tokens),
                high = self.vocab_size,
                size = (self.samples, self.maxLength + self.minLength))

        # Random length for each sequence from minLength to maxLength
        self.dataLens = np.random.randint(
                low  = self.minLength,
                high = self.maxLength,
                size = self.samples)

        # Create labels by sorting data
        self.dataLabels = np.zeros([self.samples, self.maxLength])
        for i in range(len(self.data)):
            self.data[i, self.dataLens[i]:] = self.tokens['PAD']
            self.dataLabels[i, :self.dataLens[i]] = np.sort(self.data[i, :self.dataLens[i]])
       
        # Make placeholders and stuff
        self.make_inputs()

        # Build computation graph
        self.build_graph()

    def make_inputs(self):
        self.input     = tf.placeholder(tf.int32, (self.batch_size, self.maxLength + self.minLength))
        self.lengths   = tf.placeholder(tf.int32, (self.batch_size,))
        self.labels    = tf.placeholder(tf.int32, (self.batch_size, self.maxLength))
        self.keep_prob = tf.placeholder(tf.float32)

        # Embed input
        self.embedded_input = tf.contrib.layers.embed_sequence(
                ids        = self.input,
                vocab_size = self.vocab_size,
                embed_dim  = self.embedding_size)

        # Time-major
        #self.embedded_input = tf.transpose(self.embedded_input)

    def single_layer_cell(self):
        return rnn.DropoutWrapper(rnn.LSTMCell(self.layer_size), self.keep_prob)

    def cell(self):
        return rnn.MultiRNNCell([self.single_layer_cell() for _ in range(self.layers)])

    def build_graph(self):
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw         = self.cell(),
                cell_bw         = self.cell(),
                inputs          = self.embedded_input,
                sequence_length = self.lengths + self.minLength,
                dtype           = tf.float32)
                #time_major      = True)

        # Concatenate fw and bw outputs, then reshape
        outputs = tf.concat(outputs, 2)
        outputs = tf.reshape(outputs, [-1, self.layer_size * 2])

        # Output layer
        W = tf.Variable(tf.truncated_normal([self.layer_size * 2, self.vocab_size + 1], stddev=0.1))
        b = tf.Variable(tf.zeros(self.vocab_size + 1))
        logits = tf.matmul(outputs, W) + b
        logits = tf.reshape(logits, [self.batch_size, self.maxLength + self.minLength, self.vocab_size + 1])
        logits = tf.transpose(logits, [1, 0, 2])

        # CTC layer
        sparse_labels = self.to_sparse(self.labels, self.lengths)
        self.cost = tf.reduce_mean(tf.nn.ctc_loss(
                labels          = sparse_labels,
                inputs          = logits,
                sequence_length = self.lengths + self.minLength,
                time_major      = True))
        self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.cost)


        # Decoder
        decoded, _ = tf.nn.ctc_beam_search_decoder(
                inputs = logits,
                sequence_length = self.lengths,
                beam_width = 4)
        self.error_rate = tf.reduce_mean(tf.edit_distance(sparse_labels, tf.cast(decoded[0], tf.int32)))


    def to_sparse(self, tensor, lengths):
        mask = tf.sequence_mask(lengths, self.maxLength)
        indices = tf.to_int64(tf.where(tf.equal(mask, True)))
        values = tf.to_int32(tf.boolean_mask(tensor, mask))
        shape = tf.to_int64(tf.shape(tensor))
        return tf.SparseTensor(indices, values, shape)

    def next_batch(self, i):

        start = i * self.batch_size
        stop = (i+1) * self.batch_size

        batch = {
                self.input:     self.data[start:stop],
                self.lengths:   self.dataLens[start:stop],
                self.labels:    self.dataLabels[start:stop],
                self.keep_prob: 1. - self.dropout
        }

        return batch

    def batchify(self):
        
        # Shuffle data
        a = list(zip(self.data, self.dataLens, self.dataLabels))
        shuffle(a)
        self.data, self.dataLens, self.dataLabels = zip(*a)

        for i in range(self.samples // self.batch_size):
            yield self.next_batch(i)

    def test_batch(self):

        data = np.random.randint(
                low  = len(self.tokens),
                high = self.vocab_size,
                size = (self.batch_size, self.maxLength + self.minLength))

        dataLens = np.random.randint(
                low  = self.minLength,
                high = self.maxLength,
                size = self.batch_size)

        dataLabels = np.zeros([self.batch_size, self.maxLength])
        for i in range(len(data)):
            data[i, dataLens[i]:] = self.tokens['PAD']
            dataLabels[i, :dataLens[i]] = np.sort(data[i, :dataLens[i]])

        return {
                self.input: data,
                self.lengths: dataLens,
                self.labels: dataLabels,
                self.keep_prob: 1.
        }
