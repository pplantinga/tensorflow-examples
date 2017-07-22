"""
Example of seq2seq in TensorFlow 1.2

Sorts a random list of integers

Author: Peter Plantinga
Date: Summer 2017
"""

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers.core import Dense
import numpy as np
from random import shuffle

class seq2seq_example:

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
                size = (self.samples, self.maxLength))

        # Random length for each sequence from minLength to maxLength
        self.dataLens = np.random.randint(
                low  = self.minLength,
                high = self.maxLength,
                size = self.samples)

        # Create labels by sorting data
        self.dataLabels = np.zeros_like(self.data)
        for i in range(len(self.data)):
            self.data[i, self.dataLens[i]:] = self.tokens['PAD']
            self.dataLabels[i, :self.dataLens[i]] = np.sort(self.data[i, :self.dataLens[i]])
       
        # Make placeholders and stuff
        self.make_inputs()

        # Build computation graph
        self.build_graph()

    def make_inputs(self):
        self.input     = tf.placeholder(tf.int32, (self.batch_size, self.maxLength))
        self.lengths   = tf.placeholder(tf.int32, (self.batch_size,))
        self.labels    = tf.placeholder(tf.int32, (self.batch_size, self.maxLength))
        self.keep_prob = tf.placeholder(tf.float32)

        # Embed encoder input
        self.enc_input = tf.contrib.layers.embed_sequence(
                ids        = self.input,
                vocab_size = self.vocab_size,
                embed_dim  = self.embedding_size)

        # Create decoder input (GO + label + EOS)
        eos = tf.one_hot(
                indices  = self.lengths,
                depth    = self.maxLength,
                on_value = self.tokens['EOS'])

        self.add_eos = self.labels + eos
        go_tokens = tf.constant(self.tokens['GO'], shape=[self.batch_size, 1])
        pre_embed_dec_input = tf.concat((go_tokens, self.add_eos), 1)

        # Embed decoder input
        self.dec_embed = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size]))
        self.dec_input = tf.nn.embedding_lookup(self.dec_embed, pre_embed_dec_input)

    def single_layer_cell(self):
        return rnn.DropoutWrapper(rnn.LSTMCell(self.layer_size), self.keep_prob)

    def cell(self):
        return rnn.MultiRNNCell([self.single_layer_cell() for _ in range(self.layers)])

    def build_graph(self):
        _, enc_state = tf.nn.dynamic_rnn(
                cell            = self.cell(),
                inputs          = self.enc_input,
                sequence_length = self.lengths,
                dtype           = tf.float32)

        # Replicate the top-most encoder state for starting state of all layers in the decoder
        dec_start_state = tuple(enc_state[-1] for _ in range(self.layers))

        output = Dense(self.vocab_size,
                kernel_initializer = tf.truncated_normal_initializer(stddev=0.1))
        
        # Training decoder: scheduled sampling et al.
        with tf.variable_scope("decode"):
            train_helper = seq2seq.ScheduledEmbeddingTrainingHelper(
                    inputs               = self.dec_input,
                    sequence_length      = self.lengths,
                    embedding            = self.dec_embed,
                    sampling_probability = 0.1)

            train_decoder = seq2seq.BasicDecoder(
                    cell          = self.cell(),
                    helper        = train_helper,
                    initial_state = dec_start_state,
                    output_layer  = output)
            
            train_output, _, train_lengths = seq2seq.dynamic_decode(
                    decoder            = train_decoder,
                    maximum_iterations = self.maxLength)

        beam_size = 4
        tiled = seq2seq.tile_batch(dec_start_state, beam_size)

        with tf.variable_scope("decode", reuse=True):
            test_decoder = seq2seq.BeamSearchDecoder(
                    cell          = self.cell(),
                    embedding     = self.dec_embed,
                    start_tokens  = tf.ones_like(self.lengths) * self.tokens['GO'],
                    end_token     = self.tokens['EOS'],
                    initial_state = tiled,
                    beam_width    = beam_size,
                    output_layer  = output)
            test_output, _, test_lengths = seq2seq.dynamic_decode(
                    decoder            = test_decoder,
                    maximum_iterations = self.maxLength)

        # Create train op
        mask = tf.sequence_mask(train_lengths + 1, self.maxLength - 1, dtype=tf.float32)
        self.cost = seq2seq.sequence_loss(train_output.rnn_output, self.add_eos[:, :-1], mask)
        self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.cost)

        # Create test error rate op
        predicts = self.to_sparse(test_output.predicted_ids[:,:,0], test_lengths[:, 0] - 1)
        labels = self.to_sparse(self.add_eos, self.lengths)
        self.error_rate = tf.reduce_mean(tf.edit_distance(predicts, labels))

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
                size = (self.batch_size, self.maxLength))

        dataLens = np.random.randint(
                low  = self.minLength,
                high = self.maxLength,
                size = self.batch_size)

        dataLabels = np.zeros_like(data)
        for i in range(len(data)):
            data[i, dataLens[i]:] = self.tokens['PAD']
            dataLabels[i, :dataLens[i]] = np.sort(data[i, :dataLens[i]])

        return {
                self.input: data,
                self.lengths: dataLens,
                self.labels: dataLabels,
                self.keep_prob: 1.
        }
