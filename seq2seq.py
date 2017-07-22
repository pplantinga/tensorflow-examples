"""
Simple example of sorting

Author: Peter Plantinga
Date: Summer 2017
"""

import tensorflow as tf
from seq2seq_example import seq2seq_example

epochs = 50

s2s = seq2seq_example()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epochs):

        train_cost = 0
        for batch in s2s.batchify():
            train_cost += sess.run([s2s.train_op, s2s.cost], batch)[1]

        train_cost *= s2s.batch_size / s2s.samples

        error = sess.run(s2s.error_rate, s2s.test_batch())

        print("Epoch ", (i+1), " train loss: ", train_cost, "test error: ", error)
