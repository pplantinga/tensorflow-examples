"""
Simple example of sorting using ctc

Author: Peter Plantinga
Date: Summer 2017
"""

import tensorflow as tf
from ctc_example import ctc_example

epochs = 50

ctc = ctc_example()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epochs):

        train_cost = 0
        for batch in ctc.batchify():
            train_cost += sess.run([ctc.train_op, ctc.cost], batch)[1]

        train_cost *= ctc.batch_size / ctc.samples

        error = sess.run(ctc.error_rate, ctc.test_batch())

        print("Epoch ", (i+1), " train loss: ", train_cost, "test error: ", error)
