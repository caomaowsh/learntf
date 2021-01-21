"""
tensorflow: 1.12.0
python: 3.6.9
"""

import tensorflow as tf
import numpy as np 

tf.set_random_seed(1)
np.random.seed(1)

x = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise

with tf.variable_scope('Inputs'):
	tf_x = tf.placeholder(tf.float32, x.shape, name='x')
	tf_y = tf.placeholder(tf.float32, x.shape, name='y')
with tf.variable_scope('Net'):
	h1 = tf.layers.dense(tf_x, 10, tf.nn.relu, name='hidden layer')
	output = tf.layers.dense(h1, 1, name='output layer')

	## add layer output to histogram summary
	tf.summary.histogram('h1_out', h1)
	tf.summary.histogram('pred', output)

loss = tf.losses.mean_squared_error(tf_y, output, scope='loss')
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
## add loss to scalar summary

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	## add the graph to log file
	writer = tf.summary.FileWriter('./log', sess.graph)
	merge_op = tf.summary.merge_all()

	for step in range(100):
		_, result = sess.run([train_op, merge_op], feed_dict={tf_x: x, tf_y: y})
		## record the result every step
		writer.add_summary(result, step)


# Lastly, in your terminal or CMD, type this :
# $ tensorboard --logdir path/to/log
# open you google chrome, type the link shown on your terminal or CMD. (something like this: http://localhost:6006)