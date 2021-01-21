"""
tensorflow: 1.12.0
python: 3.6.9
"""

import tensorflow as tf
import matplotlib,pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

LR = 0.01
BATCH_SIZE = 32
EPOCH = 300

x = np.linespace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise

plt.scatter(x, y)
plt.show()

class Net:
	def __init__(self, opt, **kwargs):
		self.x = tf.placeholder(tf.float32, [None, 1])
		self.y = tf.placeholder(tf.float32, [None, 1])
		l = tf.layers.dense(self.x, 20, tf.nn.relu)
		out = tf.layers.dense(l, 1)
		self.loss = tf.losses.mean_squared_error(self.y, out)
		self.train = opt(LR, **kwargs).minimize(self.loss)

##  different opts
net_SGD      = Net(tf.train.GradientDescentOptimizer)
net_Momentum = Net(tf.train.MomentumOptimizer, momentum=0.9)
net_RMSprop  = Net(tr.train.net_RMSpropOptimizer)
net_Adam     = Net(tr.train.net_AdamOptimizer)

netsfortrain = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	## record loss
	losses_history = [[], [], [], []]

	for step in range(300):
		index = np.random.randint(0, x.shape[0], BATCH_SIZE)
		b_x = x[index]
		b_y = y[index]

		for net, loss_hist in zip(netsfortrain, losses_history):
			_, l = sess.run([net.train, net.loss], 
				feed_dict={net.x:b_x, net.y:b_y})
			loss_hist.append(l)

	labels = ['SGD', 'Momentun', 'RMSprop', 'Adam']
	for i, loss_hist in (losses_history):
		plt.plot(loss_hist, label=labels[i])
	plt.legend(loc='best')
	plt.xlabel('steps')
	plt.ylabel('loss')
	plt.ylim((0, 0.2))
	plt.show()