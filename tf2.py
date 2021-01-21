'''
tensorflow: 1.12.0
python: 3.6.9
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)

x_relu = tf.nn.relu(x)
x_sigm = tf.nn.sigmoid(x)
x_tanh = tf.nn.tanh(x)
x_softplus = tf.nn.softplus(x)
x_softmax = tf.nn.softmax(x)

with tf.Session() as sess:
	x_relu, x_sigm, x_tanh, x_softplus = sess.run([
		x_relu, x_sigm, x_tanh, x_softplus])

plt.figure(1, figsize=(8,6))

plt.subplot(221)
plt.plot(x, x_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x, x_sigm, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x, x_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x, x_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()